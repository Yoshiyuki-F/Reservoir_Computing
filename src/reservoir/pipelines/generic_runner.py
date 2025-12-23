"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
Refactored to delegate reporting/logging to reservoir.utils.reporting.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from reservoir.pipelines.config import ModelStack, FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig
from reservoir.core.identifiers import TaskType
from reservoir.utils.reporting import (
    calculate_chaos_metrics,
    print_ridge_search_results,
    print_feature_stats,
    compute_score
)
from reservoir.utils.batched_compute import batched_compute

class UniversalPipeline:
    """Runs the V2 flow: pre-train model -> extract features -> fit ridge -> evaluate."""

    def __init__(self, stack: ModelStack, config: PipelineConfig) -> None:
        self.model = stack.model
        self.readout = stack.readout
        self.metric_name = stack.metric
        self.config = config

    # ------------------------------------------------------------------ #
    # Utilities (Computation only)                                       #
    # ------------------------------------------------------------------ #

    def generate_closed_loop(
        self,
        initial_input: jnp.ndarray,
        steps: int,
        projection_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        verbose: bool = True
    ) -> jnp.ndarray:
        """Generate closed-loop predictions using Fast JAX Scan."""
        history = jnp.asarray(initial_input)
        if history.ndim == 2: history = history[None, ...]
        elif history.ndim == 1: history = history[None, None, ...]

        batch_size = history.shape[0]

        if verbose:
            print(f"[Closed-Loop] Generating {steps} steps (Fast JAX Scan)...")

        initial_state = self.model.initialize_state(batch_size)
        final_state, _ = self.model.forward(initial_state, history)

        def predict_one(s):
            s_in = s[:, None, :]
            if self.readout is not None:
                out = self.readout.predict(s_in)
                return out[:, 0, :]
            else:
                 return s

        first_prediction = predict_one(final_state)

        def scan_step(carry, _):
            h_prev, x_raw = carry
            x_proj = projection_fn(x_raw) if projection_fn else x_raw
            h_next, _ = self.model.step(h_prev, x_proj)
            y_next = predict_one(h_next)
            return (h_next, y_next), y_next

        _, predictions = jax.lax.scan(scan_step, (final_state, first_prediction), None, length=steps)

        return jnp.swapaxes(predictions, 0, 1)

    # ------------------------------------------------------------------ #
    # Run                                                                #
    # ------------------------------------------------------------------ #
    def run(
        self,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
    ) -> Dict[str, Dict[str, Any]]:
        processed = frontend_ctx.processed_split
        train_X = processed.train_X
        train_y = processed.train_y
        test_X = processed.test_X
        test_y = processed.test_y

        cfg = dataset_meta.training
        ridge_lambdas = cfg.ridge_lambdas
        feature_batch_size = int(cfg.batch_size)

        start = time.time()

        # Initialize Output Variables
        closed_loop_pred_val = None
        closed_loop_truth_val = None
        best_lambda = None
        best_score = None
        search_history = {}
        weight_norms = {}

        # 1. Train Model (Reservoir Warmup)
        print(f"\n=== Step 5: Model Dynamics (Training/Warmup) [{self.config.model_type.value}] ===")
        start_train = time.time()
        train_logs = self.model.train(train_X, train_y) or {}
        train_time = time.time() - start_train
        print(f"[Step 5] Model Dynamics completed in {train_time:.2f}s.")

        # 2. Extract Features
        print("\n=== Step 6: Feature Extraction / Aggregation ===")
        train_features = batched_compute(self.model, train_X, feature_batch_size, desc="[Extracting] train")
        print_feature_stats(train_features, "post_train_features")

        val_Z = None
        val_y = processed.val_y
        if processed.val_X is not None and processed.val_y is not None:
            val_Z = batched_compute(self.model, processed.val_X, feature_batch_size, desc="[Extracting] val")
            print_feature_stats(val_Z, "post_val_features")

        test_features = batched_compute(self.model, test_X, feature_batch_size, desc="[Extracting] test")
        print_feature_stats(test_features, "post_test_features")

        readout_name = type(self.readout).__name__ if self.readout else "None"
        print(f"\n=== Step 7: Readout ({readout_name}) with val data ===")

        # Align lengths Helper (Local)
        def align_length(feat, targ):
             if feat is not None and targ is not None:
                 def get_len(arr): return arr.shape[1] if arr.ndim == 3 else arr.shape[0]
                 len_f = get_len(feat)
                 len_t = get_len(targ)
                 if len_f < len_t:
                     diff = len_t - len_f
                     if targ.ndim == 3: return targ[:, diff:, :]
                     return targ[diff:]
             return targ

        train_y = align_length(train_features, train_y)
        val_y = align_length(val_Z, val_y)
        test_y = align_length(test_features, test_y)

        # 3. Fit Readout
        if self.readout is None:
            print("Readout is None. End-to-End mode.")
            train_pred = train_features
            test_pred = test_features
            val_pred = val_Z
        elif dataset_meta.task_type is TaskType.CLASSIFICATION:
            # === Classification: Open-Loop Evaluation ===
            print("    [Runner] Classification task: Using Open-Loop evaluation.")
            
            # Reshape for fit
            tf_reshaped = train_features.reshape(-1, train_features.shape[-1]) if train_features.ndim == 3 else train_features
            ty_reshaped = train_y.reshape(-1, train_y.shape[-1]) if train_y.ndim == 3 else train_y
            vf_reshaped = val_Z.reshape(-1, val_Z.shape[-1]) if val_Z.ndim == 3 else val_Z
            vy_reshaped = val_y.reshape(-1, val_y.shape[-1]) if val_y.ndim == 3 else val_y
            
            # Use fit_and_search for hyperparameter optimization
            # Set lambda candidates if readout supports it
            if hasattr(self.readout, 'lambda_candidates'):
                self.readout.lambda_candidates = ridge_lambdas
            best_lambda, search_history, weight_norms = self.readout.fit_and_search(
                tf_reshaped, ty_reshaped,
                vf_reshaped, vy_reshaped,
                task_type=dataset_meta.task_type
            )
            best_score = search_history.get(best_lambda, 0.0)
            print(f"    [Runner] Best Lambda: {best_lambda:.5e} (Accuracy: {best_score:.5f})")
            
            # Predictions
            train_pred = self.readout.predict(train_features)
            test_pred = self.readout.predict(test_features) if test_features is not None else None
            val_pred = self.readout.predict(val_Z) if val_Z is not None else None
        else:
            # === Regression: Closed-Loop Hyperparameter Search ===
            print(f"    [Runner] Starting Closed-Loop Hyperparameter Search over {len(ridge_lambdas)} candidates...")

            # Setup Projection Fn
            proj_fn = None
            if self.config.projection is not None and hasattr(frontend_ctx, "projection_layer"):
                 def proj_fn(x): return frontend_ctx.projection_layer(x)

            # Search Loop
            best_score = float("inf")
            best_lambda = ridge_lambdas[0]

            # Reshape for fit
            tf_reshaped = train_features.reshape(-1, train_features.shape[-1]) if train_features.ndim == 3 else train_features
            ty_reshaped = train_y.reshape(-1, train_y.shape[-1]) if train_y.ndim == 3 else train_y

            # Val Seed
            val_steps = val_Z.shape[1] if val_Z.ndim == 3 else 0
            seed_len = val_steps
            if train_X.shape[1] < seed_len:
                 seed_len = train_X.shape[1]
            seed_data = train_X[:, -seed_len:, :]

            for lam in tqdm(ridge_lambdas, desc="[Closed-Loop Search]"):
                lam_val = float(lam)

                if hasattr(self.readout, "ridge_lambda"):
                    self.readout.ridge_lambda = lam_val

                # Fit
                self.readout.fit(tf_reshaped, ty_reshaped)

                # Track norm
                if hasattr(self.readout, "coef_") and self.readout.coef_ is not None:
                     weight_norms[lam_val] = float(jnp.linalg.norm(self.readout.coef_))

                # Generate & Score
                val_gen_features = self.generate_closed_loop(jnp.array(seed_data), steps=val_steps, projection_fn=proj_fn, verbose=False)
                # Use centralized compute_score
                score = compute_score(val_gen_features, val_y, self.metric_name)

                search_history[lam_val] = float(score)

                if score < best_score:
                    best_score = score
                    best_lambda = lam_val

            print(f"    [Runner] Best Closed-Loop Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")

            # Report Search Results (Delegated)
            search_results = {
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms
            }
            print_ridge_search_results(search_results, self.metric_name)

            # Refit with Best Lambda
            print(f"    [Runner] Re-fitting readout with best_lambda={best_lambda:.5e}...")
            if hasattr(self.readout, "ridge_lambda"):
                self.readout.ridge_lambda = best_lambda
            self.readout.fit(tf_reshaped, ty_reshaped)

            # Open-Loop Predictions (Train only)
            train_pred = self.readout.predict(train_features)
            test_pred = None
            val_pred = None

            print("\n=== Step 8: Final Predictions (Inverse Transformed):===")

            # 4. Closed-Loop Generation (Test)
            if "reservoir" in self.config.model_type.value or "distillation" in self.config.model_type.value:
                 try:
                     generation_steps = test_X.shape[1] if hasattr(test_X, "shape") else 0

                     # === Test Seed Preparation: Train + Val Inputs (Fixed Logic) ===
                     print(f"    [Runner] Preparing seed for Test (Train + Val)...")
                     val_X_input = processed.val_X
                     # Concatenate Train and Val features for continuous history
                     full_history_inputs = jnp.concatenate([jnp.array(train_X), jnp.array(val_X_input)], axis=1)
                     context_len = full_history_inputs.shape[1]
                     seed_input = full_history_inputs[:, -context_len:, :]

                     print(f"    [Runner] Full Closed-Loop Test: Generating {generation_steps} steps.")

                     # Generate
                     closed_loop_pred_val = self.generate_closed_loop(
                         seed_input, steps=generation_steps, projection_fn=proj_fn)

                     # Evaluate (Chaos Metrics)
                     if frontend_ctx.scaler is not None:
                         scaler = frontend_ctx.scaler
                         shape_cl = closed_loop_pred_val.shape
                         cl_pred_raw = (scaler.inverse_transform(closed_loop_pred_val.reshape(-1, shape_cl[-1]))
                         .reshape(shape_cl))

                         # Truth = Full Test Y
                         closed_loop_truth_val = test_y
                         if hasattr(closed_loop_truth_val, "shape"):
                             shape_tr = closed_loop_truth_val.shape
                             truth_raw = scaler.inverse_transform(
                                 closed_loop_truth_val.reshape(-1, shape_tr[-1])).reshape(shape_tr)

                             global_start = train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                             global_end = global_start + generation_steps

                             print(f"\n[Closed-Loop Metrics] (Global Steps {global_start} -> {global_end})")
                             # Delegated to reporting
                             calculate_chaos_metrics(truth_raw, cl_pred_raw)

                 except Exception as e:
                     print(f"[Warning] Closed-loop generation failed: {e}")

        # --- Construct Results ---

        test_score = 0.0
        results = {}
        if closed_loop_pred_val is not None and closed_loop_truth_val is not None:
            print("\n    [Runner] Overwriting Test Output with Closed-Loop result.")
            test_pred = closed_loop_pred_val
            test_y = closed_loop_truth_val
            if self.readout is not None:
                # Use centralized score
                test_score = compute_score(test_pred, test_y, self.metric_name)
                print(f"    [Runner] Closed-Loop MSE: {test_score:.5f}")
            results["is_closed_loop"] = True
        elif test_pred is None:
             print("    [Runner] Warning: No Test predictions available.")

        results["train"] = {
            "best_lambda": best_lambda,
            "search_history": search_history,
            "weight_norms": weight_norms
        }
        results["test"] = {self.metric_name: test_score}

        # Validation Result (Prioritize Closed-Loop best_score)
        val_score = best_score if best_score is not None else compute_score(val_pred, val_y, self.metric_name)
        results["validation"] = {self.metric_name: val_score}

        results["outputs"] = {
            "train_pred": train_pred,
            "test_pred": test_pred,
            "val_pred": val_pred
        }

        results["readout"] = self.readout
        results["scaler"] = frontend_ctx.scaler
        results["training_logs"] = train_logs
        results["meta"] = {
            "metric": self.metric_name,
            "elapsed_sec": time.time() - start,
            "pretrain_sec": train_time
        }

        # Cleanup
        del train_features, test_features, val_Z

        return results