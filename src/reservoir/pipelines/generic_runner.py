"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
Refactored V2: Strategy pattern for readout, separated generation primitives.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from reservoir.pipelines.config import ModelStack, FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig

from reservoir.utils.reporting import (
    calculate_chaos_metrics,
    print_ridge_search_results,
    print_feature_stats,
    compute_score
)
from reservoir.utils.batched_compute import batched_compute
from reservoir.utils.printing import print_topology


class UniversalPipeline:
    """Runs the V2 flow: pre-train model -> extract features -> fit ridge -> evaluate."""

    def __init__(self, stack: ModelStack, config: PipelineConfig) -> None:
        self.model = stack.model
        self.readout = stack.readout
        self.metric_name = stack.metric
        self.config = config
        self.topo_meta = stack.topo_meta  # Store for runtime shape updates

    # ------------------------------------------------------------------ #
    # Phase 1: Generation Primitives                                     #
    # ------------------------------------------------------------------ #

    def generate_closed_loop(
        self,
        initial_input: jnp.ndarray,
        steps: int,
        projection_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        verbose: bool = True
    ) -> jnp.ndarray:
        """Generate closed-loop predictions using Fast JAX Scan (for Reservoir models)."""
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

    def generate_closed_loop_windowed(
        self,
        seed_window: jnp.ndarray,
        steps: int,
        verbose: bool = True
    ) -> jnp.ndarray:
        """Generate closed-loop predictions for windowed FNN models."""
        window = jnp.asarray(seed_window)
        if window.ndim == 2:
            window = window[None, ...]

        batch_size, window_size, n_features = window.shape

        if verbose:
            print(f"[Closed-Loop Windowed] Generating {steps} steps (window_size={window_size})...")

        predictions = []

        for step in range(steps):
            flat_window = window.reshape(batch_size, -1)

            if hasattr(self.model, '_state') and self.model._state is not None:
                pred = self.model._model_def.apply({"params": self.model._state.params}, flat_window)
            else:
                pred = jnp.zeros((batch_size, n_features))

            predictions.append(pred)
            pred_expanded = pred[:, None, :]
            window = jnp.concatenate([window[:, 1:, :], pred_expanded], axis=1)

        return jnp.stack(predictions, axis=1)

    def _generate_trajectory(
        self,
        seed_data: jnp.ndarray,
        steps: int,
        projection_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        verbose: bool = True
    ) -> jnp.ndarray:
        """Dispatcher: routes to appropriate generation method based on model type."""
        if hasattr(self.model, 'window_size') and self.model.window_size is not None:
            window_size = self.model.window_size
            seed_window = seed_data[:, -window_size:, :]
            return self.generate_closed_loop_windowed(seed_window, steps, verbose=verbose)
        else:
            return self.generate_closed_loop(seed_data, steps, projection_fn, verbose=verbose)

    def _get_seed_sequence(self, train_X: jnp.ndarray, val_X: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Extract seed sequence from Train+Val for closed-loop generation."""
        if val_X is not None:
            return jnp.concatenate([jnp.array(train_X), jnp.array(val_X)], axis=1)
        return jnp.array(train_X)

    # ------------------------------------------------------------------ #
    # Phase 2: Helper Methods                                            #
    # ------------------------------------------------------------------ #

    def _extract_all_features(
        self,
        train_X: jnp.ndarray,
        val_X: Optional[jnp.ndarray],
        test_X: jnp.ndarray,
        batch_size: int
    ) -> tuple:
        """Extract features from all splits and aggregate (squeeze) if needed."""
        if self.readout is None:
            print("    [Runner] End-to-End mode: Using model.predict directly.")
            train_Z = self._squeeze_if_needed(self.model.predict(train_X))

            val_Z = None
            if val_X is not None:
                val_Z = self._squeeze_if_needed(self.model.predict(val_X))

            test_Z = self._squeeze_if_needed(self.model.predict(test_X))
        else:
            from functools import partial
            
            # Pass 'split_name' to prompt the zero-overhead logging callback inside the model
            
            # Train
            model_train = partial(self.model, split_name="train")
            train_Z = self._squeeze_if_needed(batched_compute(model_train, train_X, batch_size, desc="[Extracting] train"))

            val_Z = None
            if val_X is not None:
                # Val
                model_val = partial(self.model, split_name="val")
                val_Z = self._squeeze_if_needed(batched_compute(model_val, val_X, batch_size, desc="[Extracting] val"))

            # Test
            model_test = partial(self.model, split_name="test")
            test_Z = self._squeeze_if_needed(batched_compute(model_test, test_X, batch_size, desc="[Extracting] test"))

        return train_Z, val_Z, test_Z

    def _squeeze_if_needed(self, arr: jnp.ndarray) -> jnp.ndarray:
        """Squeeze time dimension if it equals 1: (N, 1, F) -> (N, F). Aggregation step."""
        if arr.ndim == 3 and arr.shape[1] == 1:
            return arr.squeeze(axis=1)
        return arr

    def _print_runtime_topology(self, processed, train_Z: jnp.ndarray, test_Z: jnp.ndarray) -> None:
        """Print topology using static metadata (no runtime overwriting)."""
        if not self.topo_meta:
            return
        # Removed runtime probing of 'feature' shape. Rely entirely on Factory metadata.
        print_topology(self.topo_meta)



    def _align_targets(self, features: Optional[jnp.ndarray], targets: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
        """Align target length (dim 0) to match feature length (warmup handling)."""
        if features is None or targets is None:
            return targets

        def get_len(arr):
            return arr.shape[1] if arr.ndim == 3 else arr.shape[0]

        len_f = get_len(features)
        len_t = get_len(targets)

        if len_f < len_t:
            diff = len_t - len_f
            if targets.ndim == 3:
                return targets[:, diff:, :]
            return targets[diff:]
        return targets

    def _compute_chaos_metrics(
        self,
        truth: jnp.ndarray,
        pred: jnp.ndarray,
        scaler,
        dataset_config,
        global_start: int = 0,
        global_end: int = 0,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Compute VPT, NDEI, and other chaos metrics with inverse transform."""
        if scaler is None:
            return None

        shape_pred = pred.shape
        shape_truth = truth.shape
        pred_raw = scaler.inverse_transform(pred.reshape(-1, shape_pred[-1])).reshape(shape_pred)
        truth_raw = scaler.inverse_transform(truth.reshape(-1, shape_truth[-1])).reshape(shape_truth)

        if verbose:
            print(f"\n[Closed-Loop Metrics] (Global Steps {global_start} -> {global_end})")

        dt = getattr(dataset_config, 'dt', 1.0)
        ltu = getattr(dataset_config, 'lyapunov_time_unit', 1.0)

        return calculate_chaos_metrics(truth_raw, pred_raw, dt=dt, lyapunov_time_unit=ltu, verbose=verbose)

    # ------------------------------------------------------------------ #
    # Phase 3: Readout Strategy Pattern                                  #
    # ------------------------------------------------------------------ #

    def _fit_readout_strategy(
        self,
        train_Z: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        test_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_y: Optional[jnp.ndarray],
        test_y: jnp.ndarray,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata
    ) -> Dict[str, Any]:
        """Route to appropriate readout fitting strategy."""
        if self.readout is None:
            return self._strategy_end_to_end(
                train_Z, val_Z, test_Z, train_y, val_y, test_y, frontend_ctx, dataset_meta)
        elif dataset_meta.classification:
            return self._strategy_classification(train_Z, val_Z, test_Z, train_y, val_y)
        else:
            return self._strategy_regression_closed_loop(
                train_Z, val_Z, test_Z, train_y, val_y, test_y, frontend_ctx, dataset_meta)

    def _strategy_end_to_end(
        self,
        train_Z: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        test_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_y: Optional[jnp.ndarray],
        test_y: jnp.ndarray,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata
    ) -> Dict[str, Any]:
        """End-to-End mode: features are predictions."""
        print("Readout is None. End-to-End mode.")

        result = {
            "train_pred": train_Z,
            "val_pred": val_Z,
            "test_pred": test_Z,
            "best_lambda": None,
            "best_score": None,
            "search_history": {},
            "weight_norms": {},
            "closed_loop_pred": None,
            "closed_loop_truth": None,
            "chaos_results": None,
        }

        # FNN Closed-Loop for regression
        if not dataset_meta.classification and hasattr(self.model, 'window_size') and self.model.window_size is not None:
            print("\n=== Step 8: FNN Closed-Loop Generation ===")
            try:
                processed = frontend_ctx.processed_split
                generation_steps = processed.test_X.shape[1] if hasattr(processed.test_X, "shape") else 0

                seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
                closed_loop_pred = self._generate_trajectory(seed_data, steps=generation_steps)

                global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                global_end = global_start + generation_steps

                chaos_results = self._compute_chaos_metrics(
                    processed.test_y, closed_loop_pred, frontend_ctx.scaler,
                    dataset_meta.preset.config, global_start, global_end)

                result["closed_loop_pred"] = closed_loop_pred
                result["closed_loop_truth"] = processed.test_y  # Use original, not aligned
                result["chaos_results"] = chaos_results
            except Exception as e:
                print(f"[Warning] FNN Closed-loop generation failed: {e}")

        return result

    def _strategy_classification(
        self,
        train_Z: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        test_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_y: Optional[jnp.ndarray]
    ) -> Dict[str, Any]:
        """Classification: Open-Loop grid search (accuracy, higher is better)."""
        print("    [Runner] Classification task: Using Open-Loop evaluation.")

        tf_reshaped = train_Z.reshape(-1, train_Z.shape[-1]) if train_Z.ndim == 3 else train_Z
        ty_reshaped = train_y.reshape(-1, train_y.shape[-1]) if train_y.ndim == 3 else train_y
        vf_reshaped = val_Z.reshape(-1, val_Z.shape[-1]) if val_Z is not None and val_Z.ndim == 3 else val_Z
        vy_reshaped = val_y.reshape(-1, val_y.shape[-1]) if val_y is not None and val_y.ndim == 3 else val_y

        search_history = {}
        weight_norms = {}
        best_lambda = None
        best_score = None

        if hasattr(self.readout, 'ridge_lambda'):
            lambda_candidates = getattr(self.readout, 'lambda_candidates', None) or [self.readout.ridge_lambda]
            print(f"    [Runner] Running hyperparameter search over {len(lambda_candidates)} lambdas...")
            best_score = -float("inf")
            best_lambda = lambda_candidates[0]

            for lam in tqdm(lambda_candidates, desc="[Lambda Search]"):
                lam_val = float(lam)
                self.readout.ridge_lambda = lam_val
                self.readout.fit(tf_reshaped, ty_reshaped)

                val_pred_tmp = self.readout.predict(vf_reshaped)
                score = compute_score(val_pred_tmp, vy_reshaped, self.metric_name)
                search_history[lam_val] = float(score)

                if hasattr(self.readout, "coef_") and self.readout.coef_ is not None:
                    weight_norms[lam_val] = float(jnp.linalg.norm(self.readout.coef_))

                if score > best_score:
                    best_score = score
                    best_lambda = lam_val

            self.readout.ridge_lambda = best_lambda
            self.readout.fit(tf_reshaped, ty_reshaped)
            print(f"    [Runner] Best Lambda: {best_lambda:.5e} (Accuracy: {best_score:.5f})")

            print_ridge_search_results({
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms
            }, is_classification=True)
        else:
            print("    [Runner] No hyperparameter search needed for this readout.")
            self.readout.fit(tf_reshaped, ty_reshaped)

        return {
            "train_pred": self.readout.predict(train_Z),
            "val_pred": self.readout.predict(val_Z) if val_Z is not None else None,
            "test_pred": self.readout.predict(test_Z),
            "best_lambda": best_lambda,
            "best_score": best_score,
            "search_history": search_history,
            "weight_norms": weight_norms,
            "closed_loop_pred": None,
            "closed_loop_truth": None,
            "chaos_results": None,
        }

    def _strategy_regression_closed_loop(
        self,
        train_Z: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        test_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_y: Optional[jnp.ndarray],
        test_y: jnp.ndarray,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata
    ) -> Dict[str, Any]:
        """Regression: Closed-Loop grid search (MSE, lower is better)."""
        processed = frontend_ctx.processed_split

        lambda_candidates = getattr(self.readout, 'lambda_candidates', None) or \
            ([self.readout.ridge_lambda] if hasattr(self.readout, 'ridge_lambda') else [1e-3])
        print(f"    [Runner] Starting Closed-Loop Hyperparameter Search over {len(lambda_candidates)} candidates...")

        proj_fn = None
        if self.config.projection is not None and hasattr(frontend_ctx, "projection_layer"):
            def proj_fn(x): return frontend_ctx.projection_layer(x)

        tf_reshaped = train_Z.reshape(-1, train_Z.shape[-1]) if train_Z.ndim == 3 else train_Z
        ty_reshaped = train_y.reshape(-1, train_y.shape[-1]) if train_y.ndim == 3 else train_y

        val_steps = processed.val_X.shape[1] if processed.val_X is not None and processed.val_X.ndim == 3 else 0
        seed_len = min(processed.train_X.shape[1], val_steps) if val_steps > 0 else processed.train_X.shape[1]
        seed_data = processed.train_X[:, -seed_len:, :]

        search_history = {}
        weight_norms = {}
        best_score = float("inf")
        best_lambda = lambda_candidates[0]

        for lam in tqdm(lambda_candidates, desc="[Closed-Loop Search]"):
            lam_val = float(lam)

            if hasattr(self.readout, "ridge_lambda"):
                self.readout.ridge_lambda = lam_val

            self.readout.fit(tf_reshaped, ty_reshaped)

            if hasattr(self.readout, "coef_") and self.readout.coef_ is not None:
                weight_norms[lam_val] = float(jnp.linalg.norm(self.readout.coef_))

            val_gen = self.generate_closed_loop(jnp.array(seed_data), steps=val_steps, projection_fn=proj_fn, verbose=False)

            # Closed-Loop (Regression) always optimizes VPT
            current_metrics = self._compute_chaos_metrics(val_y, val_gen, frontend_ctx.scaler, dataset_meta.preset.config, verbose=False)
            
            # Maximize VPT (Lyapunov Time) by negating (loop logic minimizes)
            score = -current_metrics.get("vpt_lt", 0.0)

            search_history[lam_val] = float(score)

            if score < best_score:
                best_score = score
                best_lambda = lam_val

        # Display as positive VPT
        print(f"    [Runner] Best Lambda: {best_lambda:.5e} (Val VPT: {-best_score:.5f} LT)")
        print_ridge_search_results({
            "search_history": search_history,
            "best_lambda": best_lambda,
            "weight_norms": weight_norms
        }, is_classification=False)

        print(f"    [Runner] Re-fitting readout with best_lambda={best_lambda:.5e}...")
        if hasattr(self.readout, "ridge_lambda"):
            self.readout.ridge_lambda = best_lambda
        self.readout.fit(tf_reshaped, ty_reshaped)

        print("\n=== Step 8: Final Predictions (Inverse Transformed):===")
        closed_loop_pred = None
        closed_loop_truth = None
        chaos_results = None

        if "reservoir" in self.config.model_type.value or "distillation" in self.config.model_type.value or "passthrough" in self.config.model_type.value:
            try:
                generation_steps = processed.test_X.shape[1] if hasattr(processed.test_X, "shape") else 0

                seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
                print(f"    [Runner] Full Closed-Loop Test: Generating {generation_steps} steps.")

                closed_loop_pred = self._generate_trajectory(seed_data, steps=generation_steps, projection_fn=proj_fn)
                closed_loop_truth = test_y

                global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                global_end = global_start + generation_steps

                chaos_results = self._compute_chaos_metrics(
                    closed_loop_truth, closed_loop_pred, frontend_ctx.scaler,
                    dataset_meta.preset.config, global_start, global_end)
            except Exception as e:
                print(f"[Warning] Closed-loop generation failed: {e}")

        return {
            "train_pred": self.readout.predict(train_Z),
            "val_pred": None,
            "test_pred": None,
            "best_lambda": best_lambda,
            "best_score": best_score,
            "search_history": search_history,
            "weight_norms": weight_norms,
            "closed_loop_pred": closed_loop_pred,
            "closed_loop_truth": closed_loop_truth,
            "chaos_results": chaos_results,
        }

    # ------------------------------------------------------------------ #
    # Phase 4: Main Execution Flow                                       #
    # ------------------------------------------------------------------ #

    def run(
        self,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the unified pipeline phase (in Step 4 5 6 ):
        1. Train Model
        2. Extract Features
        3. Align Targets
        4. Fit Readout (Strategy Pattern)
        5. Build Results
        """
        processed = frontend_ctx.processed_split
        start = time.time()

        # --- Step 4: Adapter Stats (Full Dataset) ---
        # This Log was previously in run.py (Probe), moved here for accuracy.
        print("\n=== Step 4: Adapter (Full Dataset Probe) ===")
        # Check for adapter
        adapter_fn = None
        if hasattr(self.model, "student_adapter") and self.model.student_adapter is not None:
             adapter_fn = self.model.student_adapter
        elif hasattr(self.model, "adapter") and self.model.adapter is not None:
             adapter_fn = self.model.adapter
        
        if adapter_fn is not None:
             # Compute and Log
             from functools import partial
             # Adapter is usually stateless and fast, but safely reuse batched_compute
             
             # Train
             train_A = batched_compute(adapter_fn, processed.train_X, dataset_meta.training.batch_size, desc="[Adapter] train")
             print_feature_stats(train_A, "4:train")
             
             # Val
             if processed.val_X is not None:
                 val_A = batched_compute(adapter_fn, processed.val_X, dataset_meta.training.batch_size, desc="[Adapter] val")
                 print_feature_stats(val_A, "4:val")
                 
             # Test
             test_A = batched_compute(adapter_fn, processed.test_X, dataset_meta.training.batch_size, desc="[Adapter] test")
             print_feature_stats(test_A, "4:test")
        else:
             print("    [Runner] No adapter found (Skipped Step 4 details).")

        print(f"\n=== Step 5: Model Dynamics (Training/Warmup) [{self.config.model_type.value}] ===")
        train_logs = self.model.train(processed.train_X, processed.train_y) or {}

        train_Z, val_Z, test_Z = self._extract_all_features(
            processed.train_X,
            processed.val_X,
            processed.test_X,
            dataset_meta.training.batch_size
        )

        print(f"\n=== Step 6: Feature Extraction / Aggregation Flattened(output is 2D) ===")

        print_feature_stats(train_Z, "6:train")
        print_feature_stats(val_Z, "6:val")
        print_feature_stats(test_Z, "6:test")

        # Assert 2D (Batch*Time, Features)
        assert train_Z.ndim == 2, f"Features must be 2D (Samples, Units), got {train_Z.shape}"
        assert val_Z.ndim == 2, f"Val Features must be 2D, got {val_Z.shape}"
        assert test_Z.ndim == 2, f"Test Features must be 2D, got {test_Z.shape}"

        # Flatten targets to match flattened features for alignment
        def _flat_y(y): return y.reshape(-1, y.shape[-1]) if y is not None and y.ndim == 3 else y

        train_y = self._align_targets(train_Z, _flat_y(processed.train_y))
        val_y = self._align_targets(val_Z, _flat_y(processed.val_y))
        test_y = self._align_targets(test_Z, _flat_y(processed.test_y))


        # Step 4: Fit Readout (Strategy Pattern)
        readout_name = type(self.readout).__name__ if self.readout else "None"
        print(f"\n=== Step 7: Readout ({readout_name}) with val data ===")
        
        fit_result = self._fit_readout_strategy(
            train_Z, val_Z, test_Z,
            train_y, val_y, test_y,
            frontend_ctx, dataset_meta
        )

        # Print topology with actual runtime shapes (after Step 7)
        self._print_runtime_topology(processed, train_Z, test_Z)

        # Step 5: Build Results
        return self._build_results(fit_result, train_logs, train_Z, val_Z, test_Z, test_y, frontend_ctx, start)

    def _build_results(
        self,
        fit_result: Dict[str, Any],
        train_logs: Dict[str, Any],
        train_Z: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        test_Z: jnp.ndarray,
        test_y: jnp.ndarray,
        frontend_ctx: FrontendContext,
        start: float
    ) -> Dict[str, Dict[str, Any]]:
        """Construct the final results dictionary."""
        results: Dict[str, Any] = {}

        if fit_result["closed_loop_pred"] is not None:
            print("\n    [Runner] Overwriting Test Output with Closed-Loop result.")
            test_pred = fit_result["closed_loop_pred"]
            test_y_final = fit_result["closed_loop_truth"]
            results["is_closed_loop"] = True
        else:
            test_pred = fit_result["test_pred"]
            test_y_final = test_y

        test_score = 0.0
        if test_pred is not None and test_y_final is not None:
            test_score = compute_score(test_pred, test_y_final, self.metric_name)
            if results.get("is_closed_loop"):
                print(f"    [Runner] Closed-Loop {self.metric_name.upper()}: {test_score:.5f}")

        results["train"] = {
            "search_history": fit_result["search_history"],
            "weight_norms": fit_result["weight_norms"],
        }
        if fit_result["best_lambda"] is not None:
            results["train"]["best_lambda"] = fit_result["best_lambda"]

        results["test"] = {self.metric_name: test_score}
        if fit_result["chaos_results"] is not None:
            chaos = fit_result["chaos_results"]
            results["test"]["chaos_metrics"] = chaos
            results["test"]["vpt_lt"] = chaos.get("vpt_lt", 0.0)
            results["test"]["ndei"] = chaos.get("ndei", float("inf"))

        val_score = fit_result["best_score"] if fit_result["best_score"] is not None else 0.0
        results["validation"] = {self.metric_name: val_score}

        results["outputs"] = {
            "train_pred": fit_result["train_pred"],
            "test_pred": test_pred,
            "val_pred": fit_result["val_pred"],
        }

        results["readout"] = self.readout
        results["scaler"] = frontend_ctx.scaler
        results["training_logs"] = train_logs
        results["meta"] = {
            "metric": self.metric_name,
            "elapsed_sec": time.time() - start,
        }

        del train_Z, val_Z, test_Z

        return results