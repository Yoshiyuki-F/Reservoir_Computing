"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/generic_runner.py
Universal pipeline that treats models as feature extractors and owns the readout.
Updated to distinguish between Aggregation (Reservoir) and Inference (FNN/Distillation) in logs.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from reservoir.pipelines.config import ModelStack, FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig
from reservoir.utils.reporting import calculate_chaos_metrics

class UniversalPipeline:
    """Runs the V2 flow: pre-train model -> extract features -> fit ridge -> evaluate."""

    def __init__(self, stack: ModelStack, config: PipelineConfig) -> None:
        self.model = stack.model
        self.readout = stack.readout
        self.metric_name = stack.metric
        self.config = config

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    def _score(self, preds: jnp.ndarray, targets: jnp.ndarray) -> float:
        preds_arr = jnp.asarray(preds)
        targets_arr = jnp.asarray(targets)
        if self.metric_name == "accuracy":
            pred_labels = preds_arr if preds_arr.ndim == 1 else jnp.argmax(preds_arr, axis=-1)
            true_labels = targets_arr if targets_arr.ndim == 1 else jnp.argmax(targets_arr, axis=-1)
            return float(jnp.mean(pred_labels == true_labels))

        aligned_preds = preds_arr
        if preds_arr.shape != targets_arr.shape and preds_arr.size == targets_arr.size:
            aligned_preds = preds_arr.reshape(targets_arr.shape)
        return float(jnp.mean((aligned_preds - targets_arr) ** 2))

    def _feature_stats(self, features: Any, stage: str) -> None:
        if isinstance(features, np.ndarray):
            feats = features
            stats = {
                "shape": feats.shape,
                "mean": float(np.mean(feats)),
                "std": float(np.std(feats)),
                "min": float(np.min(feats)),
                "max": float(np.max(feats)),
                "nans": int(np.isnan(feats).sum()),
            }
        else:
            feats = jnp.asarray(features, dtype=jnp.float64)
            stats = {
                "shape": feats.shape,
                "mean": float(jnp.mean(feats)),
                "std": float(jnp.std(feats)),
                "min": float(jnp.min(feats)),
                "max": float(jnp.max(feats)),
                "nans": int(jnp.isnan(feats).sum()),
            }
        print(
            f"[FeatureStats:{stage}] shape={stats['shape']}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, nans={stats['nans']}"
        )
        if stats["std"] < 1e-6:
            print("Feature matrix has near-zero variance. Model output may be inactive.")

    def _extract_features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if hasattr(self.model, "__call__"):
            try:
                return jnp.asarray(self.model(inputs))
            except TypeError:
                pass
        if hasattr(self.model, "predict"):
            return jnp.asarray(self.model.predict(inputs))
        raise AttributeError("Model must implement __call__ or predict to produce features.")

    def _extract_features_batched(self, inputs: Any, batch_size: int) -> np.ndarray:
        """
        CPUメモリ(Numpy)上のデータを少しずつGPU(JAX)に送り、計算結果を再びCPUに戻す。
        巨大なデータセットでVRAM不足(OOM)を防ぐための実装。
        """
        # 1. 入力をNumpy配列として確保 (絶対に jnp.array(inputs) してはいけない)
        inputs_np = np.asarray(inputs)
        n_samples = inputs_np.shape[0]

        # print(f"    [FeatureExtraction] CPU Manual Batching (Batch Size: {batch_size}, Total: {n_samples})")

        if n_samples == 0:
            return np.array([])

        # 2. 出力の形状と型を決定するためのダミー実行 (最初の1件だけGPUで計算)
        # JITコンパイルのトリガーにもなります
        dummy_in = jnp.array(inputs_np[:1])
        dummy_out = self._extract_features(dummy_in)

        # 出力形状の計算: (54000, Features...)
        out_shape = (n_samples,) + dummy_out.shape[1:]
        out_dtype = dummy_out.dtype

        # 3. 結果格納用の巨大なNumpy配列をCPUメモリに確保
        output = np.empty(out_shape, dtype=out_dtype)

        # 4. 高速化用JIT関数
        @jax.jit
        def step(batch_in):
            return self._extract_features(batch_in)

        # 5. CPUループ実行 (tqdm適用)
        # ログが冗長にならないよう、descをシンプルに
        with tqdm(total=n_samples, desc="[Pipeline] Extracting", unit="samples") as pbar:
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                current_batch_size = batch_end - i

                # (A) CPUでスライス
                batch_in_cpu = inputs_np[i: batch_end]

                # (B) GPUへ転送 & 計算
                batch_out_jax = step(jnp.array(batch_in_cpu))

                # (C) 結果を即座にCPUへ戻す
                output[i: batch_end] = np.asarray(batch_out_jax)

                # プログレスバーを進める
                pbar.update(current_batch_size)

        return output

    def batch_transform(self, inputs: Any, batch_size: int, to_numpy: bool = True) -> Any:
        """
        Public batched feature extraction that optionally moves results to CPU (numpy).
        """
        features = self._extract_features_batched(inputs, batch_size)
        if to_numpy:
            return np.asarray(features)
        return features

    def generate_closed_loop(self, initial_input: jnp.ndarray, steps: int, projection_fn: Optional[Any] = None, verbose: bool = True) -> jnp.ndarray:
        # 1. Warmup
        history = jnp.asarray(initial_input)
        if history.ndim == 2: history = history[None, ...]
        elif history.ndim == 1: history = history[None, None, ...]
        
        batch_size = history.shape[0]
        
        # Check support
        if not hasattr(self.model, "step") or not hasattr(self.model, "initialize_state"):
            return self._generate_closed_loop_naive(history, steps, projection_fn, verbose=verbose)

        if verbose:
            print(f"[Closed-Loop] Generating {steps} steps (Fast JAX Scan)...")
        
        # Initialize & Warmup
        initial_state = self.model.initialize_state(batch_size)
        final_state, _ = self.model.forward(initial_state, history)
        
        # Helper for readout
        def predict_one(s):
            s_in = s[:, None, :] 
            if self.readout is not None:
                out = self.readout.predict(s_in) 
                return out[:, 0, :] 
            else:
                 return s

        first_prediction = predict_one(final_state)
        
        # Scan Step
        def scan_step(carry, _):
            h_prev, x_raw = carry
            x_proj = projection_fn(x_raw) if projection_fn else x_raw
            h_next, _ = self.model.step(h_prev, x_proj)
            y_next = predict_one(h_next)
            return (h_next, y_next), y_next

        # Execute Scan
        _, predictions = jax.lax.scan(scan_step, (final_state, first_prediction), None, length=steps)
        
        return jnp.swapaxes(predictions, 0, 1)

    def _generate_closed_loop_naive(self, history: jnp.ndarray, steps: int, projection_fn: Optional[Any], verbose: bool = True) -> jnp.ndarray:
        predictions = []
        current_history = history
        if verbose:
            print(f"[Closed-Loop] Generating {steps} steps (Naive Loop)...")
        for _ in tqdm(range(steps), desc="[Closed-Loop]", unit="step", leave=False, disable=not verbose):
             features = self._extract_features(current_history)
             if self.readout is not None:
                 pred_seq = self.readout.predict(features)
             else:
                 pred_seq = features
             last_pred = pred_seq[:, -1:, :] 
             predictions.append(last_pred)
             
             pred_to_append = last_pred
             if projection_fn:
                  p_flat = last_pred.reshape(-1, last_pred.shape[-1])
                  p_proj = projection_fn(p_flat).reshape(last_pred.shape[0], 1, -1)
                  pred_to_append = p_proj
             
             current_history = jnp.concatenate([current_history, pred_to_append], axis=1)

        return jnp.concatenate(predictions, axis=1)

    def _fit_readout(
        self,
        train_Z: jnp.ndarray,
        train_y: jnp.ndarray,
        val_Z: Optional[jnp.ndarray],
        val_y: Optional[jnp.ndarray],
        ridge_lambdas: Optional[Sequence[float]],
    ) -> tuple[float, Dict[float, float], Dict[float, float]]:
        
        # --- FIX 1: Safely handle Read-Only Config ---
        def safe_set_lambda(val):
            try:
                self.readout.ridge_lambda = float(val)
            except (TypeError, AttributeError):
                # Config is frozen, ignore
                pass

        lambda_candidates = list(ridge_lambdas) if ridge_lambdas is not None else []
        config_init = getattr(getattr(self, "config", None), "readout", None)
        
        initial_lambda = None
        if hasattr(self.readout, "ridge_lambda"):
             initial_lambda = float(self.readout.ridge_lambda)
        elif config_init:
             initial_lambda = float(getattr(config_init, "init_lambda"))

        single_candidate = lambda_candidates[0] if len(lambda_candidates) == 1 else None
        direct_fit = val_Z is None or val_y is None or len(lambda_candidates) <= 1

        # Flatten 3D (Batch, Time, Feat) -> 2D (Sample, Feat) for Ridge
        if train_Z.ndim == 3: train_Z = train_Z.reshape(-1, train_Z.shape[-1])
        if train_y.ndim == 3: train_y = train_y.reshape(-1, train_y.shape[-1])
        
        if val_Z.ndim == 3:
             val_Z = val_Z.reshape(-1, val_Z.shape[-1])
        if val_y.ndim == 3:
             val_y = val_y.reshape(-1, val_y.shape[-1])

        if direct_fit:
            chosen = float(single_candidate if single_candidate is not None else initial_lambda)
            safe_set_lambda(chosen)
            
            self.readout.fit(train_Z, train_y)
            return chosen, {}, {}

        print(f"Search active: overriding initial ridge_lambda={initial_lambda} with {len(lambda_candidates)} candidates.")
        best_lambda, search_history, weight_norms = self.readout.fit_and_search(
            train_Z,
            train_y,
            val_Z,
            val_y,
            lambda_candidates,
            metric=self.metric_name,
        )
        safe_set_lambda(best_lambda)
        return best_lambda, search_history, weight_norms

    # ------------------------------------------------------------------ #
    # Run                                                               #
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
        
        # --- FIX 2: Initialize Variables Early ---
        closed_loop_pred_val = None
        closed_loop_truth_val = None
        val_pred = None
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
        model_cls_name = self.model.__class__.__name__
        is_static_model = "FNN" in model_cls_name or "Distillation" in model_cls_name or "CNN" in model_cls_name

        if is_static_model:
            print("\n=== Step 6: Feature Extraction (Inference) ===")
        else:
            print("\n=== Step 6: Aggregation (Time-Collapse) ===")

        train_features = self.batch_transform(train_X, batch_size=feature_batch_size)
        self._feature_stats(train_features, "post_train_features")

        val_Z = None
        val_y = processed.val_y
        if processed.val_X is not None and processed.val_y is not None:
            val_Z = self.batch_transform(processed.val_X, batch_size=feature_batch_size)
            self._feature_stats(val_Z, "post_val_features")

        test_features = self.batch_transform(test_X, batch_size=feature_batch_size)
        self._feature_stats(test_features, "post_test_features")

        print("\n=== Step 7: Readout (Ridge Regression) with val data ===")

        # Align lengths
        def align_length(feat, targ):
             if feat is not None and targ is not None:
                 def get_len(arr): return arr.shape[1] if arr.ndim == 3 else arr.shape[0]
                 len_f = get_len(feat)
                 len_t = get_len(targ)
                 if len_f < len_t:
                     diff = len_t - len_f
                     print(f"    [Runner] Aligning targets: slicing first {diff} steps.")
                     if targ.ndim == 3: return targ[:, diff:, :]
                     return targ[diff:]
             return targ
             
        train_y = align_length(train_features, train_y)
        val_y = align_length(val_Z, val_y)
        test_y = align_length(test_features, test_y)

        # 3. Fit Readout & Predict (Standard)
        if self.readout is None:
            print("Readout is None. End-to-End mode.")
            train_pred = train_features
            test_pred = test_features
            val_pred = val_Z
        else:
            # === Closed-Loop Hyperparameter Search ===
            print(f"    [Runner] Starting Closed-Loop Hyperparameter Search over {len(ridge_lambdas)} candidates...")


            # 2. Setup Projection Fn (Once)
            proj_fn = None
            if self.config.projection is not None:
                if hasattr(frontend_ctx, "projection_layer") and frontend_ctx.projection_layer is not None:
                     def proj_fn(x): return frontend_ctx.projection_layer(x)
                else:
                    print("    [Runner] Warning: Projection configured but layer not found in context.")

            # 3. Search Loop
            best_score = float("inf")
            best_lambda = ridge_lambdas[0]
            search_history = {}
            weight_norms = {}
            
            # Reshape for fit (Reuse)
            tf_reshaped = train_features.reshape(-1, train_features.shape[-1]) if train_features.ndim == 3 else train_features
            ty_reshaped = train_y.reshape(-1, train_y.shape[-1]) if train_y.ndim == 3 else train_y

            # Val Seed (Use last part of Train to predict Val)
            val_steps = val_Z.shape[1] if val_Z.ndim == 3 else 0

            # We use the processed train_X (Projected) for warmup
            # Assuming train_X is (Batch, Time, Dim)
            seed_len = val_steps
            if train_X.shape[1] < seed_len:
                 seed_len = train_X.shape[1]
            seed_data = train_X[:, -seed_len:, :]

            for lam in tqdm(ridge_lambdas, desc="[Closed-Loop Search]"):
                lam_val = float(lam)
                
                # A. Set Lambda
                if hasattr(self.readout, "ridge_lambda"):
                    self.readout.ridge_lambda = lam_val
                elif hasattr(self.readout, "config"):
                    self.readout.config.ridge_lambda = lam_val
                
                # B. Open-Loop Fit
                self.readout.fit(tf_reshaped, ty_reshaped)
                
                # Track norm
                if hasattr(self.readout, "coef_") and self.readout.coef_ is not None:
                     weight_norms[lam_val] = float(jnp.linalg.norm(self.readout.coef_))

                # C. Closed-Loop Generation (Validation)
                val_gen_features = self.generate_closed_loop(seed_data, steps=val_steps, projection_fn=proj_fn, verbose=False)
                
                # D. Score (MSE against val_y)
                score = self._score(val_gen_features, val_y)
                
                search_history[lam_val] = float(score)
                
                if score < best_score:
                    best_score = score
                    best_lambda = lam_val

            print(f"    [Runner] Best Closed-Loop Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")

            # Print Search Table Immediately
            from reservoir.utils.reporting import print_ridge_search_results
            # Construct a dummy results dict for the printer
            search_results = {
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms
            }
            print_ridge_search_results(search_results, self.metric_name)

            # 4. Refit with Best Lambda
            print(f"    [Runner] Re-fitting readout with best_lambda={best_lambda:.5e}...")
            if hasattr(self.readout, "ridge_lambda"):
                self.readout.ridge_lambda = best_lambda
            elif hasattr(self.readout, "config"):
                self.readout.config.ridge_lambda = best_lambda
            self.readout.fit(tf_reshaped, ty_reshaped)

            # 5. Standard Open-Loop Predictions (for consistency with logs)
            print("Generating final predictions (Open-Loop)...")
            train_pred = self.readout.predict(train_features)
            
            # [Refactor] Remove Open-Loop predictions for Test and Validation.
            test_pred = None
            val_pred = None

            print("\n=== Step 8: Final Predictions (Inverse Transformed):===")

            # Inverse Transform Logging
            if frontend_ctx.scaler is not None and test_pred is not None:
                try:
                    scaler = frontend_ctx.scaler
                    tp_reshaped = test_pred.reshape(-1, 1) if test_pred.ndim == 1 else test_pred
                    test_pred_raw = scaler.inverse_transform(tp_reshaped)
                    self._feature_stats(test_pred_raw, "final_test_pred_raw")
                    
                except Exception as e:
                    print(f"[Warning] Inverse transform stats failed: {e}")

            # 4. Closed-Loop Generation (True Chaos Check)
            if "reservoir" in self.config.model_type.value or "distillation" in self.config.model_type.value:
                 try:
                     # GENERATION_STEP (LENGTH OF TEST TIMESTEP)
                     # VALIDATION SET ALWAYS EXISTS (FOR REGRESSION AND CLASSIFICATION)

                     # === Test Seed Preparation: Train + Val ===
                     print(f"    [Runner] Preparing seed for Test (Train + Val)...")

                     val_X_input = processed.val_X

                     # Concatenate Train and Val features for continuous history
                     full_history_inputs = jnp.concatenate([train_X, val_X_input], axis=1)

                     context_len = full_history_inputs.shape[1]
                     # 入力データをSeedにする
                     seed_input = full_history_inputs[:, -context_len:, :]

                     generation_steps = test_X.shape[1]
                     print(f"    [Runner] Full Closed-Loop Test: Generating {generation_steps} steps.")
                     
                     # Projection Fn
                     proj_fn = None
                     if self.config.projection is not None:
                         if hasattr(frontend_ctx, "projection_layer") and frontend_ctx.projection_layer is not None:
                             print("    [Runner] Using existing InputProjection from FrontendContext.")

                             def proj_fn(x):
                                 return frontend_ctx.projection_layer(x)
                         else:
                             print(" No projection Layer loaded")

                     # 3. Generate
                     closed_loop_pred_val = self.generate_closed_loop(
                         seed_input, steps=generation_steps, projection_fn=proj_fn)

                     # 4. Evaluate (Compare to Full Test)
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

                             # Correct Global Step Calculation
                             train_len = train_X.shape[1] if hasattr(train_X, "shape") else 0
                             val_len = 0
                             if processed.val_X is not None and hasattr(processed.val_X, "shape"):
                                 val_len = processed.val_X.shape[1]

                             global_start = train_len + val_len
                             global_end = global_start + generation_steps

                             print(f"\n[Closed-Loop Metrics] (Global Steps {global_start} -> {global_end})")
                             calculate_chaos_metrics(truth_raw, cl_pred_raw)
                             
                 except Exception as e:
                     print(f"[Warning] Closed-loop generation failed: {e}")

        # --- FIX 3: Unified Output Construction & Overwrite ---
        
        # Calculate final test score (Closed-Loop)
        test_score = 0.0
        
        # Overwrite with Closed-Loop if available
        if closed_loop_pred_val is not None and closed_loop_truth_val is not None:
            print("\n    [Runner] Overwriting Test Output with Closed-Loop result.")
            test_pred = closed_loop_pred_val
            test_y = closed_loop_truth_val
            if self.readout is not None:
                test_score = self._score(test_pred, test_y)
                print(f"    [Runner] Closed-Loop MSE: {test_score:.5f}")
        elif test_pred is None:
             print("    [Runner] Warning: No Test predictions available (Open-Loop disabled, Closed-Loop failed).")

        # Construct Results (Once)
        results = {}
        results["train"] = {
            "best_lambda": best_lambda,
            "search_history": search_history,
            "weight_norms": weight_norms
        }
        results["test"] = {self.metric_name: test_score}

        # Validation Result
        # Prioritize Closed-Loop best_score.
        if best_score is not None:
            val_score = best_score
        else:
            # Fallback (e.g. End-to-End mode)
            val_score = self._score(val_pred, val_y)

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

        # Safe Cleanup
        del train_features, test_features,  val_Z

        return results