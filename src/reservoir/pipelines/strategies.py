"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/strategies.py"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from reservoir.core.types import NpF64, to_np_f64, to_jax_f64, ResultDict, JaxF64
import jax.numpy as jnp
import numpy as np

from reservoir.pipelines.config import FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig
from reservoir.utils.reporting import print_ridge_search_results, print_feature_stats, print_chaos_metrics
from reservoir.utils.metrics import compute_score, calculate_chaos_metrics
from reservoir.pipelines.evaluation import Evaluator

class ReadoutStrategy(ABC):
    """Abstract base class for readout fitting and evaluation strategies."""
    
    def __init__(self, evaluator: Evaluator, metric_name: str):
        self.evaluator = evaluator
        self.metric_name = metric_name

    @staticmethod
    def _flatten_3d_to_2d(arr: NpF64 | None, label: str = "array") -> NpF64 | None:
        """Flatten 3D states (Batch, Time, Features) -> 2D (Batch, Features)."""
        if arr is None:
            return None
        if arr.ndim == 3:
            print(f"    [Runner] Flattening 3D {label} {arr.shape} -> 2D")
            return arr.reshape(arr.shape[0], -1)
        return arr

    @staticmethod
    def _get_seed_sequence(train_X: NpF64, val_X: NpF64 | None) -> JaxF64:
        """Prepare seed for closed-loop (concat train+val)."""
        if val_X is not None:
            axis = 1 if train_X.ndim == 3 else 0
            # Transition to Device domain
            return jnp.concatenate([to_jax_f64(train_X), to_jax_f64(val_X)], axis=axis)
        return to_jax_f64(train_X)

    @abstractmethod
    def fit_and_evaluate(
        self,
        model: Callable,
        readout: Callable | None,
        train_Z: NpF64,
        val_Z: NpF64 | None,
        test_Z: NpF64 | None,
        train_y: NpF64 | None,
        val_y: NpF64 | None,
        test_y: NpF64 | None,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
        pipeline_config: PipelineConfig
    ) -> ResultDict:
        """Fit readout and return predictions/metrics."""
        pass


class EndToEndStrategy(ReadoutStrategy):
    """Strategy for End-to-End models where features are predictions."""
    
    def fit_and_evaluate(
        self, model: Callable, 
        readout: Callable | None, 
        train_Z: NpF64, 
        val_Z: NpF64 | None, 
        test_Z: NpF64 | None, 
        train_y: NpF64 | None, 
        val_y: NpF64 | None, 
        test_y: NpF64 | None,
        frontend_ctx: FrontendContext, 
        dataset_meta: DatasetMetadata, 
        pipeline_config: PipelineConfig
    ) -> ResultDict:
        print("Readout is None. End-to-End mode.")
        
        # For FNN windowed mode, align targets using the model's adapter
        aligned_test_y = test_y

        if hasattr(model, 'adapter') and test_y is not None:
            aligned_test_y = model.adapter.align_targets(test_y)

        result = {
            "train_pred": train_Z,
            "val_pred": val_Z,
            "test_pred": test_Z,
            "aligned_test_y": aligned_test_y,  # Pass aligned y for scoring
            "best_lambda": None,
            "best_score": None,
            "search_history": {},
            "weight_norms": {},
            "closed_loop_pred": None,
            "closed_loop_truth": None,
            "chaos_results": None,
            "metrics": {},
        }

        # FNN Closed-Loop for regression check
        if not dataset_meta.classification and hasattr(model, 'window_size') and model.window_size is not None:
             # Delegate to helper if needed, copying logic from legacy Runner
             # Logic from _strategy_end_to_end line 150+
             print("\n=== Step 8: FNN Closed-Loop Generation ===")
             try:
                processed = frontend_ctx.processed_split
                generation_steps = processed.test_X.shape[1] if hasattr(processed.test_X, "shape") else 0
                if processed.test_X.ndim == 2: generation_steps = processed.test_X.shape[0]

                seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
                # For E2E, readout is None or implicit, pass explicit None if needed, but signature says readout
                # EndToEnd typically has readout=None.
                closed_loop_pred = model.generate_closed_loop(seed_data, steps=generation_steps, readout=readout)
                
                # Check for divergence
                pred_std = np.std(closed_loop_pred)
                if pred_std > 50:
                    raise ValueError(f"Closed-loop prediction diverged! STD={pred_std:.2f} > 50")
                
                print_feature_stats(closed_loop_pred, "8:fnn_closed_loop_prediction")

                global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                global_end = global_start + generation_steps

                chaos_results = self.evaluator.compute_chaos_metrics(
                    jnp.array(processed.test_y), jnp.array(closed_loop_pred), frontend_ctx.preprocessor,
                    dataset_meta.preset.config, global_start, global_end)

                result["closed_loop_pred"] = closed_loop_pred
                result["closed_loop_truth"] = processed.test_y
                result["chaos_results"] = chaos_results
             except Exception as e:
                print(f"[Warning] FNN Closed-loop generation failed: {e}")
        
        return result



class ClassificationStrategy(ReadoutStrategy):
    """Open-Loop classification strategy with Accuracy optimization."""
    
    def fit_and_evaluate(
        self, model: Callable, readout: Callable | None, train_Z: NpF64, val_Z: NpF64 | None, test_Z: NpF64 | None, train_y: NpF64 | None, val_y: NpF64 | None, test_y: NpF64 | None, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata, pipeline_config: PipelineConfig
    ) -> ResultDict:
        print("    [Runner] Classification task: Using Open-Loop evaluation.")
        
        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        test_Z = self._flatten_3d_to_2d(test_Z, "test states")
        ty_reshaped, vy_reshaped = train_y, val_y
        
        search_history: dict[float, float] = {}
        weight_norms: dict[float, float] = {}
        best_lambda: float | None = None
        best_score = -float("inf")

        from reservoir.readout.ridge import RidgeCV, RidgeRegression
        if isinstance(readout, RidgeCV):
            # Optimization loop (Moving responsibility from Model to Strategy Mapper)
            print(f"    [Strategy] Optimizing RidgeCV over {len(readout.lambda_candidates)} candidates...")
            
            # Prepare JAX inputs
            train_Z_jax = to_jax_f64(tf_reshaped)
            train_y_jax = to_jax_f64(ty_reshaped)
            val_Z_jax = to_jax_f64(vf_reshaped)

            for lam in tqdm(readout.lambda_candidates, desc="[RidgeCV Search]"):
                lam_val = float(lam)
                model = RidgeRegression(ridge_lambda=lam_val, use_intercept=readout.use_intercept)
                model.fit(train_Z_jax, train_y_jax)
                
                # Predict (Device) & Score (Host)
                val_pred = model.predict(val_Z_jax)
                score = compute_score(to_np_f64(val_pred), val_y, self.metric_name)
                
                search_history[lam_val] = float(score)
                if model.coef_ is not None:
                    weight_norms[lam_val] = float(jnp.linalg.norm(model.coef_))

                if score > best_score:
                    best_score = float(score)
                    best_lambda = lam_val
            
            print(f"    [Strategy] Best Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")
            readout.best_model = RidgeRegression(ridge_lambda=best_lambda, use_intercept=readout.use_intercept) # type: ignore
            readout.best_model.fit(train_Z_jax, train_y_jax)
            
            print_ridge_search_results({
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms
            }, metric_name="Accuracy")
        else:
            print("    [Runner] No hyperparameter search needed for this readout.")
            readout.fit(to_jax_f64(tf_reshaped), to_jax_f64(ty_reshaped))

        print("\n=== Step 8: Final Predictions:===")

        # Calculate Predictions (Explicit domain crossing)
        train_pred = readout.predict(to_jax_f64(train_Z))
        val_pred = readout.predict(to_jax_f64(val_Z)) if val_Z is not None else None
        test_pred = readout.predict(to_jax_f64(test_Z)) if test_Z is not None else None

        # Train
        metrics = {"train": {
            self.metric_name: compute_score(to_np_f64(train_pred), train_y, self.metric_name)
        }}
        

        # Val
        if val_pred is not None and val_y is not None:
             metrics["val"] = {
                 self.metric_name: compute_score(to_np_f64(val_pred), val_y, self.metric_name)
             }
             
        # Test
        if test_pred is not None and test_y is not None:
             metrics["test"] = {
                 self.metric_name: compute_score(to_np_f64(test_pred), test_y, self.metric_name)
             }

        return {
            "train_pred": train_pred,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "metrics": metrics,
            "best_lambda": best_lambda,
            "best_score": best_score,
            "search_history": search_history,
            "weight_norms": weight_norms,
            "closed_loop_pred": None,
            "closed_loop_truth": None,
            "chaos_results": None,
        }


class ClosedLoopRegressionStrategy(ReadoutStrategy):
    """Closed-Loop regression strategy (VPT optimization)."""

    def fit_and_evaluate(
        self,
            model: Callable,
            readout: Callable | None,
            train_Z: NpF64,
            val_Z: NpF64 | None,
            test_Z: NpF64 | None,
            train_y: NpF64 | None,
            val_y: NpF64 | None,
            test_y: NpF64 | None,
            frontend_ctx: FrontendContext,
            dataset_meta: DatasetMetadata,
            pipeline_config: PipelineConfig
    ) -> ResultDict:


        proj_fn = None
        # Check pipeline_config for projection, not dataset_meta
        if pipeline_config.projection is not None and hasattr(frontend_ctx, "projection_layer"):
             def proj_fn(x): return frontend_ctx.projection_layer(x)

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        ty_reshaped = train_y
        
        # Open-Loop Validation Prep
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        vy_reshaped = val_y

        from reservoir.readout.ridge import RidgeCV, RidgeRegression
        if isinstance(readout, RidgeCV):
             # Define helper for unscaling inputs to compute consistent metrics (Raw NMSE)
             scaler = frontend_ctx.preprocessor
             
             def _inverse(arr: NpF64) -> NpF64:
                 if scaler is None: return arr
                 # Basic inverse logic assuming (N, features)
                 shape = arr.shape
                 try:
                     # Check if 1D and needs 2D
                     val = arr
                     if val.ndim == 1:
                         val = val.reshape(-1, 1) # Assume single feature if 1D
                     # Use scaler
                     inv = scaler.inverse_transform(val)
                     return inv.reshape(shape)
                 except Exception:
                     return arr

             print(f"    [Strategy] Optimizing RidgeCV (NMSE) over {len(readout.lambda_candidates)} candidates (JAX Vectorized)...")
             
             # Prepare JAX inputs
             X_jax = to_jax_f64(tf_reshaped)
             y_jax = to_jax_f64(ty_reshaped)
             if y_jax.ndim == 1: y_jax = y_jax[:, None]
             
             val_X_jax = to_jax_f64(vf_reshaped)
             
             # Add intercept if needed
             if readout.use_intercept:
                 ones_train = jnp.ones((X_jax.shape[0], 1))
                 X_jax = jnp.concatenate([ones_train, X_jax], axis=1)
                 ones_val = jnp.ones((val_X_jax.shape[0], 1))
                 val_X_jax = jnp.concatenate([ones_val, val_X_jax], axis=1)

             # Precompute XtX and Xty once
             XtX = X_jax.T @ X_jax
             Xty = X_jax.T @ y_jax
             n_features = XtX.shape[0]
             identity = jnp.eye(n_features)

             @jax.jit
             def solve_ridge_vectorized(lams: JaxF64) -> JaxF64:
                 def solve_one(l: float) -> JaxF64:
                     return jax.scipy.linalg.solve(XtX + l * identity, Xty, assume_a="pos")
                 return jax.vmap(solve_one)(lams)

             lambdas_jax = jnp.array(readout.lambda_candidates)
             all_weights = solve_ridge_vectorized(lambdas_jax) # (N_lam, N_feat, N_out)
             
             # Vectorized Predict on Val set
             all_val_preds = jnp.einsum("sf,lfo->lso", val_X_jax, all_weights)
             all_val_preds_np = to_np_f64(all_val_preds)

             search_history: dict[float, float] = {}
             weight_norms: dict[float, float] = {}
             residuals_history: dict[float, np.ndarray] = {}
             best_lambda = readout.lambda_candidates[0]
             best_score = float('inf')

             for i, lam in enumerate(readout.lambda_candidates):
                lam_val = float(lam)
                vp_np = all_val_preds_np[i]
                
                p_raw = _inverse(vp_np)
                t_raw = _inverse(val_y)
                score = compute_score(p_raw, t_raw, "nmse")
                
                # Normalize residuals
                energy = float(np.mean(t_raw**2))
                res_sq = (p_raw.ravel() - t_raw.ravel()) ** 2 / (energy + 1e-12)
                
                search_history[lam_val] = float(score)
                residuals_history[lam_val] = res_sq
                
                w_coef = all_weights[i, 1:] if readout.use_intercept else all_weights[i]
                weight_norms[lam_val] = float(jnp.linalg.norm(w_coef))

                if score < best_score:
                    best_score = float(score)
                    best_lambda = lam_val
             
             print(f"    [Strategy] Best Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")
             best_idx = list(readout.lambda_candidates).index(best_lambda)
             readout.best_model = RidgeRegression(ridge_lambda=best_lambda, use_intercept=readout.use_intercept)
             if readout.use_intercept:
                 readout.best_model.intercept_ = all_weights[best_idx, 0].ravel()
                 readout.best_model.coef_ = all_weights[best_idx, 1:]
             else:
                 readout.best_model.intercept_ = jnp.zeros(all_weights.shape[-1])
                 readout.best_model.coef_ = all_weights[best_idx]
             
             print_ridge_search_results({
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms,
                "residuals_history": residuals_history
             }, metric_name="NMSE")
        else:
             print("    [Runner] No hyperparameter search needed for this readout.")
             readout.fit(to_jax_f64(tf_reshaped), to_jax_f64(ty_reshaped))
             best_lambda = None
             best_score = None
             search_history = {}
             weight_norms = {}
             residuals_history = {}

        # Display Validation Metrics (Open Loop) immediately
        val_pred_early = None
        if val_Z is not None:
             val_pred_early = readout.predict(to_jax_f64(val_Z))
             if val_y is not None:
                 print("\n=== Validation Open Loop Metrics ===")
                 dt = float(getattr(dataset_meta.preset.config, 'dt', 1.0))
                 ltu = float(getattr(dataset_meta.preset.config, 'lyapunov_time_unit', 1.0))
                 
                 # _inverse is defined above inside the RidgeCV check, 
                 # we need a version available here too or move it.
                 # I'll define it locally for safety if not defined.
                 scaler = frontend_ctx.preprocessor
                 def _inv_local(arr: NpF64) -> NpF64:
                     if scaler is None: return arr
                     try:
                         val = arr
                         if val.ndim == 1: val = val.reshape(-1, 1)
                         return scaler.inverse_transform(val).reshape(arr.shape)
                     except Exception: return arr
                 
                 val_y_raw = _inv_local(val_y)
                 val_pred_raw = _inv_local(to_np_f64(val_pred_early))
                 
                 val_metrics_chaos = calculate_chaos_metrics(val_y_raw, val_pred_raw, dt=dt, lyapunov_time_unit=ltu)
                 print_chaos_metrics(val_metrics_chaos)
                 if float(val_metrics_chaos.get("vpt_lt", 0.0)) < 3:
                     raise ValueError(f"Validation VPT too low: {val_metrics_chaos.get('vpt_lt'):.2f} LT")

        # Test Generation
        print("\n=== Step 8: Final Predictions:===")
        closed_loop_pred = None
        closed_loop_truth = None
        chaos_results = None

        # Check model type for compatibility (simplified check)
        # Assuming "reservoir", "distillation", "passthrough" are all capable if they are passed here.
        # But we can safeguard.
        processed = frontend_ctx.processed_split
        if hasattr(processed.test_X, "shape"):
            if processed.test_X.ndim == 3:
                generation_steps = int(processed.test_X.shape[1])
            else:
                generation_steps = int(processed.test_X.shape[0])
        else:
            generation_steps = 0
        
        full_seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
        print(f"    [Runner] Full Closed-Loop Test: Generating {generation_steps} steps.")
        
        # seed_data is already JAX array from _get_seed_sequence
        closed_loop_pred = model.generate_closed_loop(
            full_seed_data, steps=generation_steps, readout=readout, projection_fn=proj_fn
        )
        print_feature_stats(closed_loop_pred, "8:closed_loop_prediction")

        closed_loop_truth = test_y
        print_feature_stats(closed_loop_truth, "8:closed_loop_truth")

        # Check for divergence
        pred_np = to_np_f64(closed_loop_pred)
        truth_np = to_np_f64(test_y) if test_y is not None else np.array([])

        pred_std = np.std(pred_np)
        pred_max = np.max(pred_np)
        pred_min = np.min(pred_np)

        truth_std = np.std(truth_np)
        truth_max = np.max(truth_np)
        truth_min = np.min(truth_np)

        threshold = 1.5

        # Stats to return (No recalculation needed later)
        stats_dict = {
            "pred_mean": float(np.mean(pred_np)),
            "pred_std": float(pred_std),
            "pred_min": float(pred_min),
            "pred_max": float(pred_max),
            "truth_mean": float(np.mean(truth_np)),
            "truth_std": float(truth_std),
            "truth_min": float(truth_min),
            "truth_max": float(truth_max),
        }

        if pred_std > threshold * truth_std or truth_std > threshold * pred_std:
            err = ValueError(f"Closed-loop prediction diverged! Pred STD={pred_std:.2f} > {threshold}x Truth STD={truth_std:.2f} (or collapsed)")
            err.stats = stats_dict
            raise err

        if pred_max > threshold + truth_max or truth_max > threshold + pred_max:
             err = ValueError(f"Closed-loop prediction diverged! Pred Max={pred_max:.2f} > {threshold}x Truth Max={truth_max:.2f} (or collapsed)") 
             err.stats = stats_dict
             raise err

        # Calculate global_start based on dimensions
        def get_time_steps(arr: NpF64 | None) -> int:
            if arr is None: return 0
            # If 3D (Batch, Time, Feat), return Time (shape[1])
            if arr.ndim == 3: return int(arr.shape[1])
            # If 2D (Time, Feat) or (Batch, Feat) - assuming Time for Series
            # For Reservoir time-series Time is usually axis 0 in 2D
            return int(arr.shape[0])

        train_steps = get_time_steps(processed.train_X)
        val_steps_count = get_time_steps(processed.val_X)
        global_start = train_steps + val_steps_count
        global_end = global_start + generation_steps
        
        if closed_loop_truth is not None:
            chaos_results = self.evaluator.compute_chaos_metrics(
                to_jax_f64(closed_loop_truth), closed_loop_pred, frontend_ctx.preprocessor,
                dataset_meta.preset.config, global_start, global_end, verbose=True
            )

        # Calculate Predictions (Open Loop)
        train_pred = readout.predict(to_jax_f64(train_Z))

        # Train
        metrics: ResultDict = {"train": {
            self.metric_name: compute_score(to_np_f64(train_pred), train_y, self.metric_name)
        }} # type: ignore
        

        # Val (if needed, though Strategy optimized on it)
        # Note: Strategy optimization loop computed best_score, but we can recompute or use it. 
        # Using separate predict calls ensures consistency.
        val_pred = val_pred_early
        if val_pred is None and val_Z is not None:
            val_pred = readout.predict(to_jax_f64(val_Z))
            
        if val_pred is not None and val_y is not None:
            metrics["val"] = {
                self.metric_name: compute_score(to_np_f64(val_pred), val_y, self.metric_name)
            } # type: ignore

        # Test
        test_pred = None
        if test_Z is not None:
             test_pred = readout.predict(to_jax_f64(test_Z))
             metrics["test"] = {
                 self.metric_name: compute_score(to_np_f64(test_pred), test_y, self.metric_name)
             } # type: ignore

        return {
            "train_pred": None, # Not returned by this strategy
            "val_pred": None,   # Not returned
            "test_pred": None,  # Not returned
            "metrics": metrics,
            "best_lambda": best_lambda,
            "best_score": best_score,
            "search_history": search_history,
            "weight_norms": weight_norms,
            "residuals_history": residuals_history if 'residuals_history' in locals() else None,
            "closed_loop_pred": closed_loop_pred,
            "closed_loop_truth": closed_loop_truth,
            "chaos_results": {**chaos_results, **stats_dict}, # Merge stats
        }

class ReadoutStrategyFactory:
    """Factory to create appropriate ReadoutStrategy based on config."""
    
    @staticmethod
    def create_strategy(
        readout: Callable | None,
        dataset_meta: DatasetMetadata,
        evaluator: Evaluator,
        metric_name: str
    ) -> ReadoutStrategy:
        if readout is None:
            return EndToEndStrategy(evaluator, metric_name)
        elif dataset_meta.classification:
            return ClassificationStrategy(evaluator, metric_name)
        else:
            return ClosedLoopRegressionStrategy(evaluator, metric_name)
