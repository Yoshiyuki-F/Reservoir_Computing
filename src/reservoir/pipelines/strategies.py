"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/strategies.py"""
from __future__ import annotations

from abc import ABC, abstractmethod
from reservoir.core.types import NpF64, to_np_f64, to_jax_f64, FitResultDict, JaxF64
import jax
import jax.numpy as jnp
from typing import cast, TYPE_CHECKING
import numpy as np


from reservoir.utils.batched_compute import batched_compute
from reservoir.utils.reporting import (
    print_ridge_search_results, 
    print_feature_stats, 
    print_chaos_metrics,
    plot_intermediate_regression_results
)
from reservoir.utils.metrics import compute_score, calculate_chaos_metrics

if TYPE_CHECKING:
    from reservoir.models.generative import Predictable
    from reservoir.models.presets import PipelineConfig
    from reservoir.pipelines.config import FrontendContext, DatasetMetadata
    from reservoir.readout.base import ReadoutModule
    from reservoir.pipelines.evaluation import Evaluator
    from reservoir.models.generative import ClosedLoopGenerativeModel
    from reservoir.core.types import TopologyMeta


def _check_closed_loop_divergence(pred_std: float, threshold: float) -> None:
    """Raise ValueError if closed-loop prediction has diverged."""
    if pred_std > threshold:
        raise ValueError(f"Closed-loop prediction diverged! STD={pred_std:.2f} > {threshold}")


class DivergenceError(ValueError):
    """Raised when closed-loop prediction diverges, carrying diagnostic stats."""
    def __init__(self, message: str, stats: dict[str, float]) -> None:
        super().__init__(message)
        self.stats: dict[str, float] = stats


def optimize_ridge_vmap(
    lambda_candidates: tuple[float, ...],
    use_intercept: bool,
    train_Z: JaxF64,
    train_y: JaxF64,
    val_Z: JaxF64,
    val_y: NpF64,
    metric_name: str,
    batch_size: int,
    inverse_fn: Callable[[NpF64], NpF64] | None = None,
) -> tuple[float, float, dict[float, float], dict[float, float], NpF64, JaxF64, dict[float, np.ndarray] | None]:
    """
    Common vectorized RidgeCV optimization logic using batched_compute.
    
    Returns:
        best_lambda
        best_score
        search_history
        weight_norms
        best_val_pred_np (prediction for best lambda)
        all_weights (all candidate weights)
        residuals_history (optional, for regression analysis)
    """
    from typing import Callable
    
    # 1. Add Intercept if needed
    if use_intercept:
        ones_train = jnp.ones((train_Z.shape[0], 1))
        X_train_aug = jnp.concatenate([ones_train, train_Z], axis=1)
        ones_val = jnp.ones((val_Z.shape[0], 1))
        val_Z_aug = jnp.concatenate([ones_val, val_Z], axis=1)
    else:
        X_train_aug = train_Z
        val_Z_aug = val_Z

    # 2. Precompute XtX and Xty
    y_train_mat = train_y
    if y_train_mat.ndim == 1:
        y_train_mat = y_train_mat[:, None]

    XtX = X_train_aug.T @ X_train_aug
    Xty = X_train_aug.T @ y_train_mat
    n_features = XtX.shape[0]
    identity = jnp.eye(n_features)

    # 3. Vectorized Solver
    @jax.jit
    def solve_ridge_vectorized(lams: JaxF64) -> JaxF64:
        def solve_one(lam_l: float) -> JaxF64:
            return jax.scipy.linalg.solve(XtX + lam_l * identity, Xty, assume_a="pos")
        return jax.vmap(solve_one)(lams)

    lambdas_jax = jnp.array(lambda_candidates)
    all_weights = solve_ridge_vectorized(lambdas_jax) # (N_lam, N_feat, N_out)

    # 4. Batched Prediction
    def predict_batch_fn(weights_batch: JaxF64) -> JaxF64:
        return jnp.einsum("sf,bfo->bso", val_Z_aug, weights_batch)

    # Convert weights to numpy to pass to batched_compute (avoids beartype error)
    all_weights_np = to_np_f64(all_weights)
    
    all_val_preds_np = batched_compute(
        predict_batch_fn,
        all_weights_np,
        batch_size=batch_size, 
        desc="[Step 7 RidgeCV Search]",
        file="strategies.py"
    )

    # 5. Score on CPU
    search_history: dict[float, float] = {}
    weight_norms: dict[float, float] = {}
    residuals_history: dict[float, np.ndarray] = {}
    
    # Determine optimization direction
    minimize = metric_name.lower() in ["nmse", "mse", "rmse", "nrmse", "mase"]
    
    abs_best_score = float('inf') if minimize else -float('inf')
    best_lambda = lambda_candidates[0]
    best_score_val = abs_best_score
    best_pred_idx = 0
    
    NORM_THRESHOLD = 100.0 #TODO is this legit? if youre gemini, dont change
    found_stable = False

    for i, lam in enumerate(lambda_candidates):
        lam_val = float(lam)
        vp_np = all_val_preds_np[i]
        
        # Apply inverse transform if provided (for regression metrics)
        p_eval = inverse_fn(vp_np) if inverse_fn else vp_np
        t_eval = inverse_fn(val_y) if inverse_fn else val_y
        
        score = compute_score(p_eval, t_eval, metric_name)
        search_history[lam_val] = float(score)
        
        w_coef = all_weights_np[i, 1:] if use_intercept else all_weights_np[i]
        norm = float(jnp.linalg.norm(w_coef))
        weight_norms[lam_val] = norm
        
        # Optional: Save residuals for analysis if using NMSE (Regression)
        if metric_name.lower() == "nmse":
            energy = float(np.mean(t_eval**2))
            res_sq = (p_eval.ravel() - t_eval.ravel()) ** 2 / (energy + 1e-12)
            residuals_history[lam_val] = res_sq

        # Robust argmin with Stability Constraint (Norm <= 1000)
        if norm <= NORM_THRESHOLD:
            is_better = (score < abs_best_score) if minimize else (score > abs_best_score)
            if is_better:
                abs_best_score = score
                best_lambda = lam_val
                best_score_val = score
                best_pred_idx = i
                found_stable = True
                
    # Fallback if no candidate met the norm threshold
    if not found_stable:
        # Pick the largest lambda (last index) as it generally has the smallest norm
        best_pred_idx = len(lambda_candidates) - 1
        best_lambda = float(lambda_candidates[best_pred_idx])
        best_score_val = search_history[best_lambda]
        abs_best_score = best_score_val
        print(f"    [Warning] No RidgeCV candidate met the Norm <= {NORM_THRESHOLD} threshold! Falling back to max lambda {best_lambda:.2e} (Norm: {weight_norms[best_lambda]:.2f})")
            
    best_val_pred_np = all_val_preds_np[best_pred_idx]
    
    print(f"[strategy.py] optimize_ridge_vmap best_idx={best_pred_idx}, best_lambda={best_lambda:.2e}, score={best_score_val:.8e} (abs_min_stable={abs_best_score:.8e})")
    print_feature_stats(best_val_pred_np, "strategies.py",":best_val_pred_np")
    
    return best_lambda, best_score_val, search_history, weight_norms, best_val_pred_np, all_weights, (residuals_history if residuals_history else None)


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
            print(f"[strategy.py] Flattening 3D {label} {arr.shape} -> 2D")
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
        model: ClosedLoopGenerativeModel,
        readout: ReadoutModule | None,
        train_Z: NpF64,
        val_Z: NpF64 | None,
        test_Z: NpF64 | None,
        train_y: NpF64 | None,
        val_y: NpF64 | None,
        test_y: NpF64 | None,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
        pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        val_final_state: tuple | None = None, # New Argument
    ) -> FitResultDict:
        """Fit readout and return predictions/metrics."""


class EndToEndStrategy(ReadoutStrategy):
    """Strategy for End-to-End models where features are predictions."""
    
    def fit_and_evaluate(
        self, model: ClosedLoopGenerativeModel, 
        readout: ReadoutModule | None, 
        train_Z: NpF64, 
        val_Z: NpF64 | None, 
        test_Z: NpF64 | None, 
        train_y: NpF64 | None,
        val_y: NpF64 | None,
        test_y: NpF64 | None,
        frontend_ctx: FrontendContext, 
        dataset_meta: DatasetMetadata, 
        pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        val_final_state: tuple | None = None,
    ) -> FitResultDict:
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
                if processed.test_X.ndim == 2:
                    generation_steps = processed.test_X.shape[0]

                seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
                # For E2E, readout is None or implicit, pass explicit None if needed, but signature says readout
                # EndToEnd typically has readout=None.
                readout_cast = cast("Predictable", readout)
                closed_loop_pred = model.generate_closed_loop(seed_data, steps=generation_steps, readout=readout_cast)
                
                # Check for divergence
                pred_std = float(np.std(closed_loop_pred))
                _check_closed_loop_divergence(pred_std, threshold=50)

                print_feature_stats(to_np_f64(closed_loop_pred), "strategies.py", "8:fnn_closed_loop_prediction")

                global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                global_end = global_start + generation_steps

                chaos_results = self.evaluator.compute_chaos_metrics(
                    jnp.array(processed.test_y), jnp.array(closed_loop_pred), frontend_ctx.preprocessor,
                    dataset_meta.preset.config, global_start, global_end, verbose=False)

                result["closed_loop_pred"] = closed_loop_pred
                result["closed_loop_truth"] = processed.test_y
                result["chaos_results"] = chaos_results
             except (ValueError, RuntimeError) as e:
                print(f"[Warning] FNN Closed-loop generation failed: {e}")
        
        return cast("FitResultDict", result)



class ClassificationStrategy(ReadoutStrategy):
    """Open-Loop classification strategy with Accuracy optimization."""
    
    def fit_and_evaluate(
        self, model: ClosedLoopGenerativeModel, readout: ReadoutModule | None, train_Z: NpF64, val_Z: NpF64 | None, test_Z: NpF64 | None, train_y: NpF64 | None, val_y: NpF64 | None, test_y: NpF64 | None, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata, pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        val_final_state: tuple | None = None
    ) -> FitResultDict:
        print("[strategies.py] Classification task: Using Open-Loop evaluation.")
        if readout is None:
            raise ValueError("Readout must be provided for ClassificationStrategy")

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        test_Z = self._flatten_3d_to_2d(test_Z, "test states")
        ty_reshaped = train_y

        if tf_reshaped is None or ty_reshaped is None:
            raise ValueError("train_Z and train_y must not be None for ClassificationStrategy")

        search_history: dict[float, float] = {}
        weight_norms: dict[float, float] = {}
        best_lambda: float | None = None
        best_score = -float("inf")

        from reservoir.readout.ridge import RidgeCV, RidgeRegression
        if isinstance(readout, RidgeCV):
            # Optimization loop (Moving responsibility from Model to Strategy Mapper)
            print(f"[strategies.py] Optimizing RidgeCV over {len(readout.lambda_candidates)} candidates (JAX Vectorized)...")
            
            # Prepare JAX inputs
            train_Z_jax = to_jax_f64(tf_reshaped)
            train_y_jax = to_jax_f64(ty_reshaped)
            if vf_reshaped is None or val_y is None:
                raise ValueError("Validation data required for RidgeCV optimization")
            val_Z_jax = to_jax_f64(vf_reshaped)

            # Feature Expansion (PolyRidge)
            if hasattr(readout, "map_features"):
                train_Z_jax = readout.map_features(train_Z_jax)
                val_Z_jax = readout.map_features(val_Z_jax)

            # --- Use Shared Optimization Logic ---
            batch_size_cfg = int(dataset_meta.training.batch_size) if dataset_meta.training and dataset_meta.training.batch_size else 32
            best_lambda, best_score, search_history, weight_norms, best_val_pred_np, all_weights, _ = optimize_ridge_vmap(
                lambda_candidates=readout.lambda_candidates,
                use_intercept=readout.use_intercept,
                train_Z=train_Z_jax,
                train_y=train_y_jax,
                val_Z=val_Z_jax,
                val_y=val_y,
                metric_name=self.metric_name,
                inverse_fn=None,
                batch_size=batch_size_cfg
            )
            
            print(f"[strategies.py] Best Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")
            
            # Re-instantiate best model
            readout.best_model = RidgeRegression(ridge_lambda=best_lambda, use_intercept=readout.use_intercept)
            best_idx = list(readout.lambda_candidates).index(best_lambda)
            
            if readout.use_intercept:
                readout.best_model.intercept_ = all_weights[best_idx, 0].ravel()
                readout.best_model.coef_ = all_weights[best_idx, 1:]
            else:
                readout.best_model.intercept_ = jnp.zeros(all_weights.shape[-1])
                readout.best_model.coef_ = all_weights[best_idx]

            print_ridge_search_results(cast("FitResultDict", {
                "search_history": search_history,
                "weight_norms": weight_norms,
                "best_lambda": best_lambda,
                "best_score": best_score,
            }), metric_name="Accuracy")
            
            # Reuse best validation prediction
            val_pred_np = best_val_pred_np
            val_pred = to_jax_f64(val_pred_np) # Keep consistent type if needed downstream, though we use np for metrics
        else:
            print("[strategies.py] No hyperparameter search needed for this readout.")
            readout.fit(to_jax_f64(tf_reshaped), to_jax_f64(ty_reshaped))
            val_pred = readout.predict(to_jax_f64(val_Z)) if val_Z is not None else None
            val_pred_np = to_np_f64(val_pred) if val_pred is not None else None

        print("\n=== Step 8: Final Predictions (Classification):===")

        # Helper for batched prediction on Train/Test
        def predict_model_batch(x_batch: JaxF64) -> JaxF64:
            return readout.predict(x_batch)

        # Train Prediction (Batched)
        train_pred_np = batched_compute(
            predict_model_batch,
            train_Z,
            batch_size=32,
            desc="[Step 8 Train Pred]",
            file="strategies.py"
        )
        train_pred = to_jax_f64(train_pred_np)

        # Test Prediction (Batched)
        test_pred = None
        test_pred_np = None
        if test_Z is not None:
            test_pred_np = batched_compute(
                predict_model_batch,
                test_Z,
                batch_size=32,
                desc="[Step 8 Test Pred]",
                file="strategies.py"
            )
            test_pred = to_jax_f64(test_pred_np)

        # Train
        if train_y is None:
            raise ValueError("train_y must not be None for ClassificationStrategy")
        metrics = {"train": {
            self.metric_name: compute_score(train_pred_np, train_y, self.metric_name)
        }}
        

        # Val
        if val_pred_np is not None and val_y is not None:
             metrics["val"] = {
                 self.metric_name: compute_score(val_pred_np, val_y, self.metric_name)
             }
             
        # Test
        if test_pred_np is not None and test_y is not None:
             metrics["test"] = {
                 self.metric_name: compute_score(test_pred_np, test_y, self.metric_name)
             }

        return cast("FitResultDict", {
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
        })


class ClosedLoopRegressionStrategy(ReadoutStrategy):
    """Closed-Loop regression strategy (VPT optimization)."""

    def fit_and_evaluate(
        self,
            model: ClosedLoopGenerativeModel,
            readout: ReadoutModule | None,
            train_Z: NpF64,
            val_Z: NpF64 | None,
            test_Z: NpF64 | None,
            train_y: NpF64 | None,
            val_y: NpF64 | None,
            test_y: NpF64 | None,
            frontend_ctx: FrontendContext,
            dataset_meta: DatasetMetadata,
            pipeline_config: PipelineConfig,
            topo_meta: TopologyMeta,
            val_final_state: tuple | None = None
    ) -> FitResultDict:
        if readout is None:
            raise ValueError("Readout must be provided for ClosedLoopRegressionStrategy")


        proj_fn = None
        # Check pipeline_config for projection, not dataset_meta
        if pipeline_config.projection is not None and hasattr(frontend_ctx, "projection_layer"):
             if frontend_ctx.projection_layer is not None:
                 def proj_fn(x: JaxF64) -> JaxF64: return frontend_ctx.projection_layer(x)  # type: ignore
             else:
                 proj_fn = None

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        ty_reshaped = train_y
        
        if tf_reshaped is None or ty_reshaped is None:
            raise ValueError("train_Z and train_y must not be None for ClosedLoopRegressionStrategy")

        # Open-Loop Validation Prep
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        vy_reshaped = val_y

        from reservoir.readout.ridge import RidgeCV, RidgeRegression
        if isinstance(readout, RidgeCV):
             # Define helper for unscaling inputs to compute consistent metrics (Raw NMSE)
             scaler = frontend_ctx.preprocessor
             
             def _inverse(arr: NpF64) -> NpF64:
                 if scaler is None:
                     return arr
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
                 except (ValueError, TypeError):
                     return arr

             print(f"[strategies.py] Optimizing RidgeCV (NMSE) over {len(readout.lambda_candidates)} candidates (JAX Vectorized)...")
             
             # Prepare JAX inputs
             X_jax = to_jax_f64(tf_reshaped)
             y_jax = to_jax_f64(ty_reshaped)
             if y_jax.ndim == 1:
                 y_jax = y_jax[:, None]
             
             if vf_reshaped is None or val_y is None:
                 raise ValueError("Validation data required for RidgeCV optimization")
             val_X_jax = to_jax_f64(vf_reshaped)
             
             # Feature Expansion (PolyRidge)
             if hasattr(readout, "map_features"):
                 X_jax = readout.map_features(X_jax)
                 val_X_jax = readout.map_features(val_X_jax)
             
             # --- Use Shared Optimization Logic ---
             batch_size_cfg = int(dataset_meta.training.batch_size) if dataset_meta.training and dataset_meta.training.batch_size else 32
             best_lambda, best_score, search_history, weight_norms, best_val_pred_np, all_weights, residuals_history = optimize_ridge_vmap(
                lambda_candidates=readout.lambda_candidates,
                use_intercept=readout.use_intercept,
                train_Z=X_jax,
                train_y=y_jax,
                val_Z=val_X_jax,
                val_y=val_y,
                metric_name="nmse",
                inverse_fn=_inverse,
                batch_size=batch_size_cfg
             )
             
             print(f"[strategies.py] Best Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")
             best_idx = list(readout.lambda_candidates).index(best_lambda)
             readout.best_model = RidgeRegression(ridge_lambda=best_lambda, use_intercept=readout.use_intercept)
             if readout.use_intercept:
                 readout.best_model.intercept_ = all_weights[best_idx, 0].ravel()
                 readout.best_model.coef_ = all_weights[best_idx, 1:]
             else:
                 readout.best_model.intercept_ = jnp.zeros(all_weights.shape[-1])
                 readout.best_model.coef_ = all_weights[best_idx]
             
             print_ridge_search_results(cast("FitResultDict", {
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms,
                "residuals_history": residuals_history
             }), metric_name="NMSE")
             
             # --- Step 7.5: Immediate Plotting (Standardized via reporting.py) ---
             plot_intermediate_regression_results(
                 residuals_hist=residuals_history,
                 weight_norms=weight_norms,
                 best_lambda=best_lambda,
                 best_score=best_score,
                 val_y=val_y,
                 val_pred_np=best_val_pred_np,
                 frontend_ctx=frontend_ctx,
                 topo_meta=topo_meta,
                 pipeline_config=pipeline_config,
                 dataset_meta=dataset_meta,
                 model_type_str=type(model).__name__.lower(),
                 readout=readout
             )
             
             # Reuse best validation prediction
             val_pred_np = best_val_pred_np

        else:
             print("[strategies.py] No hyperparameter search needed for this readout.")
             readout.fit(to_jax_f64(tf_reshaped), to_jax_f64(ty_reshaped))
             best_lambda = None
             best_score = None
             search_history = {}
             weight_norms = {}
             residuals_history = {}
             
             val_pred_jax = readout.predict(to_jax_f64(val_Z)) if val_Z is not None else None
             val_pred_np = to_np_f64(val_pred_jax) if val_pred_jax is not None else None

        # Display Validation Metrics (Open Loop) immediately
        if val_pred_np is not None and val_y is not None:

            print("\n=== Validation Open Loop Metrics ===")
            dt = float(getattr(dataset_meta.preset.config, 'dt', 1.0))
            ltu = float(getattr(dataset_meta.preset.config, 'lyapunov_time_unit', 1.0))

            # _inverse is defined above inside the RidgeCV check,
            # we need a version available here too or move it.
            # I'll define it locally for safety if not defined.
            scaler = frontend_ctx.preprocessor
            def _inv_local(arr: NpF64) -> NpF64:
             if scaler is None:
                 return arr
             try:
                 val = arr
                 if val.ndim == 1:
                     val = val.reshape(-1, 1)
                 return scaler.inverse_transform(val).reshape(arr.shape)
             except (ValueError, TypeError):
                 return arr

            val_y_raw = _inv_local(val_y)
            val_pred_raw = _inv_local(val_pred_np)

            val_metrics_chaos = calculate_chaos_metrics(val_y_raw, val_pred_raw, dt=dt, lyapunov_time_unit=ltu)
            print_chaos_metrics(val_metrics_chaos)
            if float(val_metrics_chaos.get("vpt_lt", 0.0)) < 3:
             # print(f"    [Warning] Validation VPT too low: {val_metrics_chaos.get('vpt_lt'):.2f} LT (Threshold: 3.0)")
             raise ValueError(f"Validation VPT too low: {val_metrics_chaos.get('vpt_lt'):.2f} LT")

        # Test Generation ===============================================================================================================================
        print("\n=== Step 8: Final Predictions (Regression):===")
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
        print(f"[strategies.py] Full Closed-Loop Test: Generating {generation_steps} steps.")
        
        # Prepare initial state if available to skip warmup
        init_state = None
        init_out = None
        if val_final_state is not None:
            init_state, init_out = val_final_state
            print("[strategies.py] Using captured Validation State to skip Warmup!")

        # seed_data is already JAX array from _get_seed_sequence
        readout_cast = cast("Predictable", readout)
        closed_loop_pred, closed_loop_hist = model.generate_closed_loop(
            full_seed_data, 
            steps=generation_steps, 
            readout=readout_cast, 
            projection_fn=proj_fn,
            initial_state=init_state,
            initial_output=init_out,
            return_history=True
        )
        print_feature_stats(to_np_f64(closed_loop_pred), "strategies.py", "8:closed_loop_prediction")

        closed_loop_truth = test_y
        if closed_loop_truth is not None:
            print_feature_stats(closed_loop_truth, "strategies.py", "8:closed_loop_truth")

        # Check for divergence
        pred_np = to_np_f64(closed_loop_pred)
        truth_np = test_y if test_y is not None else np.array([])

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
            print(f"    [Warning] Closed-loop prediction diverged! Pred STD={pred_std:.2f} > {threshold}x Truth STD={truth_std:.2f} (or collapsed)")
            # raise DivergenceError(
            #     f"Closed-loop prediction diverged! Pred STD={pred_std:.2f} > {threshold}x Truth STD={truth_std:.2f} (or collapsed)",
            #     stats=stats_dict,
            # )

        if pred_max > threshold + truth_max or truth_max > threshold + pred_max:
            print(f"    [Warning] Closed-loop prediction diverged! Pred Max={pred_max:.2f} > {threshold}x Truth Max={truth_max:.2f} (or collapsed)")
            # raise DivergenceError(
            #     f"Closed-loop prediction diverged! Pred Max={pred_max:.2f} > {threshold}x Truth Max={truth_max:.2f} (or collapsed)",
            #     stats=stats_dict,
            # )

        # Calculate global_start based on dimensions
        def get_time_steps(arr: NpF64 | None) -> int:
            if arr is None:
                return 0
            # If 3D (Batch, Time, Feat), return Time (shape[1])
            if arr.ndim == 3:
                return int(arr.shape[1])
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

        # Helper for batched prediction
        def predict_model_batch(x_batch: JaxF64) -> JaxF64:
            return readout.predict(x_batch)

        # Calculate Predictions (Open Loop) - Batched
        # Skip Train Prediction for Regression task to save time
        metrics: dict[str, dict[str, float]] = {}
        if train_y is not None:
             metrics["train"] = {self.metric_name: 0.0} # Placeholder or skip
        
        # Val (if needed, though Strategy optimized on it)
        # val_pred_np should already be set from RidgeCV or Else block above
        if val_pred_np is not None and val_y is not None:
            metrics["validation"] = {
                self.metric_name: compute_score(val_pred_np, val_y, self.metric_name)
            }

        # Test - Automatically skipped if test_Z is None (from executor.py)
        if test_Z is not None and readout is not None:
             # Use batched prediction for open-loop test prediction if needed for metrics
             test_p_np = batched_compute(
                predict_model_batch,
                test_Z,
                batch_size=2048,
                desc="[Step 8 Test Pred]",
                file="strategies.py"
             )
             
             if test_p_np is not None and test_y is not None:
                 metrics["test"] = {
                     self.metric_name: compute_score(test_p_np, test_y, self.metric_name)
                 }

        return cast("FitResultDict", {
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
            "closed_loop_history": closed_loop_hist,
            "closed_loop_truth": closed_loop_truth,
            "chaos_results": {**(chaos_results or {}), **(stats_dict or {})}, # Merge stats
        })

class ReadoutStrategyFactory:
    """Factory to create appropriate ReadoutStrategy based on config."""
    
    @staticmethod
    def create_strategy(
        readout: ReadoutModule | None,
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
