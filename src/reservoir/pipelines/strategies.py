"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/strategies.py"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import jax.numpy as jnp
import numpy as np

from reservoir.pipelines.config import FrontendContext, DatasetMetadata
from reservoir.utils.reporting import print_ridge_search_results, print_feature_stats
from reservoir.utils.metrics import compute_score
from reservoir.pipelines.evaluation import Evaluator

class ReadoutStrategy(ABC):
    """Abstract base class for readout fitting and evaluation strategies."""
    
    def __init__(self, evaluator: Evaluator, metric_name: str):
        self.evaluator = evaluator
        self.metric_name = metric_name

    @staticmethod
    def _flatten_3d_to_2d(arr: Optional[Union[jnp.ndarray, np.ndarray, Any]], label: str = "array") -> Optional[Union[jnp.ndarray, np.ndarray]]:
        """Flatten 3D states (Batch, Time, Features) -> 2D (Batch, Features)."""
        if arr is None:
            return None
        if arr.ndim == 3:
            print(f"    [Runner] Flattening 3D {label} {arr.shape} -> 2D")
            return arr.reshape(arr.shape[0], -1)
        return arr

    @staticmethod
    def _get_seed_sequence(train_X: Union[jnp.ndarray, np.ndarray, Any], val_X: Optional[Union[jnp.ndarray, np.ndarray, Any]]):
        """Prepare seed for closed-loop (concat train+val)."""
        if val_X is not None:
            axis = 1 if train_X.ndim == 3 else 0
            return jnp.concatenate([jnp.asarray(train_X, dtype=jnp.float32), jnp.asarray(val_X, dtype=jnp.float32)], axis=axis)
        return jnp.asarray(train_X, dtype=jnp.float32)

    @abstractmethod
    def fit_and_evaluate(
        self,
        model: Any,
        readout: Any,
        train_Z: Union[jnp.ndarray, np.ndarray],
        val_Z: Optional[Union[jnp.ndarray, np.ndarray]],
        test_Z: Optional[Union[jnp.ndarray, np.ndarray]],
        train_y: Optional[Union[jnp.ndarray, np.ndarray]],
        val_y: Optional[Union[jnp.ndarray, np.ndarray]],
        test_y: Optional[Union[jnp.ndarray, np.ndarray]],
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
        pipeline_config: Any
    ) -> Dict[str, Any]:
        """Fit readout and return predictions/metrics."""
        pass


class EndToEndStrategy(ReadoutStrategy):
    """Strategy for End-to-End models where features are predictions."""
    
    def fit_and_evaluate(
        self,model, readout, train_Z, val_Z, test_Z, train_y, val_y, test_y, frontend_ctx, dataset_meta, pipeline_config
    ) -> Dict[str, Any]:
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
        self, model, readout, train_Z, val_Z, test_Z, train_y, val_y, test_y, frontend_ctx, dataset_meta, pipeline_config
    ) -> Dict[str, Any]:
        print("    [Runner] Classification task: Using Open-Loop evaluation.")
        
        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        test_Z = self._flatten_3d_to_2d(test_Z, "test states")
        ty_reshaped, vy_reshaped = train_y, val_y
        
        search_history = {}
        weight_norms = {}
        best_lambda = None
        best_score = -float("inf")

        if hasattr(readout, 'fit_with_validation'):
            # Use unified optimization
            # Define scoring callback
            def scoring_fn(p, t):
                return compute_score(p, t, self.metric_name)

            best_lambda, best_score, search_history, weight_norms, _ = readout.fit_with_validation(
                train_Z=tf_reshaped, 
                train_y=ty_reshaped, 
                val_Z=vf_reshaped, 
                val_y=vy_reshaped, 
                scoring_fn=scoring_fn, 
                maximize_score=True
            )
            
            print_ridge_search_results({
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms
            }, metric_name="Accuracy")
        else:
            print("    [Runner] No hyperparameter search needed for this readout.")
            readout.fit(tf_reshaped, ty_reshaped)

        print("\n=== Step 8: Final Predictions:===")

        # Calculate Predictions
        train_pred = readout.predict(train_Z)
        val_pred = readout.predict(val_Z) if val_Z is not None else None
        test_pred = readout.predict(test_Z) if test_Z is not None else None

        # Train
        metrics = {"train": {
            self.metric_name: compute_score(train_pred, train_y, self.metric_name)
        }}
        

        # Val
        if val_pred is not None and val_y is not None:
             metrics["val"] = {
                 self.metric_name: compute_score(val_pred, val_y, self.metric_name)
             }
             
        # Test
        if test_pred is not None and test_y is not None:
             metrics["test"] = {
                 self.metric_name: compute_score(test_pred, test_y, self.metric_name)
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
        self, model, readout, train_Z, val_Z, test_Z, train_y, val_y, test_y, frontend_ctx, dataset_meta, pipeline_config
    ) -> Dict[str, Any]:


        proj_fn = None
        # Check pipeline_config for projection, not dataset_meta
        if pipeline_config.projection is not None and hasattr(frontend_ctx, "projection_layer"):
             def proj_fn(x): return frontend_ctx.projection_layer(x)

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        ty_reshaped = train_y
        
        # Open-Loop Validation Prep
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        vy_reshaped = val_y

        if hasattr(readout, 'fit_with_validation'):
             # Define scoring callback
             def scoring_fn(p, t):
                 return compute_score(p, t, "nmse")

             # Use unified optimization (Minimize MSE)
             best_lambda, best_score, search_history, weight_norms, residuals_history = readout.fit_with_validation(
                train_Z=tf_reshaped,
                train_y=ty_reshaped, 
                val_Z=vf_reshaped, 
                val_y=vy_reshaped, 
                scoring_fn=scoring_fn, 
                maximize_score=False
             )
             
             print_ridge_search_results({
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms,
                "residuals_history": residuals_history
             }, metric_name="NMSE")
        else:
             print("    [Runner] No hyperparameter search needed for this readout.")
             readout.fit(tf_reshaped, ty_reshaped)
             best_lambda = None
             best_score = None
             search_history = {}
             weight_norms = {}

        # Test Generation
        print("\n=== Step 8: Final Predictions:===")
        closed_loop_pred = None
        closed_loop_truth = None
        chaos_results = None

        # Check model type for compatibility (simplified check)
        # Assuming "reservoir", "distillation", "passthrough" are all capable if they are passed here.
        # But we can safeguard.
        try:
             processed = frontend_ctx.processed_split
             if hasattr(processed.test_X, "shape"):
                 if processed.test_X.ndim == 3:
                     generation_steps = processed.test_X.shape[1]
                 else:
                     generation_steps = processed.test_X.shape[0]
             else:
                 generation_steps = 0
             
             full_seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
             print(f"    [Runner] Full Closed-Loop Test: Generating {generation_steps} steps.")
             
             closed_loop_pred = model.generate_closed_loop(
                 full_seed_data, steps=generation_steps, readout=readout, projection_fn=proj_fn
             )
             print_feature_stats(closed_loop_pred, "8:closed_loop_prediction")
             closed_loop_truth = test_y
             print_feature_stats(closed_loop_truth, "8:closed_loop_truth")

             # Calculate global_start based on dimensions
             def get_time_steps(arr):
                 if arr is None: return 0
                 # If 3D (Batch, Time, Feat), return Time (shape[1])
                 if arr.ndim == 3: return arr.shape[1]
                 # If 2D (Time, Feat) or (Batch, Feat) - assuming Time for Series
                 # For Reservoir time-series Time is usually axis 0 in 2D
                 return arr.shape[0]

             train_steps = get_time_steps(processed.train_X)
             val_steps_count = get_time_steps(processed.val_X)
             global_start = train_steps + val_steps_count
             global_end = global_start + generation_steps
             
             if closed_loop_truth is not None:
                 chaos_results = self.evaluator.compute_chaos_metrics(
                     jnp.array(closed_loop_truth), jnp.array(closed_loop_pred), frontend_ctx.preprocessor,
                     dataset_meta.preset.config, global_start, global_end
                 )

        except Exception as e:
            print(f"[Warning] Closed-loop generation failed: {e}")
            import traceback
            traceback.print_exc()

        # Calculate Predictions (Open Loop)
        train_pred = readout.predict(train_Z)

        # Train
        metrics = {"train": {
            self.metric_name: compute_score(train_pred, train_y, self.metric_name)
        }}
        

        # Val (if needed, though Strategy optimized on it)
        # Note: Strategy optimization loop computed best_score, but we can recompute or use it. 
        # Using separate predict calls ensures consistency.
        val_pred = None
        if val_Z is not None:
             val_pred = readout.predict(val_Z)
             metrics["val"] = {
                 self.metric_name: compute_score(val_pred, val_y, self.metric_name)
             }

        # Test
        test_pred = None
        if test_Z is not None:
             test_pred = readout.predict(test_Z)
             metrics["test"] = {
                 self.metric_name: compute_score(test_pred, test_y, self.metric_name)
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
            "residuals_history": residuals_history if 'residuals_history' in locals() else None,
            "closed_loop_pred": closed_loop_pred,
            "closed_loop_truth": closed_loop_truth,
            "chaos_results": chaos_results,
        }

class ReadoutStrategyFactory:
    """Factory to create appropriate ReadoutStrategy based on config."""
    
    @staticmethod
    def create_strategy(
        readout: Optional[Any],
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
