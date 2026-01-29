from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from reservoir.pipelines.config import FrontendContext, DatasetMetadata
from reservoir.utils.reporting import print_ridge_search_results, compute_score
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
            return jnp.concatenate([jnp.array(train_X), jnp.array(val_X)], axis=axis)
        return jnp.array(train_X)

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

                global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                global_end = global_start + generation_steps

                chaos_results = self.evaluator.compute_chaos_metrics(
                    jnp.array(processed.test_y), jnp.array(closed_loop_pred), frontend_ctx.scaler,
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

        if hasattr(readout, 'ridge_lambda'):
            lambda_candidates = getattr(readout, 'lambda_candidates', None) or [readout.ridge_lambda]
            print(f"    [Runner] Running hyperparameter search over {len(lambda_candidates)} lambdas...")
            best_lambda = lambda_candidates[0]

            for lam in tqdm(lambda_candidates, desc="[Lambda Search]"):
                lam_val = float(lam)
                readout.ridge_lambda = lam_val
                readout.fit(tf_reshaped, ty_reshaped)

                val_pred_tmp = readout.predict(vf_reshaped)
                # Compute score using generic utility
                score = compute_score(np.asarray(val_pred_tmp), np.asarray(vy_reshaped), self.metric_name)
                search_history[lam_val] = float(score)

                if hasattr(readout, "coef_") and readout.coef_ is not None:
                    weight_norms[lam_val] = float(jnp.linalg.norm(readout.coef_))

                if score >= best_score:
                    best_score = score
                    best_lambda = lam_val

            readout.ridge_lambda = best_lambda
            readout.fit(tf_reshaped, ty_reshaped)
            print(f"    [Runner] Best Lambda: {best_lambda:.5e} (Accuracy: {best_score:.5f})")
            
            print_ridge_search_results({
                "search_history": search_history,
                "best_lambda": best_lambda,
                "weight_norms": weight_norms
            }, is_classification=True)
        else:
             print("    [Runner] No hyperparameter search needed for this readout.")
             readout.fit(tf_reshaped, ty_reshaped)

        return {
            "train_pred": readout.predict(train_Z),
            "val_pred": readout.predict(val_Z) if val_Z is not None else None,
            "test_pred": readout.predict(test_Z),
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
        processed = frontend_ctx.processed_split
        
        lambda_candidates = getattr(readout, 'lambda_candidates', None) or \
            ([readout.ridge_lambda] if hasattr(readout, 'ridge_lambda') else [1e-3])
        print(f"    [Runner] Starting Closed-Loop Hyperparameter Search over {len(lambda_candidates)} candidates...")

        proj_fn = None
        # Check pipeline_config for projection, not dataset_meta
        if pipeline_config.projection is not None and hasattr(frontend_ctx, "projection_layer"):
             def proj_fn(x): return frontend_ctx.projection_layer(x)

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        ty_reshaped = train_y
        
        # Prepare Validation Seeds
        val_steps = processed.val_X.shape[0] if processed.val_X is not None else 0
        seed_len = min(processed.train_X.shape[0], val_steps) if val_steps > 0 else processed.train_X.shape[0]
        seed_data = processed.train_X[-seed_len:]

        search_history = {}
        weight_norms = {}
        best_score = float("inf")
        best_lambda = lambda_candidates[0]

        for lam in tqdm(lambda_candidates, desc="[Closed-Loop Search]"):
            lam_val = float(lam)
            if hasattr(readout, "ridge_lambda"):
                readout.ridge_lambda = lam_val
            
            readout.fit(tf_reshaped, ty_reshaped)
            
            if hasattr(readout, "coef_") and readout.coef_ is not None:
                weight_norms[lam_val] = float(jnp.linalg.norm(readout.coef_))
            
            # Validation Generation
            # Note: seed_data is jnp.array already from processed or casting
            val_gen = model.generate_closed_loop(
                jnp.array(seed_data), steps=val_steps, readout=readout, projection_fn=proj_fn, verbose=False
            )
            
            # Metrics
            if val_y is not None:
                current_metrics = self.evaluator.compute_chaos_metrics(
                    jnp.array(val_y), jnp.array(val_gen), frontend_ctx.scaler, dataset_meta.preset.config, verbose=False
                )
            else:
                current_metrics = None
            
            # Score (minimize -VPT)
            score = -current_metrics.get("vpt_lt", 0.0) if current_metrics else 0.0
            search_history[lam_val] = float(score)

            if score <= best_score:
                best_score = score
                best_lambda = lam_val

        print(f"    [Runner] Best Lambda: {best_lambda:.5e} (Val VPT: {-best_score:.5f} LT)")
        print_ridge_search_results({
            "search_history": search_history,
            "best_lambda": best_lambda,
            "weight_norms": weight_norms
        }, is_classification=False)

        print(f"    [Runner] Re-fitting readout with best_lambda={best_lambda:.5e}...")
        if hasattr(readout, "ridge_lambda"):
            readout.ridge_lambda = best_lambda
        readout.fit(tf_reshaped, ty_reshaped)

        # Test Generation
        print("\n=== Step 8: Final Predictions (Inverse Transformed):===")
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
             closed_loop_truth = test_y

             global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
             global_end = global_start + generation_steps
             
             if closed_loop_truth is not None:
                 chaos_results = self.evaluator.compute_chaos_metrics(
                     jnp.array(closed_loop_truth), jnp.array(closed_loop_pred), frontend_ctx.scaler,
                     dataset_meta.preset.config, global_start, global_end
                 )

        except Exception as e:
            print(f"[Warning] Closed-loop generation failed: {e}")
            import traceback
            traceback.print_exc()

        return {
            "train_pred": readout.predict(train_Z),
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
