"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/components/reporter.py"""
import time
import jax
import jax.numpy as jnp

from reservoir.models.presets import PipelineConfig
from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
from reservoir.utils.reporting import generate_report
from reservoir.core.types import ResultDict, to_np_f64


class ResultReporter:
    """
    Handles result aggregation, metric calculation, and report generation.
    """

    def __init__(self, stack: ModelStack, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata):
        self.stack = stack
        self.frontend_ctx = frontend_ctx
        self.dataset_meta = dataset_meta
        self.start_time = time.time()

    def compile_and_save(self, execution_results: ResultDict, config: PipelineConfig) -> ResultDict:
        """
        Compile final results and trigger report generation.
        """
        fit_result: ResultDict = dict(execution_results["fit_result"])
        train_logs = execution_results["train_logs"]
        quantum_trace = execution_results.get("quantum_trace") # New
        processed = self.frontend_ctx.processed_split
        
        results: ResultDict = {}
        metric_name = self.stack.metric
        test_y = processed.test_y

        # Use aligned_test_y from fit_result if available (for FNN windowed mode)
        aligned_test_y = fit_result.get("aligned_test_y", test_y)

        if fit_result["closed_loop_pred"] is not None:
            # Predictions from strategies might be JaxF64, convert to NpF64 for reporting
            test_pred = to_np_f64(fit_result["closed_loop_pred"])
            test_y_final = to_np_f64(fit_result["closed_loop_truth"])
            results["is_closed_loop"] = True
        else:
            test_pred = to_np_f64(fit_result["test_pred"])
            test_y_final = to_np_f64(aligned_test_y)

        # Try to use pre-calculated metrics from Strategy
        metrics: ResultDict = dict(fit_result.get("metrics", {}))
        
        # Test Score
        test_score = 0.0
        test_metrics: ResultDict = dict(metrics.get("test", {}))
        if metric_name in test_metrics:
             test_score = float(test_metrics[metric_name])
        
        # Train Score 
        # (Assuming strategy populates metrics["train"], merging it)
        results["train"] = {
            "search_history": fit_result["search_history"],
            "weight_norms": fit_result["weight_norms"],
            **dict(metrics.get("train", {}))
        }
        if fit_result["best_lambda"] is not None:
            results["train"]["best_lambda"] = fit_result["best_lambda"]
        
        # Propagate residuals history for plotting
        if "residuals_history" in fit_result:
            results["residuals_history"] = fit_result["residuals_history"]

        results["test"] = {metric_name: test_score, **test_metrics}
        if fit_result["chaos_results"] is not None:
            chaos: ResultDict = dict(fit_result["chaos_results"])
            results["test"]["chaos_metrics"] = chaos
            results["test"]["vpt_lt"] = chaos.get("vpt_lt", 0.0)
            results["test"]["ndei"] = chaos.get("ndei", float("inf"))
            results["test"]["var_ratio"] = chaos.get("var_ratio", 0.0)
            results["test"]["mse"] = chaos.get("mse", float("inf"))

        # Val Score
        val_score = 0.0
        val_metrics: ResultDict = dict(metrics.get("val", {}))
        if metric_name in val_metrics:
            val_score = float(val_metrics[metric_name])
        elif fit_result["best_score"] is not None:
             # Keep this fallback as best_score corresponds to validation during fit
             val_score = float(fit_result["best_score"])
            
        results["validation"] = {metric_name: val_score, **val_metrics}

        # Ensure all predictions and outputs are moved to Host Domain (NpF64)
        def _to_np_recursive(val):
            if isinstance(val, (jax.Array, jnp.ndarray)):
                return to_np_f64(val)
            if isinstance(val, dict):
                return {k: _to_np_recursive(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_to_np_recursive(v) for v in val]
            return val

        outputs_raw = dict(fit_result.get("outputs", {})) # strategy might have returned them
        if not outputs_raw:
             # Fallback: strategy returned them directly in fit_result keys
             outputs_raw = {
                 "train_pred": fit_result.get("train_pred"),
                 "test_pred": test_pred, # already converted above
                 "val_pred": fit_result.get("val_pred"),
             }

        results["outputs"] = _to_np_recursive(outputs_raw)

        results["readout"] = self.stack.readout
        results["preprocessor"] = self.frontend_ctx.preprocessor
        results["scaler"] = self.frontend_ctx.preprocessor  # Alias for reporting.py
        results["training_logs"] = train_logs
        results["quantum_trace"] = to_np_f64(quantum_trace) if quantum_trace is not None else None
        results["meta"] = {
            "metric": metric_name,
            "elapsed_sec": time.time() - self.start_time,
        }
        
        # Trigger Report Generation
        self._generate_report(results, config)

        return results

    def _generate_report(self, results: ResultDict, config: PipelineConfig):
        processed = self.frontend_ctx.processed_split
        report_payload = dict(
            readout=self.stack.readout,
            train_X=processed.train_X,
            train_y=processed.train_y,
            test_X=processed.test_X,
            test_y=processed.test_y,
            val_X=processed.val_X,
            val_y=processed.val_y,
            training_obj=self.dataset_meta.training,
            dataset_name=self.dataset_meta.dataset_name,
            model_type_str=self.stack.model_label,
        )
        generate_report(
            results,
            config,
            self.stack.topo_meta,
            **report_payload,
            classification=self.dataset_meta.classification,
            dataset_preset=self.dataset_meta.preset,
            model_obj=self.stack.model,
        )
