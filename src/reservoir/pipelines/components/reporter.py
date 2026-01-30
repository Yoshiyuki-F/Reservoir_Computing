import time
from typing import Dict, Any

from reservoir.models.presets import PipelineConfig
from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
from reservoir.utils.reporting import compute_score, generate_report


class ResultReporter:
    """
    Handles result aggregation, metric calculation, and report generation.
    """

    def __init__(self, stack: ModelStack, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata):
        self.stack = stack
        self.frontend_ctx = frontend_ctx
        self.dataset_meta = dataset_meta
        self.start_time = time.time()

    def compile_and_save(self, execution_results: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        """
        Compile final results and trigger report generation.
        """
        fit_result = execution_results["fit_result"]
        train_logs = execution_results["train_logs"]
        quantum_trace = execution_results.get("quantum_trace") # New
        processed = self.frontend_ctx.processed_split
        
        results: Dict[str, Any] = {}
        metric_name = self.stack.metric
        test_y = processed.test_y

        # Use aligned_test_y from fit_result if available (for FNN windowed mode)
        aligned_test_y = fit_result.get("aligned_test_y", test_y)

        if fit_result["closed_loop_pred"] is not None:
            print("\n    [Runner] Overwriting Test Output with Closed-Loop result.")
            test_pred = fit_result["closed_loop_pred"]
            test_y_final = fit_result["closed_loop_truth"]
            results["is_closed_loop"] = True
        else:
            test_pred = fit_result["test_pred"]
            test_y_final = aligned_test_y

        # Debug Logging for User
        print("\n[DEBUG] Inspecting Test Predictions (First 20 steps):")
        if test_pred is not None:
            tp_flat = test_pred.flatten()[:20]
            print(f"  Pred: {tp_flat}")
            print(f"  Pred Stats: min={test_pred.min():.4f}, max={test_pred.max():.4f}, mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")
        if test_y_final is not None:
            ty_flat = test_y_final.flatten()[:20]
            print(f"  True: {ty_flat}")

        test_score = 0.0
        if test_pred is not None and test_y_final is not None:
            test_score = compute_score(test_pred, test_y_final, metric_name)
            if results.get("is_closed_loop"):
                print(f"    [Runner] Closed-Loop {metric_name.upper()}: {test_score:.5f}")

        results["train"] = {
            "search_history": fit_result["search_history"],
            "weight_norms": fit_result["weight_norms"],
        }
        if fit_result["best_lambda"] is not None:
            results["train"]["best_lambda"] = fit_result["best_lambda"]

        results["test"] = {metric_name: test_score}
        if fit_result["chaos_results"] is not None:
            chaos = fit_result["chaos_results"]
            results["test"]["chaos_metrics"] = chaos
            results["test"]["vpt_lt"] = chaos.get("vpt_lt", 0.0)
            results["test"]["ndei"] = chaos.get("ndei", float("inf"))

        val_score = fit_result["best_score"] if fit_result["best_score"] is not None else 0.0
        results["validation"] = {metric_name: val_score}

        results["outputs"] = {
            "train_pred": fit_result["train_pred"],
            "test_pred": test_pred,
            "val_pred": fit_result["val_pred"],
        }

        results["readout"] = self.stack.readout
        results["scaler"] = self.frontend_ctx.scaler
        results["training_logs"] = train_logs
        results["quantum_trace"] = quantum_trace # New
        results["meta"] = {
            "metric": metric_name,
            "elapsed_sec": time.time() - self.start_time,
        }
        
        # Trigger Report Generation
        self._generate_report(results, config)

        return results

    def _generate_report(self, results: Dict[str, Any], config: PipelineConfig):
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
            preprocessors=self.frontend_ctx.preprocessors,
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
