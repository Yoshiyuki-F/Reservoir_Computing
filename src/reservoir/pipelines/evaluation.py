import numpy as np
from reservoir.core.types import JaxF64, EvalMetrics
from reservoir.utils.metrics import calculate_chaos_metrics
from reservoir.utils.reporting import print_chaos_metrics
from reservoir.layers.preprocessing import Preprocessor
from reservoir.data.config import BaseDatasetConfig

class Evaluator:
    """Encapsulates evaluation logic including inverse transformation and metric calculation."""

    @staticmethod
    def compute_chaos_metrics(
        truth: JaxF64,
        pred: JaxF64,
        scaler: Preprocessor | None,
        dataset_config: BaseDatasetConfig,
        global_start: int,
        global_end: int,
        verbose: bool
    ) -> EvalMetrics | None:
        """Compute VPT, NDEI, and other chaos metrics with inverse transform."""
        if scaler is None:
            return None

        shape_pred = pred.shape
        shape_truth = truth.shape
        # Flatten -> Inverse Transform -> Reshape back
        pred_np = np.asarray(pred)
        truth_np = np.asarray(truth)
        pred_raw = scaler.inverse_transform(pred_np.reshape(-1, shape_pred[-1])).reshape(shape_pred)
        truth_raw = scaler.inverse_transform(truth_np.reshape(-1, shape_truth[-1])).reshape(shape_truth)

        if verbose:
            print(f"\n[Closed-Loop Metrics] (Global Steps {global_start} -> {global_end})")

        dt = getattr(dataset_config, 'dt', 1.0)
        ltu = getattr(dataset_config, 'lyapunov_time_unit', 1.0)

        metrics = calculate_chaos_metrics(truth_raw, pred_raw, dt=dt, lyapunov_time_unit=ltu)
        if verbose:
            print_chaos_metrics(metrics)
        return metrics

    @staticmethod
    def align_targets(features: JaxF64 | None, targets: JaxF64 | None) -> JaxF64 | None:
        """Align target length (dim 0) to match feature length (warmup handling)."""
        if features is None or targets is None:
            return targets

        # 2D only: length is shape[0]
        len_f = features.shape[0]
        len_t = targets.shape[0]

        if len_f < len_t:
            diff = len_t - len_f
            return targets[diff:]
        return targets
