"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/reporting.py
Reporting utilities for post-run analysis: metrics, logging, and file outputs Draw and save no recalculation.
"""
from __future__ import annotations

import numpy as np
from reservoir.core.types import NpF64, ConfigDict, ResultDict, TrainLogs, EvalMetrics, to_np_f64, FitResultDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.models.generative import ClosedLoopGenerativeModel
    from reservoir.models.presets import PipelineConfig
    from reservoir.data.config import DatasetPreset
    from reservoir.training.config import TrainingConfig
    from reservoir.core.interfaces import ReadoutModule
    from reservoir.layers.preprocessing import Preprocessor
    from collections.abc import Sequence

# --- Array Formatting Helpers ---
def _format_cls_array(arr: NpF64 | None) -> NpF64 | None:
    if arr is None:
        return None
    res = arr
    if res.ndim > 1 and res.shape[-1] > 1:
        res = np.argmax(res, axis=-1)
    return res.ravel()

def _calc_acc(y_true: NpF64 | None, y_pred: NpF64 | None) -> float:
    if y_true is None or y_pred is None:
        return 0.0
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))

def _get_seq_len(arr: NpF64 | None) -> int:
    if arr is None:
        return 0
    if arr.ndim == 3:
        return int(arr.shape[1])
    if arr.ndim >= 1:
        return int(arr.shape[0])
    return 0

def _to_2d(arr: NpF64) -> NpF64:
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

# --- Metrics Calculation ---



def print_chaos_metrics(metrics: EvalMetrics, header: str | None = None) -> None:
    """
    Print chaos metrics to console.
    """
    if header:
        print(f"{header}")
    else:
        print("=== Chaos Prediction Metrics ===")
    
    # Direct access from strictly typed EvalMetrics
    # Optional fields use 0.0 or inf as defaults to prevent crashes if not present
    print(f"MSE       : {metrics.get('mse', 0.0):.5f}")
    print(f"NMSE      : {metrics.get('nmse', float('inf')):.5f}")
    print(f"NRMSE     : {metrics.get('nrmse', float('inf')):.5f}")
    print(f"MASE      : {metrics.get('mase', float('inf')):.5f}")
    print(f"NDEI      : {metrics.get('ndei', float('inf')):.5f} (Target < 0.1)")
    print(f"Var Ratio : {metrics.get('var_ratio', 0.0):.5f} (Target ~ 1.0)")
    print(f"Corr      : {metrics.get('correlation', 0.0):.5f} (Target > 0.95)")
    
    vpt_steps = int(metrics.get("vpt_steps", 0.0))
    vpt_lt = float(metrics.get("vpt_lt", 0.0))
    vpt_threshold = float(metrics.get("vpt_threshold", 0.4))
    print(f"VPT       : {vpt_steps} steps ({vpt_lt:.2f} LT) @ threshold={vpt_threshold}")




# --- Logging / Printing ---

def _print_feature_stats_impl(features: NpF64, stage: str, backend: str = "numpy") -> None:
    """Internal implementation handling concrete numpy arrays."""
    # 基本統計量
    stats = {
        "shape": features.shape,
        "dtype": f"{backend}.{features.dtype}",
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "nans": int(np.isnan(features).sum()),
    }

    print(
        f"[FeatureStats:{stage}] dtype={stats['dtype']}, shape={stats['shape']}, "
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, max={stats['max']:.4f}, nans={stats['nans']}"
    )
    if stats["std"] < 1e-6:
        print("Feature matrix has near-zero variance. Model output may be inactive.")

def print_feature_stats(features: NpF64 | None, stage: str) -> None:
    """
    特徴量の統計情報を表示する (Host Domain).
    """
    if features is None:
        print(f"[FeatureStats:{stage}(skipped)] Closed-Loop mode: using raw data")
        return

    # In Host Domain, we expect NpF64
    _print_feature_stats_impl(features, stage, backend="numpy")

def print_ridge_search_results(train_res: FitResultDict, metric_name: str = "MSE") -> None:
    history = train_res.get("search_history")
    if not history:
        return
    best_lam = train_res.get("best_lambda")
    
    weight_norms = train_res.get("weight_norms") or {}
    
    metric_label = metric_name

    # Decide best logic for marking
    # Both minimize score internally (MSE is min, -VPT is min)
    best_by_metric = min(history.keys(), key=lambda k: float(history[k]))

    best_marker = best_lam if best_lam is not None else best_by_metric

    print("\n" + "=" * 40)
    print(f"Ridge Hyperparameter Search ({metric_label})")
    print("-" * 40)
    sorted_lambdas = sorted(history.keys())
    for lam in sorted_lambdas:
        if lam is None:
            continue
        try:
             score = float(history[lam])
             lam_val = float(lam)
        except (ValueError, TypeError):
             continue
        
        # Format score for display
        score_disp = score
        label = f"Val {metric_name}"
        
        # Legacy VPT handling: if metric is exactly "VPT", we assume stored as negative
        if metric_name == "VPT":
            score_disp = -score
            label = "Val VPT"
            
        norm = weight_norms.get(lam)
        norm_str = f"(Norm: {norm:.5e})" if norm is not None else "(Norm: n/a)"
        marker = ""
        if best_marker is not None:
             try:
                 bm_val = float(best_marker)
                 if abs(lam_val - bm_val) < 1e-12:
                     marker = " <= best"
             except (ValueError, TypeError):
                 pass
        print(f"   λ = {lam_val:.2e} : {label} = {score_disp:.10f} {norm_str}{marker}")
    print("=" * 40 + "\n")


def plot_distillation_loss(training_logs: TrainLogs, save_path: str, title: str, learning_rate: float | None = None) -> None:
    loss_history = training_logs.get("loss_history")
    if not loss_history:
        return
    try:
        from reservoir.utils.plotting import plot_loss_history
    except ImportError as exc:  # pragma: no cover
        print(f"Skipping distillation loss plotting due to import error: {exc}")
        return
    loss_list = list(loss_history)
    plot_loss_history(loss_list, save_path, title=title, learning_rate=learning_rate)


def plot_classification_report(
    train_y: NpF64 | None,
    test_y: NpF64 | None,
    val_y: NpF64 | None,
    filename: str,
    model_type_str: str,
    dataset_name: str,
    results: ResultDict,
    training_obj: TrainingConfig,
    train_pred: NpF64 | None = None,
    test_pred: NpF64 | None = None,
    val_pred: NpF64 | None = None,
    selected_lambda: float | None = None,
    class_names: Sequence[str] | None = None,
) -> None:
    try:
        from reservoir.utils.plotting import plot_classification_results
    except ImportError as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    int(getattr(training_obj, "batch_size", 0) or 0)

    # ---------------------------------------------------------
    # 1. Labels Preparation
    # ---------------------------------------------------------
    train_labels_np = _format_cls_array(train_y)
    test_labels_np = _format_cls_array(test_y)
    val_labels_np = _format_cls_array(val_y)

    # ---------------------------------------------------------
    # 2. Predictions Preparation
    # ---------------------------------------------------------
    train_pred_np = _format_cls_array(train_pred)
    test_pred_np = _format_cls_array(test_pred)
    val_pred_np = _format_cls_array(val_pred)

    # ---------------------------------------------------------
    # 3. Plot
    # ---------------------------------------------------------
    acc_train = None
    acc_test = None
    acc_val = None

    if results is not None:
         train_res = results.get("train", {})
         val_res = results.get("validation", {})
         test_res = results.get("test", {})
         
         acc_train = float(train_res.get("accuracy", 0.0))
         acc_val = float(val_res.get("accuracy", 0.0))
         acc_test = float(test_res.get("accuracy", 0.0))

    if acc_train == 0.0:
        acc_train = _calc_acc(train_labels_np, train_pred_np)
    
    if acc_test == 0.0:
        acc_test = _calc_acc(test_labels_np, test_pred_np)
    
    if acc_val == 0.0:
        acc_val = _calc_acc(val_labels_np, val_pred_np) if val_labels_np is not None else 0.0
    
    print("\n[Report] Accuracy Check (Pre-Plot):")
    print(f"  Train: {acc_train:.4%}")
    print(f"  Val  : {acc_val:.4%}")
    print(f"  Test : {acc_test:.4%}")

    # Extract lambda_norm from weight_norms for the selected lambda
    lambda_norm = None
    if selected_lambda is not None and results is not None:
        train_res = results.get("train", {})
        weight_norms = train_res.get("weight_norms", {})
        lambda_norm = float(weight_norms.get(selected_lambda, 0.0))

    metrics_test = results.get("test", {}) if results is not None else {}
    metrics_payload = {k: v for k, v in metrics_test.items()}
    if train_labels_np is None or test_labels_np is None or train_pred_np is None or test_pred_np is None:
        print("    [Reporter] Missing data for classification plot, skipping.")
        return

    plot_classification_results(
        train_labels=train_labels_np,
        test_labels=test_labels_np,
        train_predictions=train_pred_np,
        test_predictions=test_pred_np,
        val_labels=val_labels_np,
        val_predictions=val_pred_np,
        title=f"{model_type_str.upper()} on {dataset_name}",
        filename=filename,
        metrics_info=metrics_payload,
        best_lambda=selected_lambda,
        lambda_norm=lambda_norm,
        class_names=class_names,
    )


def _get_preprocess_label(topo_meta: ConfigDict, config: PipelineConfig | None) -> str:
    details = topo_meta.get("details", {})
    
    raw_label = str(details.get("preprocess", ""))
    if raw_label == "CustomRangeScaler":
        scale = 0.0
        centering = False
        if config is not None and hasattr(config, "preprocess"):
            scale = float(getattr(config.preprocess, "input_scale", 0.0))
            centering = bool(getattr(config.preprocess, "centering", False))
        return f"{'T' if centering else 'F'}CRS{scale}"
    elif raw_label == "MinMaxScaler":
        f_min, f_max = 0.0, 1.0
        if config is not None and hasattr(config, "preprocess"):
            f_min = float(getattr(config.preprocess, "feature_min", 0.0))
            f_max = float(getattr(config.preprocess, "feature_max", 1.0))
        return f"Min{f_min:.2f}Max{f_max:.2f}"
    elif raw_label == "AffineScaler":
        input_scale, shift = 1.0, 0.0
        if config is not None and hasattr(config, "preprocess"):
            input_scale = float(getattr(config.preprocess, "input_scale", 1.0))
            shift = float(getattr(config.preprocess, "shift", 0.0))
        return f"Affine_a{input_scale:.2f}_b{shift:.2f}"
    
    return raw_label if raw_label else "raw"


def _get_projection_label(config: PipelineConfig, topo_meta: ConfigDict) -> str | None:
    if not hasattr(config, 'projection') or config.projection is None:
        return None
    
    proj_config = config.projection
    proj_type = str(getattr(proj_config, "type", "")).lower()
    proj_units = int(getattr(proj_config, "n_units", 0))
    
    if proj_type == "random":
        input_scale = float(getattr(proj_config, "input_scale", 0.0))
        input_conn = float(getattr(proj_config, "input_connectivity", 0.0))
        bias_scale = float(getattr(proj_config, "bias_scale", 0.0))
        return f"RP{proj_units}_is{input_scale:.2f}_c{input_conn:.2f}_bs{bias_scale:.2f}"
    elif proj_type == "center_crop":
        return f"CCP{proj_units}"
    elif proj_type == "resize":
        return f"Res{proj_units}"
    elif proj_type == "polynomial":
        shapes = topo_meta.get("shapes", {})
        projected_shape = shapes.get("projected")
        poly_output = 0
        if isinstance(projected_shape, (list, tuple)) and len(projected_shape) > 0:
            p_out = projected_shape[-1]
            if isinstance(p_out, (int, float, str, bool)):
                poly_output = int(p_out)
        return f"Poly{poly_output}"
    elif proj_type == "pca":
        return f"PCA{proj_units}"
    elif proj_type == "angle_embedding":
        freq = float(getattr(proj_config, "frequency", 0.0))
        phase = float(getattr(proj_config, "phase_offset", 0.0))
        return f"AEP{proj_units}f{freq}p{phase}"
    elif proj_units:
        return f"Proj{proj_units}"
    return None


def _infer_filename_parts(topo_meta: ConfigDict, training_obj: TrainingConfig, model_type_str: str, readout: ReadoutModule | None = None, config: PipelineConfig | None = None) -> list[str]:
    type_lower = str(model_type_str).lower()
    is_fnn = "fnn" in type_lower
    student_layers = None
    
    details = topo_meta.get("details", {})
    student_layers = details.get("student_layers")
    topo_type = str(topo_meta.get("type", "")).lower()
    is_fnn = is_fnn or "fnn" in topo_type or "rnn" in topo_type or "nn" in topo_type

    preprocess_label = _get_preprocess_label(topo_meta, config)

    # Append feedback_scale to model_type_str for quantum models
    if config is not None:
        model_cfg = getattr(config, 'model', None)
        if model_cfg:
            has_feedback = hasattr(model_cfg, 'feedback_scale') and model_cfg.feedback_scale is not None
            has_leak = hasattr(model_cfg, 'leak_rate') and model_cfg.leak_rate is not None
            if has_feedback:
                n_qubits = getattr(model_cfg, "n_qubits", None)
                if n_qubits is None and hasattr(config, "projection") and config.projection:
                    n_qubits = getattr(config.projection, "n_units", None)
                q_str = f"q{n_qubits}" if n_qubits is not None else "q?"
                basis = str(getattr(model_cfg, "measurement_basis", "Z"))
                model_type_str = f"{model_type_str}_{q_str}_f{float(model_cfg.feedback_scale)}_{basis}"
            elif has_leak:
                sr = getattr(model_cfg, 'spectral_radius', None)
                lr = float(getattr(model_cfg, 'leak_rate', 1.0))
                rc_conn = getattr(model_cfg, 'rc_connectivity', None)
                tag = f"_sr{float(sr):.2f}" if sr is not None else ""
                tag += f"_lr{lr:.2f}"
                tag += f"_rc{float(rc_conn):.2f}" if rc_conn is not None else ""
                model_type_str = f"{model_type_str}{tag}"

    filename_parts: list[str] = [model_type_str, preprocess_label]

    # Window Size marker (for WindowsFNN/TDE)
    if config is not None:
        model_cfg = getattr(config, 'model', None)
        if hasattr(model_cfg, 'window_size') and model_cfg.window_size is not None:
             filename_parts.append(f"k{int(model_cfg.window_size)}")
        elif hasattr(model_cfg, 'student') and hasattr(model_cfg.student, 'window_size') and model_cfg.student.window_size is not None:
             filename_parts.append(f"k{int(model_cfg.student.window_size)}")

    # Projection marker
    if config is not None:
        proj_lbl = _get_projection_label(config, topo_meta)
        if proj_lbl:
            filename_parts.append(proj_lbl)

    # Readout type suffix
    if readout is not None:
        readout_type = type(readout).__name__
        if hasattr(readout, 'hidden_layers') and readout.hidden_layers:
            layers_str = "-".join(str(int(v)) for v in readout.hidden_layers)
            lr = float(getattr(training_obj, 'learning_rate', 0.0)) if training_obj else 0.0
            if lr > 0:
                filename_parts.append(f"{readout_type}{layers_str}_LR{lr:.0e}")
            else:
                filename_parts.append(f"{readout_type}{layers_str}")
        else:
            filename_parts.append(f"{readout_type}RO")

    # NN marker
    if is_fnn:
        layers_list = student_layers if isinstance(student_layers, (list, tuple)) else []
        layers = tuple(int(v) for v in layers_list if isinstance(v, (int, float, str, bool)) and v is not None)
        if layers:
            filename_parts.append(f"nn{'-'.join(str(int(v)) for v in layers)}")
        else:
            filename_parts.append("nn0")
        filename_parts.append(f"epochs{int(getattr(training_obj, 'epochs', 0) or 0)}")
        
    return filename_parts


def generate_report(
    results: ResultDict,
    config: PipelineConfig,
    topo_meta: ConfigDict,
    readout: ReadoutModule | None,
    train_y: NpF64 | None,
    test_y: NpF64 | None,
    val_y: NpF64 | None,
    training_obj: TrainingConfig,
    dataset_name: str,
    model_type_str: str,
    classification: bool = False,
    # preprocessors removed
    dataset_preset: DatasetPreset | None = None,  # DatasetPreset for dt/lyapunov_time_unit
    model_obj: ClosedLoopGenerativeModel | None = None, # New Argument
) -> None:
    """
    Coordinator for generating all report elements (plots, logs).
    Delegates specific plotting tasks to specialized functions.
    """
    # 1. Common: Distillation Loss (if available)
    _plot_distillation_section(results, topo_meta, training_obj, model_type_str, readout, config, dataset_name)

    # 2. Main Task Plots (Classification vs Regression using MSE)
    metric = "accuracy" if classification else "mse"
    
    if classification:
        _plot_classification_section(
            results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
            train_y, test_y, val_y
        )
    elif metric == "mse":
        _plot_regression_section(
            results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
            train_y, val_y, test_y, dataset_preset
        )

    # 3. Quantum Dynamics (if available)
    _plot_quantum_section(results, topo_meta, training_obj, dataset_name, model_type_str, readout, config, model_obj)


def _plot_distillation_section(results: ResultDict, topo_meta: ConfigDict, training_obj: TrainingConfig, model_type_str: str, readout: ReadoutModule | None, config: PipelineConfig, dataset_name: str) -> None:
    training_logs = results.get("training_logs")
    if training_logs is not None:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
        lr = float(getattr(training_obj, 'learning_rate', 0.0))
        plot_distillation_loss(training_logs, loss_filename, title=f"{model_type_str.upper()} Distillation Loss", learning_rate=lr if lr > 0 else None)


def _plot_classification_section(
    results: ResultDict, config: PipelineConfig, topo_meta: ConfigDict, training_obj: TrainingConfig, dataset_name: str, model_type_str: str, readout: ReadoutModule | None,
    train_y: NpF64 | None, test_y: NpF64 | None, val_y: NpF64 | None
) -> None:
    filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    confusion_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
    
    train_res = results.get("train", {})
    selected_lambda = None
    lam_val = train_res.get("best_lambda")
    if isinstance(lam_val, (float, int)):
        selected_lambda = float(lam_val)
    
    # Extract predictions from ResultDict and ensure Host Domain (NpF64)
    outputs = results.get("outputs") or {}
    train_pred_raw = outputs.get("train_pred")
    test_pred_raw = outputs.get("test_pred")
    val_pred_raw = outputs.get("val_pred")
    
    train_p = train_pred_raw
    test_p = test_pred_raw
    val_p = val_pred_raw

    plot_classification_report(
        train_y=train_y,
        test_y=test_y,
        val_y=val_y,
        filename=confusion_filename,
        model_type_str=model_type_str,
        dataset_name=dataset_name,
        # metric removed
        selected_lambda=selected_lambda,
        results=results,
        training_obj=training_obj,
        train_pred=train_p,
        test_pred=test_p,
        val_pred=val_p,
        # preprocessors removed
    )
    
    # FNN Readout Loss Plot
    if readout is not None and hasattr(readout, 'training_logs') and readout.training_logs:
        fnn_loss_history = readout.training_logs.get("loss_history")
        if fnn_loss_history:
            loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
            lr = float(getattr(training_obj, 'learning_rate', 0.0))
            plot_distillation_loss(readout.training_logs, loss_filename, title=f"{model_type_str.upper()} FNN Readout Loss", learning_rate=lr if lr > 0 else None)


def _plot_regression_section(
    results: ResultDict, config: PipelineConfig, topo_meta: ConfigDict, training_obj: TrainingConfig, dataset_name: str, model_type_str: str, readout: ReadoutModule | None,
    train_y: NpF64 | None, val_y: NpF64 | None, test_y: NpF64 | None, dataset_preset: DatasetPreset | None
) -> None:
    filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    prediction_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_prediction.png"
    
    test_results = results.get("test")
    test_mse = 0.0
    if isinstance(test_results, dict) and test_results.get("mse") is not None:
        test_mse = float(test_results["mse"])
        
    scaler = results.get("scaler")
    is_closed_loop = bool(results.get("is_closed_loop", False))

    # Extract predictions from ResultDict and ensure Host Domain (NpF64)
    outputs = results.get("outputs") or {}
    test_pred_raw = outputs.get("test_pred")

    test_p = test_pred_raw

    # Get dt and lyapunov_time_unit for VPT calculation
    dt = None
    ltu = None
    if dataset_preset is not None:
        ds_config = getattr(dataset_preset, 'config', None)
        if ds_config is not None:
            dt = float(getattr(ds_config, 'dt', 1.0))
            ltu = float(getattr(ds_config, 'lyapunov_time_unit', 1.0))

    if readout is None:
        print("    [Reporter] Missing Readout module for regression plot. Skipping.")
        return

    plot_regression_report(
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            filename=prediction_filename,
            model_type_str=model_type_str,
            mse=test_mse if test_mse > 0 else None,
            test_pred=test_p,
            scaler=scaler,
            is_closed_loop=is_closed_loop,
            dt=dt,
            lyapunov_time_unit=ltu,
        )

    # New: Lambda Search BoxPlot
    residuals_hist = results.get("residuals_history")
    if residuals_hist:
        try:
             from reservoir.utils.plotting import plot_lambda_search_boxplot
             boxplot_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_lambda_boxplot.png"
             train_res = results.get("train") or {}
             lam_val = train_res.get("best_lambda")
             best_lam = float(lam_val) if isinstance(lam_val, (float, int)) else None
             plot_lambda_search_boxplot(
                 residuals_hist, boxplot_filename,
                 title=f"Lambda Search Residuals ({model_type_str})",
                 best_lambda=best_lam,
                 metric_name="NMSE",
             )
        except ImportError:
             pass


def _plot_quantum_section(results: ResultDict, topo_meta: ConfigDict, training_obj: TrainingConfig, dataset_name: str, model_type_str: str, readout: ReadoutModule | None, config: PipelineConfig, model_obj: ClosedLoopGenerativeModel | None) -> None:
    quantum_trace = results.get("quantum_trace")
    if quantum_trace is not None:
        try:
            from reservoir.utils.quantum_plotting import plot_qubit_dynamics

            filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
            dynamics_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_quantum_dynamics.png"

            trace_np = to_np_f64(quantum_trace) if quantum_trace is not None else None
            feature_names = None
            if model_obj is not None and hasattr(model_obj, "get_observable_names"):
                    feature_names = model_obj.get_observable_names()
            elif hasattr(training_obj, "get_observable_names"):
                    # Fallback but unlikely
                    feature_names = training_obj.get_observable_names()
            
            if trace_np is not None:
                plot_qubit_dynamics(trace_np, dynamics_filename, title=f"{model_type_str.upper()} Dynamics ({dataset_name})", feature_names=feature_names)

        except ImportError:
            pass # Skipping quantum plotting (ImportError)
        except (RuntimeError, ValueError) as e:
            print(f"Skipping quantum plotting (Error: {e})")


def plot_regression_report(
    *,
    train_y: NpF64 | None,
    val_y: NpF64 | None = None, # New Argument
    test_y: NpF64 | None,
    filename: str,
    model_type_str: str,
    mse: float | None = None,
    test_pred: NpF64 | None = None,
    scaler: Preprocessor | None = None,
    is_closed_loop: bool = False,
    dt: float | None = None,
    lyapunov_time_unit: float | None = None,
    vpt_threshold: float = 0.4,
) -> None:
    try:
        from reservoir.utils.plotting import plot_timeseries_comparison
    except ImportError as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    # Generate Test Predictions
    test_pred_final = test_pred

    # Infer global time offset
    # Offset = Length(Train) + Length(Val)
    offset = _get_seq_len(train_y) + _get_seq_len(val_y)

    # Align lengths if predictions are shorter (e.g. TimeDelayEmbedding)
    if test_y is not None and test_pred_final is not None:
        len_t = _get_seq_len(test_y)
        len_p = _get_seq_len(test_pred_final)
        
        if len_p < len_t:
             diff = len_t - len_p
             # print(f"  [Report] Aligning plot targets: slicing first {diff} steps.")
             test_y_np = test_y
             if test_y_np.ndim == 3:
                 test_y = test_y_np[:, diff:, :]
             else:
                 test_y = test_y_np[diff:]

    # Prepare for plotting (Inverse Transform to Raw Domain)
    if test_pred_final is not None:
        test_pred_plot = _to_2d(test_pred_final)
    else:
        test_pred_plot = None
        
    test_y_plot = _to_2d(test_y) if test_y is not None else None

    if scaler is not None:
        try:
            if test_pred_plot is not None:
                test_pred_plot = scaler.inverse_transform(test_pred_plot)
            if test_y_plot is not None:
                test_y_plot = scaler.inverse_transform(test_y_plot)
        except (ValueError, TypeError) as e:
            print(f"  [Report] Scaler inverse transform failed: {e}")

    # Update variables for plotting
    test_pred = test_pred_plot
    test_y = test_y_plot

    title_str = f"Test Predictions ({model_type_str})"
    if is_closed_loop:
        title_str = f"{title_str} closed-loop"
    
    # Calculate VPT using shared metric function
    vpt_lt = None
    if dt is not None and lyapunov_time_unit is not None and test_y is not None and test_pred is not None:
        try:
            from reservoir.utils.metrics import vpt_score
            # vpt_score handles multivariate logic correctly (Euclidean norm)
            # It expects (Time, Features), which we have as test_y and test_pred (already inverse transformed)
            vpt_steps = vpt_score(test_y, test_pred, threshold=vpt_threshold)
            
            steps_per_lt = int(lyapunov_time_unit / dt) if dt > 0 else 1
            vpt_lt = float(vpt_steps) / steps_per_lt if steps_per_lt > 0 else 0.0
        except ImportError:
            pass

    # Display VPT if calculated, otherwise fallback to MSE
    if vpt_lt is not None:
        title_str += f" | VPT: {vpt_lt:.2f} LT"
    elif mse is not None:
        title_str += f" | MSE: {mse:.4f} (Scaled)"

    if test_y is not None and test_pred is not None:
        plot_timeseries_comparison(
            targets=test_y,
            predictions=test_pred,
            filename=filename,
            title=title_str,
            time_offset=offset,
        )
