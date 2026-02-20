"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/reporting.py
Reporting utilities for post-run analysis: metrics, logging, and file outputs Draw and save no recalculation.
"""
from __future__ import annotations

from typing import Optional, Sequence, List
import numpy as np
from reservoir.core.types import NpF64, ConfigDict, ResultDict, TrainLogs, EvalMetrics
from reservoir.layers.preprocessing import Preprocessor
from reservoir.core.interfaces import ReadoutModule, Component
from reservoir.training.config import TrainingConfig
from reservoir.data.config import DatasetPreset
from reservoir.models.presets import PipelineConfig
from reservoir.models.generative import ClosedLoopGenerativeModel

def _safe_get(d: ResultDict, key: str, default: Optional[ResultValue] = None) -> Optional[ResultValue]:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)

# --- Metrics Calculation ---



def print_chaos_metrics(metrics: EvalMetrics, header: Optional[str] = None) -> None:
    """
    Print chaos metrics to console.
    """
    if header:
        print(f"{header}")
    else:
        print(f"=== Chaos Prediction Metrics ===")
    
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

def print_feature_stats(features: Optional[NpF64], stage: str) -> None:
    """
    特徴量の統計情報を表示する (Host Domain).
    """
    if features is None:
        print(f"[FeatureStats:{stage}(skipped)] Closed-Loop mode: using raw data")
        return

    # In Host Domain, we expect NpF64
    _print_feature_stats_impl(features, stage, backend="numpy")

def print_ridge_search_results(train_res: ResultDict, metric_name: str = "MSE") -> None:
    if not isinstance(train_res, dict):
        return
    history = dict(train_res.get("search_history", {})) # type: ignore
    if not history:
        return
    best_lam = train_res.get("best_lambda")
    weight_norms = dict(train_res.get("weight_norms", {}) or {}) # type: ignore
    
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
        if lam is None: continue
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
        if best_marker is not None and isinstance(best_marker, (int, float, str)):
             try:
                 bm_val = float(best_marker)
                 if abs(lam_val - bm_val) < 1e-12:
                     marker = " <= best"
             except (ValueError, TypeError):
                 pass
        print(f"   λ = {lam_val:.2e} : {label} = {score_disp:.10f} {norm_str}{marker}")
    print("=" * 40 + "\n")


def plot_distillation_loss(training_logs: TrainLogs, save_path: str, title: str, learning_rate: Optional[float] = None) -> None:
    if not isinstance(training_logs, dict):
        return
    loss_history = training_logs.get("loss_history")
    if not loss_history:
        return
    try:
        from reservoir.utils.plotting import plot_loss_history
    except Exception as exc:  # pragma: no cover
        print(f"Skipping distillation loss plotting due to import error: {exc}")
        return
    loss_list = list(loss_history) if isinstance(loss_history, (list, tuple, np.ndarray)) else []
    plot_loss_history(loss_list, save_path, title=title, learning_rate=learning_rate)


def plot_classification_report(
    *,
    runner: Optional[Component] = None,
    readout: Optional[ReadoutModule],
    train_X: Optional[NpF64],
    train_y: Optional[NpF64],
    test_X: Optional[NpF64],
    test_y: Optional[NpF64],
    val_X: Optional[NpF64],
    val_y: Optional[NpF64],
    filename: str,
    model_type_str: str,
    dataset_name: str,
    results: ResultDict,
    training_obj: TrainingConfig,
    train_pred: Optional[NpF64] = None,
    test_pred: Optional[NpF64] = None,
    val_pred: Optional[NpF64] = None,
    selected_lambda: Optional[float] = None,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    try:
        from reservoir.utils.plotting import plot_classification_results
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    feature_batch_size = int(getattr(training_obj, "batch_size", 0) or 0)

    # ---------------------------------------------------------
    # 1. Labels Preparation
    # ---------------------------------------------------------
    train_labels_np = train_y if train_y is not None else np.array([])
    test_labels_np = test_y if test_y is not None else np.array([])
    if train_labels_np.ndim > 1:
        train_labels_np = np.argmax(train_labels_np, axis=-1)
    if test_labels_np.ndim > 1:
        test_labels_np = np.argmax(test_labels_np, axis=-1)

    # ---------------------------------------------------------
    # 2. Predictions (Train/Test) - Use passed values
    # ---------------------------------------------------------
    # --- Train ---
    if train_pred is not None:
        train_pred_np = train_pred
    else:
        # Fallback: 重い計算を実行
        print("  [Report] Calculating Train Predictions (Fallback)...")
        if runner is not None and hasattr(runner, "batch_transform"):
             # type ignore since runner is object but we duck type it
             train_features_np = runner.batch_transform(train_X, batch_size=feature_batch_size) # type: ignore
        else:
             train_features_np = np.array([])

        if readout is None:
            train_pred_np = train_features_np
        else:
            train_pred_np = to_np_f64(readout.predict(to_jax_f64(train_features_np)))

    # --- Test ---
    if test_pred is not None:
        test_pred_np = test_pred
    else:
        # Fallback
        print("  [Report] Calculating Test Predictions (Fallback)...")
        if runner is not None and hasattr(runner, "batch_transform"):
             test_features_np = runner.batch_transform(test_X, batch_size=feature_batch_size) # type: ignore
        else:
             test_features_np = np.array([])

        if readout is None:
            test_pred_np = test_features_np
        else:
            test_pred_np = to_np_f64(readout.predict(to_jax_f64(test_features_np)))

    # Argmax adjustment only if multi-class logits
    if train_pred_np.ndim > 1:
        if train_pred_np.shape[-1] > 1:
            train_pred_np = np.argmax(train_pred_np, axis=-1)
        # Flatten to ensure (N,) for comparison, handling (N, 1) argmax result or (N, 1) raw input
        train_pred_np = train_pred_np.ravel()
            
    if test_pred_np.ndim > 1:
        if test_pred_np.shape[-1] > 1:
            test_pred_np = np.argmax(test_pred_np, axis=-1)
        test_pred_np = test_pred_np.ravel()

    # ---------------------------------------------------------
    # 3. Validation
    # ---------------------------------------------------------
    val_labels_np = None
    val_pred_np = None
    if val_X is not None:
        val_labels_np = val_y
        if val_labels_np is not None and val_labels_np.ndim > 1:
            val_labels_np = np.argmax(val_labels_np, axis=-1)

        if val_pred is not None:
            val_pred_np = val_pred
        else:
            # Fallback
            print("  [Report] Calculating Validation Predictions (Fallback)...")
            if runner is not None and hasattr(runner, "batch_transform"):
                 val_features_np = runner.batch_transform(val_X, batch_size=feature_batch_size) # type: ignore
            else:
                 val_features_np = np.array([])

            if readout is None:
                val_pred_np = val_features_np
            else:
                val_pred_np = to_np_f64(readout.predict(to_jax_f64(val_features_np)))

        if val_pred_np.ndim > 1:
            if val_pred_np.shape[-1] > 1:
                val_pred_np = np.argmax(val_pred_np, axis=-1)
            val_pred_np = val_pred_np.ravel()

    # ---------------------------------------------------------
    # 4. Plot
    # ---------------------------------------------------------
    def _calc_acc(y_true: Optional[NpF64], y_pred: Optional[NpF64]) -> float:
        if y_true is None or y_pred is None: return 0.0
        # Ensure 1D
        y_t = y_true.ravel()
        y_p = y_pred.ravel()
        if len(y_t) == 0: return 0.0
        return float(np.mean(y_t == y_p))

    acc_train = None
    acc_test = None
    acc_val = None

    if results is not None:
         acc_train = float(_safe_get(dict(results.get("train", {})), "accuracy", 0.0)) # type: ignore
         acc_val = float(_safe_get(dict(results.get("validation", {})), "accuracy", 0.0)) # type: ignore
         acc_test = float(_safe_get(dict(results.get("test", {})), "accuracy", 0.0)) # type: ignore

    if acc_train == 0.0:
        acc_train = _calc_acc(train_labels_np, train_pred_np)
    
    if acc_test == 0.0:
        acc_test = _calc_acc(test_labels_np, test_pred_np)
    
    if acc_val == 0.0:
        acc_val = _calc_acc(val_labels_np, val_pred_np) if val_labels_np is not None else 0.0
    
    print(f"\n[Report] Accuracy Check (Pre-Plot):")
    print(f"  Train: {acc_train:.4%}")
    print(f"  Val  : {acc_val:.4%}")
    print(f"  Test : {acc_test:.4%}")

    # Extract lambda_norm from weight_norms for the selected lambda
    lambda_norm = None
    if selected_lambda is not None and isinstance(results, dict):
        train_res = results.get("train", {})
        if isinstance(train_res, dict):
            weight_norms = dict(train_res.get("weight_norms", {}) or {}) # type: ignore
            lambda_norm = float(weight_norms.get(selected_lambda, 0.0)) # type: ignore

    metrics_test = dict(results.get("test", {})) if isinstance(results, dict) else {} # type: ignore
    metrics_payload = {k: v for k, v in metrics_test.items()}
    plot_classification_results(
        train_labels=to_np_f64(train_labels_np),
        test_labels=to_np_f64(test_labels_np),
        train_predictions=to_np_f64(train_pred_np),
        test_predictions=to_np_f64(test_pred_np),
        val_labels=to_np_f64(val_labels_np),
        val_predictions=to_np_f64(val_pred_np),
        title=f"{model_type_str.upper()} on {dataset_name}",
        filename=filename,
        metrics_info=metrics_payload,
        best_lambda=selected_lambda,
        lambda_norm=lambda_norm,
        class_names=class_names,
    )


def _infer_filename_parts(topo_meta: ConfigDict, training_obj: TrainingConfig, model_type_str: str, readout: Optional[ReadoutModule] = None, config: Optional[PipelineConfig] = None) -> List[str]:
    student_layers = None
    preprocess_label = "raw"
    type_lower = str(model_type_str).lower()
    is_fnn = "fnn" in type_lower
    if isinstance(topo_meta, dict):
        details = dict(topo_meta.get("details") or {}) # type: ignore
        student_layers = details.get("student_layers")
        if details.get("preprocess"):
            raw_label = str(details["preprocess"])
            if raw_label == "CustomRangeScaler":
                centering = False
                scale = 0.0
                if config is not None and hasattr(config, "preprocess"):
                    # Duck typing since config.preprocess might be various types
                    scale = float(getattr(config.preprocess, "input_scale", 0.0))
                    centering = bool(getattr(config.preprocess, "centering", False))
                
                prefix = "T" if centering else "F"
                preprocess_label = f"{prefix}CRS{scale}"
            elif raw_label == "MinMaxScaler":
                f_min, f_max = 0.0, 1.0
                if config is not None and hasattr(config, "preprocess"):
                    f_min = float(getattr(config.preprocess, "feature_min", 0.0))
                    f_max = float(getattr(config.preprocess, "feature_max", 1.0))
                preprocess_label = f"Min{f_min:.2f}Max{f_max:.2f}"
            elif raw_label == "AffineScaler":
                input_scale = 1.0
                shift = 0.0
                if config is not None and hasattr(config, "preprocess"):
                    input_scale = float(getattr(config.preprocess, "input_scale", 1.0))
                    shift = float(getattr(config.preprocess, "shift", 0.0))
                preprocess_label = f"Affine_a{input_scale:.2f}_b{shift:.2f}"
            else:
                preprocess_label = raw_label

        topo_type = str(topo_meta.get("type", "")).lower()
        is_fnn = is_fnn or "fnn" in topo_type or "rnn" in topo_type or "nn" in topo_type

    # Append feedback_scale to model_type_str for quantum models
    if config is not None:
        model_cfg = getattr(config, 'model', None)
        if model_cfg:
            has_feedback = hasattr(model_cfg, 'feedback_scale') and model_cfg.feedback_scale is not None
            has_leak = hasattr(model_cfg, 'leak_rate') and model_cfg.leak_rate is not None
            if has_feedback:
                # Quantum model: use q{n_qubits}f{feedback_scale} format
                n_qubits = getattr(model_cfg, "n_qubits", None)
                if n_qubits is None and hasattr(config, "projection") and config.projection:
                    n_qubits = getattr(config.projection, "n_units", None)
                
                # If still None (and not in config), implies pure default or runtime inference.
                # However, for filenames, we prefer explicit values.
                # If we cannot find it, we might omit q{n} or fallback to something.
                # But typically one of them is set.
                q_str = f"q{n_qubits}" if n_qubits is not None else "q?"
                basis = str(getattr(model_cfg, "measurement_basis", "Z"))
                
                model_type_str = f"{model_type_str}_{q_str}_f{float(model_cfg.feedback_scale)}_{basis}"
            elif has_leak:
                # Classical reservoir: sr, lr, rc_connectivity
                sr = getattr(model_cfg, 'spectral_radius', None)
                lr = float(getattr(model_cfg, 'leak_rate', 1.0))
                rc_conn = getattr(model_cfg, 'rc_connectivity', None)
                tag = f"_sr{float(sr):.2f}" if sr is not None else ""
                tag += f"_lr{lr:.2f}"
                tag += f"_rc{float(rc_conn):.2f}" if rc_conn is not None else ""
                model_type_str = f"{model_type_str}{tag}"

    filename_parts: List[str] = [model_type_str, preprocess_label]

    # Window Size marker (for WindowsFNN/TDE)
    if config is not None:
        # Check direct model config (FNNConfig)
        model_cfg = getattr(config, 'model', None)
        if hasattr(model_cfg, 'window_size') and model_cfg.window_size is not None:
             filename_parts.append(f"k{int(model_cfg.window_size)}")
        # Check student config (Distillation)
        elif hasattr(model_cfg, 'student') and hasattr(model_cfg.student, 'window_size') and model_cfg.student.window_size is not None:
             filename_parts.append(f"k{int(model_cfg.student.window_size)}")

    # Projection marker (Proj) only if config.projection is defined
    if config is not None and hasattr(config, 'projection') and config.projection is not None:
        proj_config = config.projection
        proj_dict = proj_config.to_dict() if hasattr(proj_config, 'to_dict') else {}
        proj_type = str(proj_dict.get("type", "")).lower()
        proj_units = int(proj_dict.get("n_units", 0)) # type: ignore
        
        if proj_type == "random":
            input_scale = float(proj_dict.get("input_scale", 0.0)) # type: ignore
            input_conn = float(proj_dict.get("input_connectivity", 0.0)) # type: ignore
            bias_scale = float(proj_dict.get("bias_scale", 0.0)) # type: ignore
            filename_parts.append(
                f"RP{proj_units}_is{input_scale:.2f}_c{input_conn:.2f}_bs{bias_scale:.2f}"
            )
        elif proj_type == "center_crop":
            filename_parts.append(f"CCP{proj_units}")
        elif proj_type == "resize":
            filename_parts.append(f"Res{proj_units}")
        elif proj_type == "polynomial":
            # Get output_dim from projected_shape in topo_meta (already computed)
            shapes = dict(topo_meta.get("shapes", {})) if isinstance(topo_meta, dict) else {} # type: ignore
            projected_shape = shapes.get("projected")
            poly_output = int(projected_shape[-1]) if isinstance(projected_shape, (list, tuple)) else 0 # type: ignore
            filename_parts.append(f"Poly{poly_output}")
        elif proj_type == "pca":
            filename_parts.append(f"PCA{proj_units}")
        elif proj_type == "angle_embedding":
            freq = float(proj_dict.get("frequency", 0.0)) # type: ignore
            phase = float(proj_dict.get("phase_offset", 0.0)) # type: ignore
            filename_parts.append(f"AEP{proj_units}f{freq}p{phase}")
        elif proj_units:
            filename_parts.append(f"Proj{proj_units}")

    # Readout type suffix
    if readout is not None:
        readout_type = type(readout).__name__
        # Include hidden layers info for FNN readout
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
        layers = tuple(int(v) for v in layers_list)
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
    *,
    runner: Optional[Component] = None,
    readout: Optional[ReadoutModule],
    train_X: Optional[NpF64],
    train_y: Optional[NpF64],
    test_X: Optional[NpF64],
    test_y: Optional[NpF64],
    val_X: Optional[NpF64],
    val_y: Optional[NpF64],
    training_obj: TrainingConfig,
    dataset_name: str,
    model_type_str: str,
    classification: bool = False,
    # preprocessors removed
    dataset_preset: Optional[DatasetPreset] = None,  # DatasetPreset for dt/lyapunov_time_unit
    model_obj: Optional[ClosedLoopGenerativeModel] = None, # New Argument
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
            runner, train_X, train_y, test_X, test_y, val_X, val_y
        )
    elif metric == "mse":
        _plot_regression_section(
            results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
            runner, train_y, val_y, test_X, test_y, dataset_preset
        )

    # 3. Quantum Dynamics (if available)
    _plot_quantum_section(results, topo_meta, training_obj, dataset_name, model_type_str, readout, config, model_obj)


def _plot_distillation_section(results: ResultDict, topo_meta: ConfigDict, training_obj: TrainingConfig, model_type_str: str, readout: Optional[ReadoutModule], config: PipelineConfig, dataset_name: str) -> None:
    training_logs = dict(results.get("training_logs", {})) # type: ignore
    if training_logs:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
        lr = float(getattr(training_obj, 'learning_rate', 0.0))
        plot_distillation_loss(training_logs, loss_filename, title=f"{model_type_str.upper()} Distillation Loss", learning_rate=lr if lr > 0 else None) # type: ignore


def _plot_classification_section(
    results: ResultDict, config: PipelineConfig, topo_meta: ConfigDict, training_obj: TrainingConfig, dataset_name: str, model_type_str: str, readout: Optional[ReadoutModule],
    runner: Optional[Component], train_X: Optional[NpF64], train_y: Optional[NpF64], test_X: Optional[NpF64], test_y: Optional[NpF64], val_X: Optional[NpF64], val_y: Optional[NpF64]
) -> None:
    filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    confusion_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
    
    train_res = dict(results.get("train", {})) # type: ignore
    selected_lambda = None
    if isinstance(train_res, dict):
        selected_lambda = float(train_res.get("best_lambda", 0.0)) if train_res.get("best_lambda") is not None else None
    
    # Extract predictions from ResultDict and ensure Host Domain (NpF64)
    outputs = dict(results.get("outputs", {})) # type: ignore
    train_p = to_np_f64(outputs.get("train_pred")) if outputs.get("train_pred") is not None else None # type: ignore
    test_p = to_np_f64(outputs.get("test_pred")) if outputs.get("test_pred") is not None else None # type: ignore
    val_p = to_np_f64(outputs.get("val_pred")) if outputs.get("val_pred") is not None else None # type: ignore

    plot_classification_report(
        runner=runner,
        readout=readout,
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        val_X=val_X,
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
    results: ResultDict, config: PipelineConfig, topo_meta: ConfigDict, training_obj: TrainingConfig, dataset_name: str, model_type_str: str, readout: Optional[ReadoutModule],
    runner: Optional[Component], train_y: Optional[NpF64], val_y: Optional[NpF64], test_X: Optional[NpF64], test_y: Optional[NpF64], dataset_preset: Optional[DatasetPreset]
) -> None:
    filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    prediction_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_prediction.png"
    
    test_mse = float(_safe_get(dict(results.get("test", {})), "mse", 0.0)) # type: ignore
    scaler = results.get("scaler")
    is_closed_loop = bool(results.get("is_closed_loop", False))

    # Extract predictions from ResultDict and ensure Host Domain (NpF64)
    outputs = dict(results.get("outputs", {})) # type: ignore
    train_p = to_np_f64(outputs.get("train_pred")) if outputs.get("train_pred") is not None else None # type: ignore
    test_p = to_np_f64(outputs.get("test_pred")) if outputs.get("test_pred") is not None else None # type: ignore
    val_p = to_np_f64(outputs.get("val_pred")) if outputs.get("val_pred") is not None else None # type: ignore

    # Get dt and lyapunov_time_unit for VPT calculation
    dt = None
    ltu = None
    if dataset_preset is not None:
        ds_config = getattr(dataset_preset, 'config', None)
        if ds_config is not None:
            dt = float(getattr(ds_config, 'dt', 1.0))
            ltu = float(getattr(ds_config, 'lyapunov_time_unit', 1.0))

    plot_regression_report(
            runner=runner,
            readout=readout, # type: ignore
            train_y=train_y,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            filename=prediction_filename,
            model_type_str=model_type_str,
            mse=test_mse if test_mse > 0 else None,
            train_pred=train_p,
            test_pred=test_p,
            val_pred=val_p,
            scaler=scaler, # type: ignore
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
             train_res = dict(results.get("train", {})) # type: ignore
             best_lam = float(train_res.get("best_lambda", 0.0)) if train_res.get("best_lambda") is not None else None
             plot_lambda_search_boxplot(
                 residuals_hist, boxplot_filename, # type: ignore
                 title=f"Lambda Search Residuals ({model_type_str})",
                 best_lambda=best_lam,
                 metric_name="NMSE",
             )
        except ImportError:
             pass


def _plot_quantum_section(results: ResultDict, topo_meta: ConfigDict, training_obj: TrainingConfig, dataset_name: str, model_type_str: str, readout: Optional[ReadoutModule], config: PipelineConfig, model_obj: Optional[ClosedLoopGenerativeModel]) -> None:
    quantum_trace = results.get("quantum_trace")
    if quantum_trace is not None:
        try:
            from reservoir.utils.quantum_plotting import plot_qubit_dynamics

            filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
            dynamics_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_quantum_dynamics.png"

            # Convert to numpy and plot
            trace_np = to_np_f64(quantum_trace)
            feature_names = None
            if model_obj is not None and hasattr(model_obj, "get_observable_names"):
                    feature_names = model_obj.get_observable_names() # type: ignore
            elif hasattr(training_obj, "get_observable_names"):
                    # Fallback but unlikely
                    feature_names = training_obj.get_observable_names() # type: ignore
            
            plot_qubit_dynamics(trace_np, dynamics_filename, title=f"{model_type_str.upper()} Dynamics ({dataset_name})", feature_names=feature_names)

        except ImportError:
            pass # Skipping quantum plotting (ImportError)
        except Exception as e:
            print(f"Skipping quantum plotting (Error: {e})")


def plot_regression_report(
    *,
    runner: Optional[Component] = None,
    readout: ReadoutModule,
    train_y: Optional[NpF64],
    val_y: Optional[NpF64] = None, # New Argument
    test_X: Optional[NpF64],
    test_y: Optional[NpF64],
    filename: str,
    model_type_str: str,
    mse: Optional[float] = None,
    train_pred: Optional[NpF64] = None,
    test_pred: Optional[NpF64] = None,
    val_pred: Optional[NpF64] = None,
    # preprocessors removed
    scaler: Optional[Preprocessor] = None,
    is_closed_loop: bool = False,
    dt: Optional[float] = None,
    lyapunov_time_unit: Optional[float] = None,
    vpt_threshold: float = 0.4,
) -> None:
    try:
        from reservoir.utils.plotting import plot_timeseries_comparison
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    # Generate Test Predictions - Use explicit parameters
    if test_pred is not None:
        test_pred_final = test_pred
    else:
        # Fallback: Use model directly if available
        if runner is not None and hasattr(runner, 'model') and runner.model is not None: # type: ignore
            test_features = runner.model(to_jax_f64(test_X)) # type: ignore
            if readout is None:
                test_pred_final = to_np_f64(test_features)
            else:
                test_pred_final = to_np_f64(readout.predict(to_jax_f64(test_features)))
        else:
            print("  [Report] Cannot generate test predictions: no model or precalc data")
            return

    # Infer global time offset
    # Offset = Length(Train) + Length(Val)
    offset = 0
    
    def get_len(arr: Optional[NpF64]) -> int:
        if arr is None: return 0
        arr_np = arr
        if arr_np.ndim == 3: return int(arr_np.shape[1])
        if arr_np.ndim == 2: return int(arr_np.shape[0])
        if arr_np.ndim == 1: return int(arr_np.shape[0])
        return 0

    offset += get_len(train_y)
    offset += get_len(val_y)

    # Align lengths if predictions are shorter (e.g. TimeDelayEmbedding)
    if test_y is not None and test_pred_final is not None:
        len_t = get_len(test_y)
        len_p = get_len(test_pred_final)
        
        if len_p < len_t:
             diff = len_t - len_p
             # print(f"  [Report] Aligning plot targets: slicing first {diff} steps.")
             test_y_np = test_y
             if test_y_np.ndim == 3:
                 test_y = test_y_np[:, diff:, :]
             else:
                 test_y = test_y_np[diff:]

    # Prepare for plotting (Inverse Transform to Raw Domain)
    # Ensure (N, F) shape for scaler
    def to_2d(arr: NpF64) -> NpF64:
        if arr.ndim == 3: return arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 1: return arr.reshape(-1, 1)
        return arr

    test_pred_plot = to_2d(test_pred_final)
    test_y_plot = to_2d(test_y) if test_y is not None else None

    if scaler is not None:
        try:
            test_pred_plot = scaler.inverse_transform(test_pred_plot)
            if test_y_plot is not None:
                test_y_plot = scaler.inverse_transform(test_y_plot)
        except Exception as e:
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
    # 1. Common: Distillation Loss (if available)
    _plot_distillation_section(results, topo_meta, training_obj, model_type_str, readout, config, dataset_name)

    # 2. Main Task Plots (Classification vs Regression using MSE)
    metric = "accuracy" if classification else "mse"
    
    if classification:
        _plot_classification_section(
            results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
            runner, train_X, train_y, test_X, test_y, val_X, val_y
        )
    elif metric == "mse":
        _plot_regression_section(
            results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
            runner, train_y, val_y, test_X, test_y, dataset_preset
        )

    # 3. Quantum Dynamics (if available)
    _plot_quantum_section(results, topo_meta, training_obj, dataset_name, model_type_str, readout, config, model_obj)


def _plot_distillation_section(results, topo_meta, training_obj, model_type_str, readout, config, dataset_name):
    training_logs = _safe_get(results, "training_logs", {})
    if training_logs:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
        lr = getattr(training_obj, 'learning_rate', None)
        plot_distillation_loss(training_logs, loss_filename, title=f"{model_type_str.upper()} Distillation Loss", learning_rate=lr)


def _plot_classification_section(
    results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
    runner, train_X, train_y, test_X, test_y, val_X, val_y
):
    filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    confusion_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
    
    train_res = _safe_get(results, "train", {})
    selected_lambda = None
    if isinstance(train_res, dict):
        selected_lambda = train_res.get("best_lambda")
    precalc_preds = _safe_get(results, "outputs", {})

    plot_classification_report(
        runner=runner,
        readout=readout,
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        val_X=val_X,
        val_y=val_y,
        filename=confusion_filename,
        model_type_str=model_type_str,
        dataset_name=dataset_name,
        # metric removed
        selected_lambda=selected_lambda,
        results=results,
        training_obj=training_obj,
        precalc_preds=precalc_preds,
        # preprocessors removed
    )
    
    # FNN Readout Loss Plot
    if readout is not None and hasattr(readout, 'training_logs') and readout.training_logs:
        fnn_loss_history = readout.training_logs.get("loss_history")
        if fnn_loss_history:
            loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
            lr = getattr(training_obj, 'learning_rate', None)
            plot_distillation_loss(readout.training_logs, loss_filename, title=f"{model_type_str.upper()} FNN Readout Loss", learning_rate=lr)


def _plot_regression_section(
    results, config, topo_meta, training_obj, dataset_name, model_type_str, readout,
    runner, train_y, val_y, test_X, test_y, dataset_preset
):
    filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    prediction_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_prediction.png"
    
    test_mse = _safe_get(results, "test", {}).get("mse")
    scaler = results.get("scaler")
    precalc_preds = _safe_get(results, "outputs", {})
    test_pred_cached = precalc_preds.get("test_pred")
    is_closed_loop = results.get("is_closed_loop", False)

    # Get dt and lyapunov_time_unit for VPT calculation
    dt = None
    ltu = None
    if dataset_preset is not None:
        ds_config = getattr(dataset_preset, 'config', None)
        if ds_config is not None:
            dt = getattr(ds_config, 'dt', None)
            ltu = getattr(ds_config, 'lyapunov_time_unit', None)

    plot_regression_report(
            runner=runner,
            readout=readout,
            train_y=train_y,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            filename=prediction_filename,
            model_type_str=model_type_str,
            mse=test_mse,
            precalc_test_pred=test_pred_cached, 
            scaler=scaler,
            is_closed_loop=is_closed_loop,
            dt=dt,
            lyapunov_time_unit=ltu,
        )

    # New: Lambda Search BoxPlot
    residuals_hist = _safe_get(results, "residuals_history")
    if residuals_hist:
        try:
             from reservoir.utils.plotting import plot_lambda_search_boxplot
             boxplot_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_lambda_boxplot.png"
             train_res = _safe_get(results, "train", {})
             best_lam = train_res.get("best_lambda") if isinstance(train_res, dict) else None
             plot_lambda_search_boxplot(
                 residuals_hist, boxplot_filename,
                 title=f"Lambda Search Residuals ({model_type_str})",
                 best_lambda=best_lam,
                 metric_name="NMSE",
             )
        except ImportError:
             pass


def _plot_quantum_section(results, topo_meta, training_obj, dataset_name, model_type_str, readout, config, model_obj):
    quantum_trace = _safe_get(results, "quantum_trace")
    if quantum_trace is not None:
        try:
            from reservoir.utils.quantum_plotting import plot_qubit_dynamics

            filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
            dynamics_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_quantum_dynamics.png"

            # Convert to numpy and plot
            trace_np = to_np_f64(quantum_trace)
            feature_names = None
            if model_obj is not None and hasattr(model_obj, "get_observable_names"):
                    feature_names = model_obj.get_observable_names()
            elif hasattr(training_obj, "get_observable_names"):
                    # Fallback but unlikely
                    feature_names = training_obj.get_observable_names()
            
            plot_qubit_dynamics(trace_np, dynamics_filename, title=f"{model_type_str.upper()} Dynamics ({dataset_name})", feature_names=feature_names)

        except ImportError:
            pass # Skipping quantum plotting (ImportError)
        except Exception as e:
            print(f"Skipping quantum plotting (Error: {e})")


def plot_regression_report(
    *,
    runner: Optional[Component] = None,
    readout: ReadoutModule,
    train_y: Optional[NpF64],
    val_y: Optional[NpF64] = None, # New Argument
    test_X: Optional[NpF64],
    test_y: Optional[NpF64],
    filename: str,
    model_type_str: str,
    mse: Optional[float] = None,
    train_pred: Optional[NpF64] = None,
    test_pred: Optional[NpF64] = None,
    val_pred: Optional[NpF64] = None,
    # preprocessors removed
    scaler: Optional[Preprocessor] = None,
    is_closed_loop: bool = False,
    dt: Optional[float] = None,
    lyapunov_time_unit: Optional[float] = None,
    vpt_threshold: float = 0.4,
) -> None:
    try:
        from reservoir.utils.plotting import plot_timeseries_comparison
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    # Generate Test Predictions
    if test_pred is not None:
        test_pred_final = test_pred
    else:
        # Fallback: Use model directly if available
        if runner is not None and hasattr(runner, 'model') and runner.model is not None: # type: ignore
            test_features = runner.model(to_jax_f64(test_X)) # type: ignore
            if readout is None:
                test_pred_final = to_np_f64(test_features)
            else:
                test_pred_final = to_np_f64(readout.predict(to_jax_f64(test_features)))
        else:
            print("  [Report] Cannot generate test predictions: no model or precalc data")
            return

    # Infer global time offset
    # Offset = Length(Train) + Length(Val)
    offset = 0
    
    def get_len(arr: Optional[NpF64]) -> int:
        if arr is None: return 0
        arr_np = arr
        if arr_np.ndim == 3: return int(arr_np.shape[1])
        if arr_np.ndim == 2: return int(arr_np.shape[0])
        if arr_np.ndim == 1: return int(arr_np.shape[0])
        return 0

    offset += get_len(train_y)
    offset += get_len(val_y)

    # Align lengths if predictions are shorter (e.g. TimeDelayEmbedding)
    if test_y is not None and test_pred_final is not None:
        len_t = get_len(test_y)
        len_p = get_len(test_pred_final)
        
        if len_p < len_t:
             diff = len_t - len_p
             # print(f"  [Report] Aligning plot targets: slicing first {diff} steps.")
             test_y_np = test_y
             if test_y_np.ndim == 3:
                 test_y = test_y_np[:, diff:, :]
             else:
                 test_y = test_y_np[diff:]

    # Prepare for plotting (Inverse Transform to Raw Domain)
    # Ensure (N, F) shape for scaler
    def to_2d(arr: NpF64) -> NpF64:
        if arr.ndim == 3: return arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 1: return arr.reshape(-1, 1)
        return arr

    test_pred_plot = to_2d(test_pred_final)
    test_y_plot = to_2d(test_y) if test_y is not None else None

    offset += get_len(train_y)
    offset += get_len(val_y)

    # Align lengths if predictions are shorter (e.g. TimeDelayEmbedding)
    if test_y is not None and test_pred is not None:
        len_t = get_len(test_y)
        len_p = get_len(test_pred)
        
        if len_p < len_t:
             diff = len_t - len_p
             # print(f"  [Report] Aligning plot targets: slicing first {diff} steps.")
             if test_y.ndim == 3:
                 test_y = test_y[:, diff:, :]
             else:
                 test_y = test_y[diff:]

    # Prepare for plotting (Inverse Transform to Raw Domain)
    # Ensure (N, F) shape for scaler
    def to_2d(arr):
        if arr.ndim == 3: return arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 1: return arr.reshape(-1, 1)
        return arr

    test_pred_plot = to_2d(to_np_f64(test_pred))
    test_y_plot = to_2d(to_np_f64(test_y)) if test_y is not None else None

    if scaler is not None:
        try:
            test_pred_plot = scaler.inverse_transform(test_pred_plot)
            if test_y_plot is not None:
                test_y_plot = scaler.inverse_transform(test_y_plot)
        except Exception as e:
            print(f"  [Report] Scaler inverse transform failed: {e}")
            print(f"  [Report] Scaler inverse transform failed: {e}")
    # preprocessors block removed

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
