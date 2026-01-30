"""
Reporting utilities for post-run analysis: metrics, logging, and file outputs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import jax.numpy as jnp

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default

# --- Metrics Calculation ---

def compute_score(preds: Any, targets: Any, metric_name: str) -> float:
    """
    汎用スコア計算 (MSE または Accuracy)
    Runnerの _score メソッドを移動
    """
    preds_arr = jnp.asarray(preds)
    targets_arr = jnp.asarray(targets)

    if metric_name == "accuracy":
        pred_labels = preds_arr if preds_arr.ndim == 1 else jnp.argmax(preds_arr, axis=-1)
        true_labels = targets_arr if targets_arr.ndim == 1 else jnp.argmax(targets_arr, axis=-1)
        return float(jnp.mean(pred_labels == true_labels))

    # Regression (MSE)
    aligned_preds = preds_arr
    if preds_arr.shape != targets_arr.shape and preds_arr.size == targets_arr.size:
        aligned_preds = preds_arr.reshape(targets_arr.shape)

    return float(jnp.mean((aligned_preds - targets_arr) ** 2))

def calculate_chaos_metrics(
    y_true: Any, 
    y_pred: Any,
    dt: float,
    lyapunov_time_unit: float,
    vpt_threshold: float = 0.4,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Mackey-Glassなどのカオス予測専用の評価指標
    
    Args:
        y_true: Ground truth time series
        y_pred: Predicted time series
        dt: Simulation time step
        lyapunov_time_unit: 1 LT in time units (e.g., 1.1 for Lorenz 63)
        vpt_threshold: Threshold for VPT calculation (default 0.4 = sqrt(2)*0.3)
        verbose: If True, print metrics to console
    
    Returns:
        Dictionary with NDEI, var_ratio, correlation, VPT (steps), VPT (LT)
    """
    y_true_np = np.asarray(y_true).flatten()
    y_pred_np = np.asarray(y_pred).flatten()

    # Ensure same length
    min_len = min(len(y_true_np), len(y_pred_np))
    y_true_np = y_true_np[:min_len]
    y_pred_np = y_pred_np[:min_len]

    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    std_true = np.std(y_true_np)
    std_pred = np.std(y_pred_np)

    ndei = rmse / std_true if std_true > 1e-9 else float('inf')
    var_ratio = std_pred / std_true if std_true > 1e-9 else 0.0

    corr = 0.0
    if std_true > 1e-9 and std_pred > 1e-9:
        corr = np.corrcoef(y_true_np, y_pred_np)[0, 1]

    # --- VPT Calculation ---
    # Normalized error at each time step: |y_pred - y_true| / std(y_true)
    # VPT = first time step where error exceeds threshold
    if std_true > 1e-9:
        normalized_errors = np.abs(y_pred_np - y_true_np) / std_true
        # Find first index where error exceeds threshold
        exceed_indices = np.where(normalized_errors > vpt_threshold)[0]
        if len(exceed_indices) > 0:
            vpt_steps = int(exceed_indices[0])
        else:
            # Prediction never exceeds threshold
            vpt_steps = len(y_true_np)
    else:
        vpt_steps = 0
    
    # Convert VPT to Lyapunov time
    steps_per_lt = int(lyapunov_time_unit / dt) if dt > 0 else 1
    vpt_lt = vpt_steps / steps_per_lt if steps_per_lt > 0 else 0.0

    if verbose:
        print(f"=== Chaos Prediction Metrics ===")
        print(f"MSE       : {mse:.5f}")
        print(f"NDEI      : {ndei:.5f} (Target < 0.1)")
        print(f"Var Ratio : {var_ratio:.5f} (Target ~ 1.0)")
        print(f"Corr      : {corr:.5f} (Target > 0.95)")
        print(f"VPT       : {vpt_steps} steps ({vpt_lt:.2f} LT) @ threshold={vpt_threshold}")

    return {
        "mse": mse,
        "ndei": ndei,
        "var_ratio": var_ratio,
        "correlation": corr,
        "vpt_steps": vpt_steps,
        "vpt_lt": vpt_lt,
        "vpt_threshold": vpt_threshold,
    }

# --- Logging / Printing ---

import jax

def _print_feature_stats_impl(features: np.ndarray, stage: str, backend: str = "numpy") -> None:
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

def print_feature_stats(features: Any, stage: str) -> None:
    """
    特徴量の統計情報を表示する
    Runnerの _feature_stats メソッドを移動
    JITコンパイル時は jax.debug.callback を経由してホスト側で実行する。
    """
    if features is None:
        print(f"[FeatureStats:{stage}(skipped)] Closed-Loop mode: using raw data")
        return

    # Determine original backend before any conversion
    original_backend = type(features).__module__.split('.')[0]  # 'numpy' or 'jax'
    
    if isinstance(features, np.ndarray):
        _print_feature_stats_impl(features, stage, backend=original_backend)
    else:
        # JAX array or Tracer
        def _cb(f):
            # Convert to numpy for stats calculation, but report original backend
            _print_feature_stats_impl(np.asarray(f), stage, backend=original_backend)
        jax.debug.callback(_cb, features)

def print_ridge_search_results(train_res: Dict[str, Any], is_classification: bool) -> None:
    if not isinstance(train_res, dict):
        return
    history = train_res.get("search_history")
    if not history:
        return
    best_lam = train_res.get("best_lambda")
    weight_norms = train_res.get("weight_norms", {}) or {}
    
    # Determine Metric Label based on Task Type
    metric_label = "MSE" if is_classification else "VPT (Lyapunov Time)"

    # Decide best logic for marking
    # Both minimize score internally (MSE is min, -VPT is min)
    best_by_metric = min(history, key=history.get)

    best_marker = best_lam if best_lam is not None else best_by_metric

    print("\n" + "=" * 40)
    print(f"Ridge Hyperparameter Search ({metric_label})")
    print("-" * 40)
    sorted_lambdas = sorted(history.keys())
    for lam in sorted_lambdas:
        score = float(history[lam])
        
        # Format score for display
        score_disp = score
        label = "Val Score"
        if not is_classification:
            score_disp = -score  # flip back to positive VPT
            label = "Val VPT"
            
        norm = weight_norms.get(lam)
        norm_str = f"(Norm: {norm:.2e})" if norm is not None else "(Norm: n/a)"
        marker = " <= best" if (best_marker is not None and abs(float(lam) - float(best_marker)) < 1e-12) else ""
        print(f"   λ = {float(lam):.2e} : {label} = {score_disp:.4f} {norm_str}{marker}")
    print("=" * 40 + "\n")


def plot_distillation_loss(training_logs: Dict[str, Any], save_path: str, title: str, learning_rate: Optional[float] = None) -> None:
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
    plot_loss_history(loss_history, save_path, title=title, learning_rate=learning_rate)


def plot_classification_report(
    *,
    runner: Optional[Any] = None,
    readout: Any,
    train_X: Any,
    train_y: Any,
    test_X: Any,
    test_y: Any,
    val_X: Optional[Any],
    val_y: Optional[Any],
    filename: str,
    model_type_str: str,
    dataset_name: str,
    metric: str,
    results: Dict[str, Any],
    training_obj: Any,
    # 追加: 計算済みの予測値を受け取るオプション
    precalc_preds: Optional[Dict[str, Any]] = None,
    preprocessors: Optional[list[Any]] = None,
    selected_lambda: Optional[float] = None,
    lambda_norm: Optional[float] = None,
) -> None:
    try:
        from reservoir.utils.plotting import plot_classification_results
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    feature_batch_size = int(getattr(training_obj, "batch_size", 0) or 0)
    precalc_preds = precalc_preds or {}

    # ---------------------------------------------------------
    # 1. Labels Preparation
    # ---------------------------------------------------------
    train_labels_np = np.asarray(train_y)
    test_labels_np = np.asarray(test_y)
    if train_labels_np.ndim > 1:
        train_labels_np = np.argmax(train_labels_np, axis=-1)
    if test_labels_np.ndim > 1:
        test_labels_np = np.argmax(test_labels_np, axis=-1)

    # ---------------------------------------------------------
    # 2. Predictions (Train/Test) - Use cached if available
    # ---------------------------------------------------------
    train_pred_cached = precalc_preds.get("train_pred")
    test_pred_cached = precalc_preds.get("test_pred")

    # --- Train ---
    if train_pred_cached is not None:
        train_pred_np = np.asarray(train_pred_cached)
    else:
        # Fallback: 重い計算を実行
        print("  [Report] Calculating Train Predictions (Fallback)...")
        train_features_np = runner.batch_transform(train_X, batch_size=feature_batch_size)
        if readout is None:
            train_pred_np = np.asarray(train_features_np)
        else:
            train_pred_np = np.asarray(readout.predict(train_features_np))

    # --- Test ---
    if test_pred_cached is not None:
        test_pred_np = np.asarray(test_pred_cached)
    else:
        # Fallback
        print("  [Report] Calculating Test Predictions (Fallback)...")
        test_features_np = runner.batch_transform(test_X, batch_size=feature_batch_size)
        if readout is None:
            test_pred_np = np.asarray(test_features_np)
        else:
            test_pred_np = np.asarray(readout.predict(test_features_np))

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
        val_labels_np = np.asarray(val_y)
        if val_labels_np.ndim > 1:
            val_labels_np = np.argmax(val_labels_np, axis=-1)

        val_pred_cached = precalc_preds.get("val_pred")
        if val_pred_cached is not None:
            val_pred_np = np.asarray(val_pred_cached)
        else:
            # Fallback
            print("  [Report] Calculating Validation Predictions (Fallback)...")
            val_features_np = runner.batch_transform(val_X, batch_size=feature_batch_size)
            if readout is None:
                val_pred_np = np.asarray(val_features_np)
            else:
                val_pred_np = np.asarray(readout.predict(val_features_np))

        if val_pred_np.ndim > 1:
            if val_pred_np.shape[-1] > 1:
                val_pred_np = np.argmax(val_pred_np, axis=-1)
            val_pred_np = val_pred_np.ravel()

    # ---------------------------------------------------------
    # 4. Plot
    # ---------------------------------------------------------
    def _calc_acc(y_true, y_pred):
        if y_true is None or y_pred is None: return 0.0
        # Ensure 1D
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        return float(np.mean(y_t == y_p))

    acc_train = _calc_acc(train_labels_np, train_pred_np)
    acc_test = _calc_acc(test_labels_np, test_pred_np)
    acc_val = _calc_acc(val_labels_np, val_pred_np) if val_labels_np is not None else 0.0
    
    print(f"\n[Report] Accuracy Check (Pre-Plot):")
    print(f"  Train: {acc_train:.4%}")
    print(f"  Val  : {acc_val:.4%}")
    print(f"  Test : {acc_test:.4%}")

    metrics_payload = dict(results.get("test", {})) if isinstance(results, dict) else {}
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
    )


def _infer_filename_parts(topo_meta: Dict[str, Any], training_obj: Any, model_type_str: str, readout: Any = None, config: Any = None) -> list[str]:
    feature_shape = None
    student_layers = None
    readout_label = None
    preprocess_label = "raw"
    type_lower = str(model_type_str).lower()
    is_fnn = "fnn" in type_lower or "rnn" in type_lower or "nn" in type_lower
    if isinstance(topo_meta, dict):
        shapes = topo_meta.get("shapes") or {}
        feature_shape = shapes.get("feature")
        details = topo_meta.get("details") or {}
        student_layers = details.get("student_layers")
        readout_label = details.get("readout")
        if details.get("preprocess"):
            preprocess_label = details["preprocess"]

        topo_type = str(topo_meta.get("type", "")).lower()
        is_fnn = is_fnn or "fnn" in topo_type or "rnn" in topo_type or "nn" in topo_type

    filename_parts = [model_type_str, preprocess_label]

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
        proj_units = getattr(config.projection, 'n_units', 0)
        filename_parts.append(f"Proj{int(proj_units)}")

    # Readout type suffix
    if readout is not None:
        readout_type = type(readout).__name__
        # Include hidden layers info for FNN readout
        if hasattr(readout, 'hidden_layers') and readout.hidden_layers:
            layers_str = "-".join(str(int(v)) for v in readout.hidden_layers)
            lr = getattr(training_obj, 'learning_rate', None) if training_obj else None
            if lr is not None:
                filename_parts.append(f"{readout_type}{layers_str}_LR{lr:.0e}")
            else:
                filename_parts.append(f"{readout_type}{layers_str}")
        else:
            filename_parts.append(f"{readout_type}RO")

    # NN marker
    if is_fnn:
        layers = tuple(int(v) for v in student_layers) if student_layers is not None else ()
        if layers:
            filename_parts.append(f"nn{'-'.join(str(int(v)) for v in layers)}")
        else:
            filename_parts.append("nn0")
        filename_parts.append(f"epochs{int(getattr(training_obj, 'epochs', 0) or 0)}")
    return filename_parts


def generate_report(
    results: Dict[str, Any],
    config: Any,
    topo_meta: Dict[str, Any],
    *,
    runner: Optional[Any] = None,
    readout: Any,
    train_X: Any,
    train_y: Any,
    test_X: Any,
    test_y: Any,
    val_X: Optional[Any],
    val_y: Optional[Any],
    training_obj: Any,
    dataset_name: str,
    model_type_str: str,
    classification: bool = False,
    preprocessors: Optional[list[Any]] = None,
    dataset_preset: Optional[Any] = None,  # DatasetPreset for dt/lyapunov_time_unit
    model_obj: Optional[Any] = None, # New Argument
) -> None:
    # Loss plotting (distillation)
    training_logs = _safe_get(results, "training_logs", {})
    if training_logs:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
        lr = getattr(training_obj, 'learning_rate', None)
        plot_distillation_loss(training_logs, loss_filename, title=f"{model_type_str.upper()} Distillation Loss", learning_rate=lr)

    # Ridge search reporting
    train_res = _safe_get(results, "train", {})
    metric = "accuracy" if classification else "mse"
    # print_ridge_search_results(train_res, metric)

    # ---------------------------------------------------------
    # Retrieve Pre-calculated Predictions (Optimization)
    # ---------------------------------------------------------
    # 前のステップで計算された予測値があれば取得する
    precalc_preds = _safe_get(results, "outputs", {})

    # Classification plots
    if classification:
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        confusion_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
        selected_lambda = None
        lambda_norm = None
        if isinstance(train_res, dict):
            selected_lambda = train_res.get("best_lambda")
            weight_norms = train_res.get("weight_norms") or {}
            if selected_lambda is not None:
                lambda_norm = weight_norms.get(selected_lambda) or weight_norms.get(float(selected_lambda))

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
            metric=metric,
            selected_lambda=selected_lambda,
            lambda_norm=lambda_norm,
            results=results,
            training_obj=training_obj,
            precalc_preds=precalc_preds,  # <--- ここで渡す
            preprocessors=preprocessors,
        )
        
        # FNN Readout Loss Plot
        if readout is not None and hasattr(readout, 'training_logs') and readout.training_logs:
            fnn_loss_history = readout.training_logs.get("loss_history")
            if fnn_loss_history:
                loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
                lr = getattr(training_obj, 'learning_rate', None)
                plot_distillation_loss(readout.training_logs, loss_filename, title=f"{model_type_str.upper()} FNN Readout Loss", learning_rate=lr)
    elif metric == "mse":
        # Regression Plots
        filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        prediction_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_prediction.png"
        test_mse = _safe_get(results, "test", {}).get("mse")
        scaler = results.get("scaler")

        # Regressionの方も同様に最適化（必要な場合）
        # 現状は簡易的に予測値を渡すだけにしていますが、
        # plot_regression_report も同様に修正すれば高速化できます
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
             val_y=val_y, # Pass val_y for correct offset
             test_X=test_X,
             test_y=test_y,
             filename=prediction_filename,
             model_type_str=model_type_str,
             mse=test_mse,
             precalc_test_pred=test_pred_cached, 
             preprocessors=preprocessors,
             scaler=scaler,
             is_closed_loop=is_closed_loop,
             dt=dt,
             lyapunov_time_unit=ltu,
         )
         

    # Quantum Dynamics Plotting
    quantum_trace = _safe_get(results, "quantum_trace")
    if quantum_trace is not None:
        try:
            from reservoir.utils.quantum_plotting import plot_qubit_dynamics

            filename_parts = _infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
            dynamics_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_quantum_dynamics.png"

            # Convert to numpy and plot
            trace_np = np.asarray(quantum_trace)
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
    runner: Optional[Any] = None,
    readout: Any,
    train_y: Any,
    val_y: Optional[Any] = None, # New Argument
    test_X: Any,
    test_y: Any,
    filename: str,
    model_type_str: str,
    mse: Optional[float] = None,
    precalc_test_pred: Optional[Any] = None, 
    preprocessors: Optional[list[Any]] = None,
    scaler: Optional[Any] = None,
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
    if precalc_test_pred is not None:
        test_pred = np.asarray(precalc_test_pred)
    else:
        # Fallback: Use model directly if available
        if hasattr(runner, 'model') and runner.model is not None:
            test_features = runner.model(np.asarray(test_X))
            if readout is None:
                test_pred = test_features
            else:
                test_pred = readout.predict(test_features)
        else:
            print("  [Report] Cannot generate test predictions: no model or precalc data")
            return

    # Infer global time offset
    # Offset = Length(Train) + Length(Val)
    offset = 0
    
    def get_len(arr):
        if arr is None: return 0
        arr_np = np.asarray(arr)
        if arr_np.ndim == 3: return arr_np.shape[1]
        if arr_np.ndim == 2: return arr_np.shape[0]
        if arr_np.ndim == 1: return arr_np.shape[0]
        return 0

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

    test_pred_plot = to_2d(test_pred)
    test_y_plot = to_2d(test_y) if test_y is not None else None

    if scaler is not None:
        try:
            test_pred_plot = scaler.inverse_transform(test_pred_plot)
            if test_y_plot is not None:
                test_y_plot = scaler.inverse_transform(test_y_plot)
        except Exception as e:
            print(f"  [Report] Scaler inverse transform failed: {e}")
    elif preprocessors:
        for p in reversed(preprocessors):
            if hasattr(p, "inverse_transform"):
                try:
                    test_pred_plot = p.inverse_transform(test_pred_plot)
                    if test_y_plot is not None:
                        test_y_plot = p.inverse_transform(test_y_plot)
                except Exception as e:
                    print(f"  [Report] Inverse transform failed for {type(p).__name__}: {e}")

    # Update variables for plotting
    test_pred = test_pred_plot
    test_y = test_y_plot

    title_str = f"Test Predictions ({model_type_str})"
    if is_closed_loop:
        title_str = f"{title_str} closed-loop"
    
    # Calculate VPT if dt and lyapunov_time_unit are provided
    vpt_lt = None
    if dt is not None and lyapunov_time_unit is not None and test_y is not None and test_pred is not None:
        y_true_flat = test_y.flatten()
        y_pred_flat = test_pred.flatten()
        
        # Ensure same length
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # Avoid NaN
        valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
        if np.sum(valid_mask) > 0:
            y_t = y_true_flat[valid_mask]
            y_p = y_pred_flat[valid_mask]
            
            std_true = np.std(y_t)
            if std_true > 1e-9:
                # Normalized error at each step
                normalized_errors = np.abs(y_p - y_t) / std_true
                # Find first index where error exceeds threshold
                exceed_indices = np.where(normalized_errors > vpt_threshold)[0]
                if len(exceed_indices) > 0:
                    vpt_steps = int(exceed_indices[0])
                else:
                    vpt_steps = len(y_t)
                
                # Convert to LT
                steps_per_lt = int(lyapunov_time_unit / dt) if dt > 0 else 1
                vpt_lt = vpt_steps / steps_per_lt if steps_per_lt > 0 else 0.0

    # Display VPT if calculated, otherwise fallback to MSE
    if vpt_lt is not None:
        title_str += f" | VPT: {vpt_lt:.2f} LT"
    elif mse is not None:
        title_str += f" | MSE: {mse:.4f} (Scaled)"

    plot_timeseries_comparison(
        targets=test_y,
        predictions=test_pred,
        filename=filename,
        title=title_str,
        time_offset=offset,
    )
