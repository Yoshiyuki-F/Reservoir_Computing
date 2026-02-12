"""
src/reservoir/utils/metrics.py
Refactored metrics calculation logic.
"""
from typing import Any, Dict
import jax.numpy as jnp
import numpy as np

def compute_score(preds: Any, targets: Any, metric_name: str) -> float:
    """
    Compute generic score (MSE, NMSE, or Accuracy).
    """
    preds_arr = jnp.asarray(preds)
    targets_arr = jnp.asarray(targets)

    if metric_name.lower() == "accuracy":
        pred_labels = preds_arr if preds_arr.ndim == 1 else jnp.argmax(preds_arr, axis=-1)
        true_labels = targets_arr if targets_arr.ndim == 1 else jnp.argmax(targets_arr, axis=-1)
        return float(jnp.mean(pred_labels == true_labels))

    # Regression
    aligned_preds = preds_arr
    if preds_arr.shape != targets_arr.shape and preds_arr.size == targets_arr.size:
        aligned_preds = preds_arr.reshape(targets_arr.shape)

    mse = float(jnp.mean((aligned_preds - targets_arr) ** 2))
    
    if metric_name.lower() == "nmse":
        # NMSE = MSE / Variance
        var_true = float(jnp.var(targets_arr))
        return mse / var_true if var_true > 1e-9 else float('inf')

    return mse

def calculate_chaos_metrics(
    y_true: Any, 
    y_pred: Any,
    dt: float,
    lyapunov_time_unit: float,
    vpt_threshold: float = 0.4,
) -> Dict[str, float]:
    """
    Calculate Chaos prediction metrics (Mackey-Glass, etc).
    Pure calculation, no printing.
    """
    if y_true is None or y_pred is None:
        return {}

    y_true_np = np.asarray(y_true).flatten()
    y_pred_np = np.asarray(y_pred).flatten()

    if y_true_np.size == 0 or y_pred_np.size == 0:
        return {}

    # Ensure same length
    min_len = min(len(y_true_np), len(y_pred_np))
    y_true_np = y_true_np[:min_len]
    y_pred_np = y_pred_np[:min_len]

    mse = np.mean((y_true_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    std_true = np.std(y_true_np)
    std_pred = np.std(y_pred_np)

    ndei = rmse / std_true if std_true > 1e-9 else float('inf')
    
    # NMSE (User def): sum((y - y_hat)^2) / sum(y^2)
    sum_sq_error = np.sum((y_true_np - y_pred_np) ** 2)
    sum_sq_true = np.sum(y_true_np ** 2)
    nmse = sum_sq_error / sum_sq_true if sum_sq_true > 1e-9 else float('inf')

    # NRMSE (User def): RMSE / sigma(y)
    nrmse = rmse / std_true if std_true > 1e-9 else float('inf')

    # MASE: MAE / Mean Absolute Diff of True (Naive Forecast)
    # denominator = (1/(T-1)) * sum(|y_t - y_{t-1}|)
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    if len(y_true_np) > 1:
        naive_mae = np.mean(np.abs(np.diff(y_true_np)))
        mase = mae / naive_mae if naive_mae > 1e-9 else float('inf')
    else:
        mase = float('inf')

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

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "nrmse": float(nrmse),
        "mase": float(mase),
        "ndei": float(ndei),
        "var_ratio": float(var_ratio),
        "correlation": float(corr),
        "vpt_steps": float(vpt_steps), # vpt_steps is int, but Dict is [str, float]. Maybe Optional[float] or convert to float. Int satisfies float in some contexts but type checker might complain if strict. Let's cast to float.
        "vpt_lt": float(vpt_lt),
        "vpt_threshold": float(vpt_threshold),
    }
