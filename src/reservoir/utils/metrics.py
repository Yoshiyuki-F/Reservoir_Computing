"""
src/reservoir/utils/metrics.py
Refactored metrics calculation logic.
"""
from typing import Any, Dict
import jax.numpy as jnp

# --- Standard Metric Functions ---

def mse(y_true: Any, y_pred: Any) -> float:
    """Mean Squared Error."""
    y_true_arr = jnp.asarray(y_true)
    y_pred_arr = jnp.asarray(y_pred)
    # Align if needed
    if y_pred_arr.shape != y_true_arr.shape and y_pred_arr.size == y_true_arr.size:
        y_pred_arr = y_pred_arr.reshape(y_true_arr.shape)
    return float(jnp.mean((y_true_arr - y_pred_arr) ** 2))

def rmse(y_true: Any, y_pred: Any) -> float:
    """Root Mean Squared Error."""
    return float(jnp.sqrt(mse(y_true, y_pred)))

def nmse(y_true: Any, y_pred: Any) -> float:
    """
    Normalized Mean Squared Error (User Definition).
    NMSE = sum((y - y_hat)^2) / sum(y^2)
    Normalization by Second Moment (Energy).
    """
    y_true_arr = jnp.asarray(y_true)
    y_pred_arr = jnp.asarray(y_pred)
    
    if y_pred_arr.shape != y_true_arr.shape and y_pred_arr.size == y_true_arr.size:
        y_pred_arr = y_pred_arr.reshape(y_true_arr.shape)

    numerator = jnp.sum((y_true_arr - y_pred_arr) ** 2)
    denominator = jnp.sum(y_true_arr ** 2)
    
    return float(numerator / denominator) if denominator > 1e-9 else float('inf')

def nrmse(y_true: Any, y_pred: Any) -> float:
    """
    Normalized Root Mean Squared Error.
    NRMSE = RMSE / std(y)
    """
    rmse_val = rmse(y_true, y_pred)
    std_true = float(jnp.std(jnp.asarray(y_true)))
    return float(rmse_val / std_true) if std_true > 1e-9 else float('inf')

def ndei(y_true: Any, y_pred: Any) -> float:
    """
    Non-Dimensional Error Index.
    Defined as RMSE / std(y). Same as NRMSE within this context.
    """
    return nrmse(y_true, y_pred)

def mase(y_true: Any, y_pred: Any) -> float:
    """
    Mean Absolute Scaled Error.
    MAE / Mean Absolute Error of Naive Forecast (on true data).
    """
    y_true_arr = jnp.asarray(y_true).flatten()
    y_pred_arr = jnp.asarray(y_pred).flatten()
    
    # Ensure same length
    min_len = min(len(y_true_arr), len(y_pred_arr))
    y_true_arr = y_true_arr[:min_len]
    y_pred_arr = y_pred_arr[:min_len]

    mae = jnp.mean(jnp.abs(y_true_arr - y_pred_arr))
    
    if len(y_true_arr) > 1:
        naive_mae = jnp.mean(jnp.abs(jnp.diff(y_true_arr)))
        return float(mae / naive_mae) if naive_mae > 1e-9 else float('inf')
    return float('inf')

def var_ratio(y_true: Any, y_pred: Any) -> float:
    """Variance Ratio: std(pred) / std(true)."""
    std_true = float(jnp.std(jnp.asarray(y_true)))
    std_pred = float(jnp.std(jnp.asarray(y_pred)))
    return std_pred / std_true if std_true > 1e-9 else 0.0

def correlation(y_true: Any, y_pred: Any) -> float:
    """Pearson Correlation Coefficient."""
    y_true_arr = jnp.asarray(y_true).flatten()
    y_pred_arr = jnp.asarray(y_pred).flatten()
    
    min_len = min(len(y_true_arr), len(y_pred_arr))
    y_true_arr = y_true_arr[:min_len]
    y_pred_arr = y_pred_arr[:min_len]

    std_true = float(jnp.std(y_true_arr))
    std_pred = float(jnp.std(y_pred_arr))
    
    if std_true > 1e-9 and std_pred > 1e-9:
        # jnp.corrcoef returns 2x2 matrix
        matrix = jnp.corrcoef(y_true_arr, y_pred_arr)
        return float(matrix[0, 1])
    return 0.0

def accuracy(y_true: Any, y_pred: Any) -> float:
    """Classification Accuracy."""
    y_true_arr = jnp.asarray(y_true)
    y_pred_arr = jnp.asarray(y_pred)
    
    pred_labels = y_pred_arr if y_pred_arr.ndim == 1 else jnp.argmax(y_pred_arr, axis=-1)
    true_labels = y_true_arr if y_true_arr.ndim == 1 else jnp.argmax(y_true_arr, axis=-1)
    
    return float(jnp.mean(pred_labels == true_labels))

def vpt_score(y_true: Any, y_pred: Any, threshold: float = 0.4) -> int:
    """
    Valid Prediction Time (VPT) - explicit step calculation.
    """
    y_true_arr = jnp.asarray(y_true).flatten()
    y_pred_arr = jnp.asarray(y_pred).flatten()
    
    min_len = min(len(y_true_arr), len(y_pred_arr))
    y_true_arr = y_true_arr[:min_len]
    y_pred_arr = y_pred_arr[:min_len]
    
    std_true = float(jnp.std(y_true_arr))
    
    if std_true > 1e-9:
        normalized_errors = jnp.abs(y_pred_arr - y_true_arr) / std_true
        # Find first index where error exceeds threshold
        # using argmax to find first true, checking if any true exists
        is_exceeded = normalized_errors > threshold
        if jnp.any(is_exceeded):
            # argmax on boolean returns index of first True
            return int(jnp.argmax(is_exceeded))
        else:
             return len(y_true_arr)
    return 0

# --- Dispatcher and Aggregator ---

def compute_score(preds: Any, targets: Any, metric_name: str) -> float:
    """
    Compute generic score (MSE, NMSE, or Accuracy).
    Dispatcher to specific functions.
    """
    name = metric_name.lower()
    
    if name == "accuracy":
        return accuracy(targets, preds)
    elif name == "nmse":
        return nmse(targets, preds)
    elif name == "mse":
        return mse(targets, preds)
    elif name == "rmse":
        return rmse(targets, preds)
    elif name == "nrmse":
        return nrmse(targets, preds)
    elif name == "mase":
        return mase(targets, preds)
    
    # Default to MSE if unknown (or raise error, but maintaining old behavior)
    return mse(targets, preds)

def calculate_chaos_metrics(
    y_true: Any, 
    y_pred: Any,
    dt: float,
    lyapunov_time_unit: float,
    vpt_threshold: float = 0.4,
) -> Dict[str, float]:
    """
    Calculate Chaos prediction metrics (Mackey-Glass, etc).
    Wrapper around detailed metric functions.
    """
    if y_true is None or y_pred is None:
        return {}
        
    # We use flattened arrays for chaos metrics usually
    # Helpers handle flattening internally if needed (MASE, Correlation, VPT)
    # MSE/NMSE handle logic internally using jnp
    
    # Calculate all metrics
    val_mse = mse(y_true, y_pred)
    val_nmse = nmse(y_true, y_pred)
    val_nrmse = nrmse(y_true, y_pred)
    val_mase = mase(y_true, y_pred)
    val_ndei = ndei(y_true, y_pred)
    val_var_ratio = var_ratio(y_true, y_pred)
    val_corr = correlation(y_true, y_pred)
    val_vpt_steps = vpt_score(y_true, y_pred, vpt_threshold)
    
    # Convert VPT to Lyapunov time
    steps_per_lt = int(lyapunov_time_unit / dt) if dt > 0 else 1
    val_vpt_lt = float(val_vpt_steps) / steps_per_lt if steps_per_lt > 0 else 0.0

    return {
        "mse": val_mse,
        "nmse": val_nmse,
        "nrmse": val_nrmse,
        "mase": val_mase,
        "ndei": val_ndei,
        "var_ratio": val_var_ratio,
        "correlation": val_corr,
        "vpt_steps": float(val_vpt_steps),
        "vpt_lt": val_vpt_lt,
        "vpt_threshold": float(vpt_threshold),
    }
