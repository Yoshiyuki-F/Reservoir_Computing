"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/metrics.py
Reservoir Computing用の評価メトリクス。
"""

from .jax_config import ensure_x64_enabled

ensure_x64_enabled()

import jax.numpy as jnp


def calculate_mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """平均二乗誤差（Mean Squared Error）を計算。
    
    予測値と目標値の差の二乗の平均を計算します。値が小さいほど
    予測精度が高いことを示します。外れ値の影響を受けやすい特徴があります。
    
    数式: MSE = (1/n) * Σ(y_pred - y_true)²
    
    Args:
        predictions: 予測値の配列。任意の形状
        targets: 目標値の配列。predictionsと同じ形状である必要がある
        
    Returns:
        計算されたMSE値（スカラー）
        
    Examples:
        >>> predictions = jnp.array([1.0, 2.0, 3.0])
        >>> targets = jnp.array([1.1, 1.9, 3.2])
        >>> mse = calculate_mse(predictions, targets)
        >>> print(f"MSE: {mse:.4f}")
        MSE: 0.0200
        
    Note:
        両方の配列は自動的にfloat64に変換されます。
    """
    predictions = predictions.astype(jnp.float64)
    targets = targets.astype(jnp.float64)
    return float(jnp.mean((predictions - targets) ** 2))


def mse_score(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Alias to calculate_mse for Runner integration."""
    return calculate_mse(predictions, targets)


def accuracy_score(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute classification accuracy for logits/probabilities or label indices."""
    preds = jnp.asarray(predictions)
    targs = jnp.asarray(targets)

    if preds.shape != targs.shape and preds.size == targs.size:
        preds = preds.reshape(targs.shape)

    if preds.ndim > 1:
        pred_labels = jnp.argmax(preds, axis=-1)
    else:
        pred_labels = preds
        if preds.dtype in (jnp.float16, jnp.float32, jnp.float64):
            pred_labels = preds > 0.5

    true_labels = targs
    if targs.ndim > 1:
        true_labels = jnp.argmax(targs, axis=-1)

    return float(jnp.mean(pred_labels == true_labels))


def calculate_mae(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """平均絶対誤差（Mean Absolute Error）を計算。
    
    予測値と目標値の差の絶対値の平均を計算します。MSEと比較して
    外れ値に対してロバストな特徴があり、解釈しやすい指標です。
    
    数式: MAE = (1/n) * Σ|y_pred - y_true|
    
    Args:
        predictions: 予測値の配列。任意の形状
        targets: 目標値の配列。predictionsと同じ形状である必要がある
        
    Returns:
        計算されたMAE値（スカラー）
        
    Examples:
        >>> predictions = jnp.array([1.0, 2.0, 3.0])
        >>> targets = jnp.array([1.1, 1.9, 3.2])
        >>> mae = calculate_mae(predictions, targets)
        >>> print(f"MAE: {mae:.4f}")
        MAE: 0.1333
        
    Note:
        両方の配列は自動的にfloat64に変換されます。
        MAEはMSEより外れ値の影響を受けにくく、解釈が容易です。
    """
    predictions = predictions.astype(jnp.float64)
    targets = targets.astype(jnp.float64)
    return float(jnp.mean(jnp.abs(predictions - targets)))
