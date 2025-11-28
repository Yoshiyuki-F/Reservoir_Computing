"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/reservoir/training.py
Unified training/prediction helpers for reservoir models."""

from __future__ import annotations

from typing import Optional, Sequence, Literal, Dict, Any

import jax
import jax.numpy as jnp

from .base import BaseReservoir


def _prepare_design_matrix(
    rc: BaseReservoir,
    states: jnp.ndarray,
    *,
    is_training: bool,
    mode: Literal["regression", "classification"],
) -> jnp.ndarray:
    """Apply washout (for regression) and transform raw states into design features."""
    data = jnp.asarray(states, dtype=jnp.float64)
    if mode == "regression" and is_training and rc.washout_steps > 0 and data.shape[0] > rc.washout_steps:
        data = data[rc.washout_steps :, ...]
    return jnp.asarray(rc.transform_states(data, fit=is_training), dtype=jnp.float64)


def _fit_readout(
    rc: BaseReservoir,
    design_matrix: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    classification: bool,
    ridge_lambdas: Optional[Sequence[float]],
) -> Any:
    """Delegate fitting to the reservoir readout with sensible defaults."""
    kwargs: Dict[str, Any] = {}
    if hasattr(rc, "_readout_cv"):
        kwargs["cv"] = getattr(rc, "_readout_cv")
    if hasattr(rc, "_readout_n_folds"):
        kwargs["n_folds"] = getattr(rc, "_readout_n_folds")
    if hasattr(rc, "initial_random_seed"):
        kwargs["random_state"] = getattr(rc, "initial_random_seed")

    lambdas = ridge_lambdas
    if lambdas is None and hasattr(rc, "ridge_lambdas"):
        lambdas = getattr(rc, "ridge_lambdas")

    return rc.readout.fit(
        design_matrix,
        targets,
        classification=classification,
        lambdas=lambdas,
        **kwargs,
    )


def train_reservoir(
    rc: BaseReservoir,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    mode: Literal["regression", "classification"],
    ridge_lambdas: Optional[Sequence[float]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Unified training loop for classical/quantum reservoirs."""
    raw_states = rc.get_states(jnp.asarray(inputs))
    design_matrix = _prepare_design_matrix(rc, raw_states, is_training=True, mode=mode)

    if mode == "classification":
        labels = jnp.asarray(targets)
        if labels.ndim > 1:
            labels = jnp.argmax(labels, axis=-1)
        labels = labels.astype(jnp.int32)
        n_classes = int(num_classes) if num_classes is not None else int(jnp.max(labels)) + 1
        targets_enc = jax.nn.one_hot(labels, num_classes=n_classes, dtype=jnp.float64)
    else:
        targets_enc = jnp.asarray(targets, dtype=jnp.float64)
        # Align targets with design matrix after washout
        if targets_enc.shape[0] > design_matrix.shape[0]:
            diff = targets_enc.shape[0] - design_matrix.shape[0]
            targets_enc = targets_enc[diff:, ...]
        elif targets_enc.shape[0] != design_matrix.shape[0]:
            raise ValueError(
                f"Target length {targets_enc.shape[0]} does not match design matrix length {design_matrix.shape[0]}"
            )
        n_classes = None

    result = _fit_readout(
        rc,
        design_matrix,
        targets_enc,
        classification=(mode == "classification"),
        ridge_lambdas=ridge_lambdas,
    )

    rc.W_out = jnp.asarray(result.weights, dtype=jnp.float64)
    if hasattr(rc, "best_ridge_lambda"):
        rc.best_ridge_lambda = result.best_lambda
    if hasattr(rc, "ridge_search_log"):
        rc.ridge_search_log = result.logs
    rc.readout_logs = result.logs
    if hasattr(rc, "last_training_score"):
        rc.last_training_score = result.score_val
    if hasattr(rc, "last_training_score_name"):
        rc.last_training_score_name = result.score_name

    if mode == "classification":
        setattr(rc, "classification_mode", True)
        setattr(rc, "num_classes", n_classes)
    else:
        setattr(rc, "classification_mode", False)
        setattr(rc, "num_classes", None)

    if hasattr(rc, "trained"):
        rc.trained = True

    return {
        "trained": True,
        "classification_mode": getattr(rc, "classification_mode", False),
        "best_lambda": getattr(rc, "best_ridge_lambda", None),
        "score": getattr(rc, "last_training_score", None),
    }


def predict_reservoir(
    rc: BaseReservoir,
    inputs: jnp.ndarray,
    *,
    precomputed_features: Optional[jnp.ndarray] = None,
    mode: Optional[Literal["regression", "classification"]] = None,
) -> jnp.ndarray:
    """Unified prediction using a trained reservoir readout."""
    if rc.W_out is None:
        raise RuntimeError("Reservoir not trained.")

    effective_mode = mode
    if effective_mode is None:
        effective_mode = "classification" if getattr(rc, "classification_mode", False) else "regression"

    if precomputed_features is None:
        raw_states = rc.get_states(jnp.asarray(inputs))
        design_matrix = _prepare_design_matrix(rc, raw_states, is_training=False, mode=effective_mode)
    else:
        design_matrix = jnp.asarray(precomputed_features, dtype=jnp.float64)

    return jnp.asarray(design_matrix @ rc.W_out, dtype=jnp.float64)
