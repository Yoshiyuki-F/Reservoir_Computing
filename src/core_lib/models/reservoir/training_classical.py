"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/reservoir/training_classical.py
Training and prediction helpers for the classical reservoir computer."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import jax.numpy as jnp


def _build_design_matrix(
    rc: Any,
    features: jnp.ndarray,
    *,
    fit: bool,
    washout: bool,
) -> jnp.ndarray:
    """Construct a design matrix from raw reservoir features."""
    data = features
    if washout and data.shape[0] > rc.washout_steps:
        data = data[rc.washout_steps :, ...]
    data = jnp.asarray(data, dtype=jnp.float64)
    if not rc.use_preprocessing:
        bias = jnp.ones((data.shape[0], 1), dtype=data.dtype)
        return jnp.concatenate([data, bias], axis=1)
    if fit:
        normalized = rc.scaler.fit_transform(data)
        design = rc.design_builder.fit_transform(normalized)
    else:
        normalized = rc.scaler.transform(data)
        design = rc.design_builder.transform(normalized)
    return design


def _train_readout(
    rc: Any,
    design_matrix: jnp.ndarray,
    target_data: jnp.ndarray,
    *,
    classification: bool,
    ridge_lambdas: Optional[Sequence[float]],
) -> None:
    """Fit the readout on the given design matrix and targets."""
    result = rc.readout.fit(
        design_matrix,
        jnp.asarray(target_data, dtype=jnp.float64),
        classification=classification,
        lambdas=ridge_lambdas or rc.ridge_lambdas,
        cv=rc._readout_cv,
        n_folds=rc._readout_n_folds,
        random_state=rc.initial_random_seed,
    )
    rc.W_out = jnp.asarray(result.weights, dtype=jnp.float64)
    rc.best_ridge_lambda = result.best_lambda
    rc.last_training_score = result.score_val
    rc.last_training_score_name = result.score_name
    rc.last_training_mse = (
        result.score_val if result.score_name.lower() == "mse" else None
    )
    rc.ridge_search_log = result.logs


def train_hidden_layer_regression(
    rc: Any,
    input_data: jnp.ndarray,
    target_data: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]] = None,
) -> None:
    """Train reservoir readout for regression using Ridge regression."""
    input_data = input_data.astype(jnp.float64)
    target_data = target_data.astype(jnp.float64)

    reservoir_states = rc.run_hidden_layer(input_data)
    design_matrix = _build_design_matrix(
        rc,
        reservoir_states,
        fit=True,
        washout=True,
    )

    target_aligned = target_data
    target_len = target_data.shape[0]
    design_len = design_matrix.shape[0]
    if target_len > design_len:
        start = target_len - design_len
        target_aligned = target_data[start:, ...]
    elif target_len < design_len:
        raise ValueError(
            f"Aligned target data shorter than design matrix "
            f"({target_len} vs {design_len})"
        )

    _train_readout(
        rc,
        design_matrix,
        target_aligned,
        classification=False,
        ridge_lambdas=ridge_lambdas,
    )


def predict_reservoir_regression(rc: Any, input_data: jnp.ndarray) -> jnp.ndarray:
    """Run regression prediction using a trained reservoir readout."""
    rc._ensure_trained()
    if rc.W_out is None:
        raise ValueError("Model has not been trained. Call train() first.")

    input_data = jnp.asarray(input_data, dtype=jnp.float64)
    rc._validate_input_data(input_data, rc.n_inputs)

    reservoir_states = rc.run_hidden_layer(input_data)
    design_matrix = _build_design_matrix(
        rc,
        reservoir_states,
        fit=False,
        washout=True,
    )
    predictions = design_matrix @ rc.W_out
    return jnp.asarray(predictions, dtype=jnp.float64)


def train_hidden_layer_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]] = None,
    num_classes: int = 10,
    return_features: bool = False,
) -> Optional[jnp.ndarray]:
    """Train reservoir readout for classification."""
    labels = labels.astype(jnp.int32)
    target_one_hot = jnp.zeros((labels.shape[0], num_classes), dtype=jnp.float64)
    target_one_hot = target_one_hot.at[jnp.arange(labels.shape[0]), labels].set(1.0)

    reservoir_states = rc.encode_batch(sequences)
    design_matrix = _build_design_matrix(
        rc,
        reservoir_states,
        fit=True,
        washout=False,
    )

    _train_readout(
        rc,
        design_matrix,
        target_one_hot,
        classification=True,
        ridge_lambdas=ridge_lambdas,
    )

    rc.classification_mode = True
    rc.num_classes = num_classes
    rc.trained = True

    if return_features:
        return reservoir_states
    return None


def predict_reservoir_classification(
    rc: Any,
    sequences: Optional[jnp.ndarray] = None,
    *,
    precomputed_features: Optional[jnp.ndarray] = None,
    progress_desc: Optional[str] = None,
) -> jnp.ndarray:
    """Predict classification logits using a trained reservoir."""
    del progress_desc  # placeholder for future logging
    if not rc.classification_mode or rc.num_classes is None:
        raise ValueError("Classification mode not enabled. Call train_classification first.")
    if rc.W_out is None:
        raise ValueError("Model has not been trained.")
    if precomputed_features is None and sequences is None:
        raise ValueError("Either sequences or precomputed_features must be provided.")
    if precomputed_features is not None and sequences is not None:
        raise ValueError("Specify only one of sequences or precomputed_features.")

    if precomputed_features is not None:
        features = jnp.asarray(precomputed_features, dtype=jnp.float64)
    else:
        features = rc.encode_batch(sequences)
    design_matrix = _build_design_matrix(
        rc,
        features,
        fit=False,
        washout=False,
    )
    logits = design_matrix @ rc.W_out
    return jnp.asarray(logits, dtype=jnp.float64)
