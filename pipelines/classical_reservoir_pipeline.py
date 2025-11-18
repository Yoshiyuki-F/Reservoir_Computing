"""Pipelines for training and using the classical ReservoirComputer.

This module contains the end-to-end workflows (training, feature extraction,
readout fitting) for the classical reservoir model. The ReservoirComputer
class itself is responsible only for weight initialization and state
propagation; all learning logic lives here.
"""

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
    """Construct a design matrix from raw reservoir features.

    This mirrors the former `_build_design_matrix` method on ReservoirComputer
    but is now implemented as a pipeline helper operating on the model
    instance.
    """
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
    """Fit the readout on the given design matrix and targets.

    Updates readout weights and logging fields on the ReservoirComputer
    instance.
    """
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


def train_hiddenLayer_regression(
    rc: Any,
    input_data: jnp.ndarray,
    target_data: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]] = None,
) -> None:
    """Train reservoir readout for regression using Ridge regression.

    Args:
        rc: ReservoirComputer instance.
        input_data: Input series of shape (time_steps, n_inputs).
        target_data: Target series of shape (time_steps, n_outputs).
        ridge_lambdas: Candidate ridge strengths, overrides config if given.
    """
    input_data = input_data.astype(jnp.float64)
    target_data = target_data.astype(jnp.float64)

    reservoir_states = rc.run_hiddenLayer(input_data)
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
    rc.classification_mode = False
    rc.num_classes = None
    rc.trained = True


def predict_reservoir_regression(
    rc: Any,
    input_data: jnp.ndarray,
) -> jnp.ndarray:
    """Run regression prediction using a trained reservoir readout."""
    rc._ensure_trained()

    if rc.W_out is None:
        raise ValueError("Model has not been trained. Call train() first.")

    input_data = input_data.astype(jnp.float64)
    reservoir_states = rc.run_hiddenLayer(input_data)
    design_matrix = _build_design_matrix(
        rc,
        reservoir_states,
        fit=False,
        washout=True,
    )
    predictions = rc.readout.predict(design_matrix)
    return jnp.asarray(predictions, dtype=jnp.float64)


def train_hiddenLayer_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]] = None,
    num_classes: int = 10,
    return_features: bool = False,
) -> Optional[jnp.ndarray]:
    """Train reservoir readout for classification."""
    features = rc._encode_sequences(
        sequences,
        desc="[TRAIN] Encoding sequences",
        leave=True,
    )
    design_matrix = _build_design_matrix(
        rc,
        features,
        fit=True,
        washout=False,
    )

    labels = labels.astype(jnp.int32)
    targets = jnp.zeros((labels.shape[0], num_classes), dtype=jnp.float64)
    targets = targets.at[jnp.arange(labels.shape[0]), labels].set(1.0)

    _train_readout(
        rc,
        design_matrix,
        targets,
        classification=True,
        ridge_lambdas=ridge_lambdas,
    )
    rc.classification_mode = True
    rc.num_classes = num_classes
    rc.trained = True

    if return_features:
        return features
    return None


def predict_reservoir_classification(
    rc: Any,
    sequences: Optional[jnp.ndarray] = None,
    *,
    precomputed_features: Optional[jnp.ndarray] = None,
    progress_desc: Optional[str] = None,
) -> jnp.ndarray:
    """Predict classification logits using a trained reservoir."""
    if not rc.classification_mode or rc.num_classes is None:
        raise ValueError(
            "Classification mode not enabled. "
            "Call train_hiddenLayer_classification first."
        )
    if rc.W_out is None:
        raise ValueError("Model has not been trained.")

    if precomputed_features is None and sequences is None:
        raise ValueError("Either sequences or precomputed_features must be provided.")
    if precomputed_features is not None and sequences is not None:
        raise ValueError("Specify only one of sequences or precomputed_features.")

    phase_desc = progress_desc or "Encoding eval sequences"
    desc_label = f"[PREDICT] {phase_desc}"
    if precomputed_features is not None:
        features = jnp.asarray(precomputed_features, dtype=jnp.float64)
    else:
        features = rc._encode_sequences(
            sequences,  # type: ignore[arg-type]
            desc=desc_label,
            leave=True,
        )
    design_matrix = _build_design_matrix(
        rc,
        features,
        fit=False,
        washout=False,
    )
    logits = rc.readout.predict(design_matrix)
    return jnp.asarray(logits, dtype=jnp.float64)

