"""Pipelines for training and using the gate-based QuantumReservoirComputer.

This module contains the end-to-end workflows (training, feature extraction,
readout fitting) for the gate-based quantum reservoir model. The
QuantumReservoirComputer class itself is responsible only for quantum state
evolution and feature construction; all learning logic lives here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import jax.numpy as jnp


def train_quantum_reservoir_regression(
    rc: Any,
    input_data: jnp.ndarray,
    target_data: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]] = None,
) -> None:
    """Train gate-based quantum reservoir readout for regression.

    Mirrors QuantumReservoirComputer.train but is implemented as a pipeline
    helper that operates on the model instance.
    """
    rc._validate_input_data(input_data, rc.n_inputs)  # type: ignore[attr-defined]
    rc._validate_target_data(  # type: ignore[attr-defined]
        target_data,
        rc.n_outputs,
        input_data.shape[0],
    )

    input_data = jnp.array(input_data, dtype=jnp.float32)
    target_data = jnp.array(target_data, dtype=jnp.float32)

    quantum_states = rc._run_quantum_reservoir(input_data)  # type: ignore[attr-defined]
    design_matrix = rc._prepare_design_matrix(quantum_states, fit=True)  # type: ignore[attr-defined]
    rc.feature_dim_ = design_matrix.shape[1] - 1

    rc._fit_ridge_with_grid(design_matrix, target_data, ridge_lambdas)  # type: ignore[attr-defined]

    rc.classification_mode = False
    rc.num_classes = None
    rc.trained = True


def predict_quantum_reservoir_regression(
    rc: Any,
    input_data: jnp.ndarray,
) -> jnp.ndarray:
    """Run regression prediction using a trained quantum reservoir readout."""
    rc._ensure_trained()  # type: ignore[attr-defined]
    if rc.W_out is None:
        raise ValueError("Model has not been trained. Call train() first.")

    rc._validate_input_data(input_data, rc.n_inputs)  # type: ignore[attr-defined]
    input_data = jnp.array(input_data, dtype=jnp.float32)

    quantum_states = rc._run_quantum_reservoir(input_data)  # type: ignore[attr-defined]
    design_matrix = rc._prepare_design_matrix(quantum_states, fit=False)  # type: ignore[attr-defined]
    predictions = design_matrix @ rc.W_out
    return jnp.asarray(predictions, dtype=jnp.float64)


def train_quantum_reservoir_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]] = None,
    num_classes: int = 10,
    return_features: bool = False,
) -> Optional[jnp.ndarray]:
    """Train quantum reservoir readout for classification."""
    features = rc._encode_sequences(sequences)  # type: ignore[attr-defined]
    design_matrix = rc._prepare_design_matrix(features, fit=True)  # type: ignore[attr-defined]
    rc.feature_dim_ = design_matrix.shape[1] - 1

    labels = labels.astype(jnp.int32)
    targets = jnp.zeros((labels.shape[0], num_classes), dtype=jnp.float64)
    targets = targets.at[jnp.arange(labels.shape[0]), labels].set(1.0)

    rc._fit_ridge_with_grid(design_matrix, targets, ridge_lambdas)  # type: ignore[attr-defined]
    rc.classification_mode = True
    rc.num_classes = num_classes
    rc.trained = True

    if return_features:
        return features
    return None


def predict_quantum_reservoir_classification(
    rc: Any,
    sequences: Optional[jnp.ndarray] = None,
    *,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    precomputed_features: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Predict classification logits using a trained quantum reservoir."""
    if not rc.classification_mode or rc.num_classes is None:
        raise ValueError(
            "Classification mode not enabled. "
            "Call train_hidden_layer_classification first."
        )
    if rc.W_out is None:
        raise ValueError("Model has not been trained.")

    if precomputed_features is None and sequences is None:
        raise ValueError("Either sequences or precomputed_features must be provided.")
    if precomputed_features is not None and sequences is not None:
        raise ValueError("Specify only one of sequences or precomputed_features.")

    if precomputed_features is not None:
        features = jnp.asarray(precomputed_features, dtype=jnp.float64)
    else:
        _ = progress_desc, progress_position  # placeholders for future logging
        features = rc._encode_sequences(sequences)  # type: ignore[arg-type,attr-defined]
    design_matrix = rc._prepare_design_matrix(features, fit=False)  # type: ignore[attr-defined]
    logits = design_matrix @ rc.W_out
    return jnp.asarray(logits, dtype=jnp.float64)

