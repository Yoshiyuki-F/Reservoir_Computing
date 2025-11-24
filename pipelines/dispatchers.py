"""Dispatcher utilities for training and prediction pipelines."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import jax.numpy as jnp

from pipelines.classical_reservoir_pipeline import (
    predict_reservoir_classification,
    predict_reservoir_regression,
    train_hiddenLayer_classification,
    train_hiddenLayer_regression,
)
from pipelines.gatebased_quantum_pipeline import (
    predict_quantum_reservoir_classification,
    predict_quantum_reservoir_regression,
    train_quantum_reservoir_classification,
    train_quantum_reservoir_regression,
)



def get_model_factory(model_type: str):
    """Get the appropriate model factory based on model type."""
    if "reservoir" in model_type or "quantum" in model_type:
        from core_lib.models.reservoir import ReservoirComputerFactory

        return ReservoirComputerFactory
    if "ffn" in model_type:
        raise NotImplementedError("FFN models not yet implemented")
    raise ValueError(f"Unknown model type: {model_type}")


TrainFn = Callable[[Any, jnp.ndarray, jnp.ndarray, Optional[Sequence[float]]], None]
PredictFn = Callable[[Any, jnp.ndarray], jnp.ndarray]

ClassTrainFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, Optional[Sequence[float]], int, bool],
    Optional[jnp.ndarray],
]
ClassPredictFn = Callable[
    [Any, jnp.ndarray, Optional[jnp.ndarray], str],
    jnp.ndarray,
]


def _legacy_train_regression(
    rc: Any,
    input_data: jnp.ndarray,
    target_data: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
) -> None:
    """Legacy training path for models that still own their train() logic."""
    rc.train(input_data, target_data, ridge_lambdas=ridge_lambdas)


def _legacy_predict_regression(
    rc: Any,
    input_data: jnp.ndarray,
) -> jnp.ndarray:
    """Legacy prediction path for models that still own their predict() logic."""
    return rc.predict(input_data)


REGRESSION_PIPELINES: Dict[str, Tuple[TrainFn, PredictFn]] = {
    "classical": (train_hiddenLayer_regression, predict_reservoir_regression),
    "gatebased_quantum": (train_quantum_reservoir_regression, predict_quantum_reservoir_regression),
    "analog_quantum_legacy": (_legacy_train_regression, _legacy_predict_regression),
}


def _train_classical_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
    num_classes: int,
    return_features: bool,
) -> Optional[jnp.ndarray]:
    return train_hiddenLayer_classification(
        rc,
        sequences,
        labels,
        ridge_lambdas=ridge_lambdas,
        num_classes=num_classes,
        return_features=return_features,
    )


def _predict_classical_classification(
    rc: Any,
    sequences: jnp.ndarray,
    cached_features: Optional[jnp.ndarray],
    desc: str,
) -> jnp.ndarray:
    if cached_features is not None:
        return predict_reservoir_classification(
            rc,
            sequences=None,
            precomputed_features=cached_features,
            progress_desc=desc,
        )
    return predict_reservoir_classification(
        rc,
        sequences=sequences,
        precomputed_features=None,
        progress_desc=desc,
    )


def _train_quantum_classification(
    rc: Any,
    sequences: jnp.ndarray,
    labels: jnp.ndarray,
    ridge_lambdas: Optional[Sequence[float]],
    num_classes: int,
    return_features: bool,
) -> Optional[jnp.ndarray]:
    return train_quantum_reservoir_classification(
        rc,
        sequences,
        labels,
        ridge_lambdas=ridge_lambdas,
        num_classes=num_classes,
        return_features=return_features,
    )


def _predict_quantum_classification(
    rc: Any,
    sequences: jnp.ndarray,
    cached_features: Optional[jnp.ndarray],
    desc: str,
) -> jnp.ndarray:
    if cached_features is not None:
        return predict_quantum_reservoir_classification(
            rc,
            sequences=None,
            progress_desc=desc,
            progress_position=0,
            precomputed_features=cached_features,
        )
    return predict_quantum_reservoir_classification(
        rc,
        sequences=sequences,
        progress_desc=desc,
        progress_position=0,
        precomputed_features=None,
    )


CLASSIFICATION_PIPELINES: Dict[str, Tuple[ClassTrainFn, ClassPredictFn]] = {
    "classical": (_train_classical_classification, _predict_classical_classification),
    "gatebased_quantum": (_train_quantum_classification, _predict_quantum_classification)
}
