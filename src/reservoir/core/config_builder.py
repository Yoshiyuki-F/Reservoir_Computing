from __future__ import annotations

from typing import Any, Dict, Optional


def build_run_config(
    *,
    model_type: str,
    dataset: str,
    hidden_dim: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seq_len: Optional[int] = None,
    reservoir_preset: Optional[str] = None,
    training_preset: Optional[str] = "standard",
    use_design_matrix: Optional[bool] = None,
    poly_degree: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Assemble a run configuration, applying only explicit overrides.

    Anything left as None is omitted so that downstream preset resolution
    can supply defaults (e.g., TRAINING_PRESETS).
    """

    config: Dict[str, Any] = {
        "model_type": model_type.lower(),
        "dataset": dataset.lower(),
        "training_preset": training_preset,
        "training": {},
        "reservoir": {},
        "model": {},
    }

    if hidden_dim is not None:
        config["hidden_dim"] = hidden_dim

    if epochs is not None:
        config["training"]["num_epochs"] = epochs
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if learning_rate is not None:
        config["training"]["learning_rate"] = learning_rate

    if seq_len is not None:
        config["model"]["seq_len"] = seq_len

    if use_design_matrix is not None:
        config["use_design_matrix"] = use_design_matrix
    if poly_degree is not None:
        config["poly_degree"] = poly_degree

    if config.get("model_type") == "reservoir":
        config["reservoir_preset"] = reservoir_preset or "classical"

    config.update(kwargs)
    return config


__all__ = ["build_run_config"]
