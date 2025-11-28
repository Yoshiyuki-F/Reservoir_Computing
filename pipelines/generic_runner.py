"""/home/yoshi/PycharmProjects/Reservoir/pipelines/generic_runner.py
Universal pipeline that can run any BaseModel implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp

from core_lib.models import BaseModel


class UniversalPipeline:
    """Minimal runner that trains, evaluates, and optionally saves a model."""

    def __init__(self, model: BaseModel, save_path: Optional[Path | str] = None):
        self.model = model
        self.save_path = Path(save_path) if save_path is not None else None

    def run(
        self,
        train_X: Any,
        train_y: Any,
        test_X: Any,
        test_y: Any,
        *,
        validation: Optional[tuple[Any, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        train_X = jnp.asarray(train_X)
        train_y = jnp.asarray(train_y)
        test_X = jnp.asarray(test_X)
        test_y = jnp.asarray(test_y)

        train_metrics = self.model.train(train_X, train_y) or {}
        test_metrics = self.model.evaluate(test_X, test_y)

        val_metrics: Dict[str, float] = {}
        if validation is not None:
            val_X, val_y = validation
            val_metrics = self.model.evaluate(jnp.asarray(val_X), jnp.asarray(val_y))

        if self.save_path is not None and hasattr(self.model, "save"):
            self.model.save(self.save_path)

        results: Dict[str, Dict[str, float]] = {"train": train_metrics, "test": test_metrics}
        if val_metrics:
            results["validation"] = val_metrics
        return results
