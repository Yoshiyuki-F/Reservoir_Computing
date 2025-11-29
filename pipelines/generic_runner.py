"""/home/yoshi/PycharmProjects/Reservoir/pipelines/generic_runner.py
Universal pipeline updated to support ReservoirModel orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp


class UniversalPipeline:
    """Minimal runner that can operate on BaseModel or ReservoirModel."""

    def __init__(
        self,
        model: Any,
        save_path: Optional[Path | str] = None,
        *,
        metric: str = "mse",
    ):
        self.model = model
        self.save_path = Path(save_path) if save_path is not None else None
        self.metric_name = metric

    def _has_attr(self, name: str) -> bool:
        return hasattr(self.model, name)

    def _train(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        if self._has_attr("train"):
            return self.model.train(X, y) or {}
        if self._has_attr("fit"):
            self.model.fit(X, y)
            return {}
        raise AttributeError("Model lacks both train and fit methods.")

    def _metric_score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        if hasattr(self.model, "score"):
            return float(self.model.score(X, y, metric=self.metric_name))
        if hasattr(self.model, "predict"):
            preds = jnp.asarray(self.model.predict(X))
            if self.metric_name == "accuracy":
                pred_labels = jnp.argmax(preds, axis=-1) if preds.ndim > 1 else (preds > 0.5)
                true_labels = y if y.ndim == 1 else jnp.argmax(y, axis=-1)
                return float(jnp.mean(pred_labels == true_labels))
            return float(jnp.mean((preds - y) ** 2))
        raise AttributeError("Model lacks predict/score methods for evaluation.")

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

        start = time.time()
        train_metrics = self._train(train_X, train_y)
        if not train_metrics:
            train_metrics = {self.metric_name: self._metric_score(train_X, train_y)}
        test_metrics = {self.metric_name: self._metric_score(test_X, test_y)}

        val_metrics: Dict[str, float] = {}
        if validation is not None:
            val_X, val_y = validation
            val_metrics = {self.metric_name: self._metric_score(jnp.asarray(val_X), jnp.asarray(val_y))}

        if self.save_path is not None and hasattr(self.model, "save"):
            self.model.save(self.save_path)

        elapsed = time.time() - start
        results: Dict[str, Dict[str, float]] = {
            "train": train_metrics,
            "test": test_metrics,
        }
        if val_metrics:
            results["validation"] = val_metrics
        results["meta"] = {"metric": self.metric_name, "elapsed_sec": elapsed}
        return results
