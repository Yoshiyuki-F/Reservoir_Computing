"""/home/yoshi/PycharmProjects/Reservoir/src/core_lib/models/flax_wrapper.py
Flax/JAX supervised wrapper implementing the BaseModel interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import serialization
import optax
import flax.linen as nn

from .base import BaseModel


@dataclass
class FlaxTrainingConfig:
    """Lightweight training hyperparameters for FlaxSupervisedModel."""

    learning_rate: float
    batch_size: int
    num_epochs: int
    classification: bool = False
    l2_weight_decay: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.l2_weight_decay < 0:
            raise ValueError("l2_weight_decay must be >= 0")


class FlaxSupervisedModel(BaseModel):
    """Wrap a Flax Module with BaseModel's train/predict/evaluate API."""

    def __init__(self, module: nn.Module, config: FlaxTrainingConfig):
        self.module = module
        self.config = config
        self.state: Optional[train_state.TrainState] = None
        self.trained: bool = False
        self._rng = jax.random.PRNGKey(config.seed)
        self.n_inputs: Optional[int] = None
        self.n_outputs: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API (BaseModel)                                             #
    # ------------------------------------------------------------------ #
    def train(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, Any]:
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched batch dimension: X {X.shape}, y {y.shape}")

        if self.state is None:
            self._initialize_state(X)

        train_step = self._build_train_step()
        metrics_accum: Dict[str, float] = {}
        batches = 0

        for _ in range(self.config.num_epochs):
            for batch_x, batch_y in self._batch_iterator(X, y):
                self.state, batch_metrics = train_step(self.state, batch_x, batch_y)
                for k, v in batch_metrics.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
                batches += 1

        averaged = {k: v / max(1, batches) for k, v in metrics_accum.items()}
        self.trained = True
        return averaged

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.state is None:
            raise RuntimeError("Model must be trained or initialized before predict()")
        X = jnp.asarray(X)
        return self.state.apply_fn({"params": self.state.params}, X)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        self._ensure_trained()
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        logits = self.predict(X)
        if self.config.classification:
            labels = self._to_one_hot(y, logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
            preds = jnp.argmax(logits, axis=-1)
            true_labels = jnp.argmax(labels, axis=-1)
            acc = jnp.mean(preds == true_labels)
            return {"loss": float(loss), "accuracy": float(acc)}

        mse = jnp.mean((logits - y) ** 2)
        mae = jnp.mean(jnp.abs(logits - y))
        return {"mse": float(mse), "mae": float(mae)}

    def save(self, path: Path | str) -> None:
        """Persist parameters to disk."""
        if self.state is None:
            raise RuntimeError("Model has no parameters to save (train first).")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bytes_data = serialization.to_bytes(self.state.params)
        path.write_bytes(bytes_data)

    def load(self, path: Path | str, sample_input: jnp.ndarray) -> None:
        """Load parameters from disk, initializing state if needed."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No parameter file at {path}")
        if self.state is None:
            self.state = self._create_state(jnp.asarray(sample_input))
        params_template = self.state.params  # type: ignore[union-attr]
        self.state = self.state.replace(
            params=serialization.from_bytes(params_template, path.read_bytes())  # type: ignore[union-attr]
        )
        self.trained = True

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _initialize_state(self, X: jnp.ndarray) -> None:
        """Initialize TrainState lazily with the first batch."""
        self.state = self._create_state(X)

    def _build_train_step(self):
        classification = self.config.classification
        apply_fn = self.state.apply_fn  # type: ignore[union-attr]

        def loss_and_metrics(params, batch_x, batch_y):
            logits = apply_fn({"params": params}, batch_x)
            if classification:
                labels = self._to_one_hot(batch_y, logits.shape[-1])
                loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
                preds = jnp.argmax(logits, axis=-1)
                true_labels = jnp.argmax(labels, axis=-1)
                acc = jnp.mean(preds == true_labels)
                return loss, {"loss": loss, "accuracy": acc}
            loss = jnp.mean((logits - batch_y) ** 2)
            mae = jnp.mean(jnp.abs(logits - batch_y))
            return loss, {"loss": loss, "mae": mae}

        def train_step(state: train_state.TrainState, batch_x, batch_y):
            (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
                state.params, batch_x, batch_y
            )
            new_state = state.apply_gradients(grads=grads)
            metrics_out = {k: jax.lax.stop_gradient(v) for k, v in metrics.items()}
            metrics_out["loss"] = jax.lax.stop_gradient(loss)
            return new_state, metrics_out

        return jax.jit(train_step)

    def _batch_iterator(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
        batch_size = self.config.batch_size
        n_samples = X.shape[0]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield X[start:end], y[start:end]

    @staticmethod
    def _to_one_hot(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
        """Accepts integer labels or pre-encoded one-hot targets."""
        if labels.ndim == 1:
            return jax.nn.one_hot(labels, num_classes=num_classes, dtype=jnp.float32)
        if labels.shape[-1] != num_classes:
            raise ValueError(
                f"Expected labels with last dim {num_classes}, got shape {labels.shape}"
            )
        return labels

    def _ensure_trained(self) -> None:
        if self.state is None or not self.trained:
            raise RuntimeError("Call train() before evaluate() or set trained params.")

    def _create_state(self, sample_input: jnp.ndarray) -> train_state.TrainState:
        """Initialize TrainState with a sample input batch."""
        self._rng, init_key = jax.random.split(self._rng)
        params = self.module.init(init_key, sample_input)["params"]
        tx = optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.l2_weight_decay,
        )
        state = train_state.TrainState.create(
            apply_fn=self.module.apply,
            params=params,
            tx=tx,
        )
        self.n_inputs = int(sample_input.shape[-1])
        sample_output = state.apply_fn({"params": params}, sample_input)
        if sample_output.ndim >= 1:
            self.n_outputs = int(sample_output.shape[-1])
        return state
