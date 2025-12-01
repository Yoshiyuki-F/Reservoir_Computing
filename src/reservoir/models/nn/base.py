"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/base.py
Flax-based BaseModel adapter optimized with jax.lax.scan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class BaseModel(ABC):
    """Minimal training/evaluation contract shared by Flax adapters."""

    @abstractmethod
    def train(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, Any]:
        ...

    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        ...

    @abstractmethod
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        ...


class BaseFlaxModel(BaseModel, ABC):
    """Adapter that turns a flax.linen Module into a BaseModel."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.learning_rate: float = float(config.get("learning_rate"))
        self.epochs: int = int(config.get("num_epochs", config.get("epochs")))
        self.batch_size: int = int(config.get("batch_size"))
        self.classification: bool = bool(config.get("classification", False))
        self.seed: int = int(config.get("seed"))
        self._model_def = self._create_model_def()
        self._state: Optional[train_state.TrainState] = None
        self.trained: bool = False

    @abstractmethod
    def _create_model_def(self) -> Any:
        """Return the flax.linen Module definition."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Training utilities                                                 #
    # ------------------------------------------------------------------ #
    def _init_train_state(self, key: jnp.ndarray, sample_input: jnp.ndarray) -> train_state.TrainState:
        variables = self._model_def.init(key, sample_input)
        params = variables["params"]
        tx = optax.adam(self.learning_rate)
        return train_state.TrainState.create(
            apply_fn=self._model_def.apply,
            params=params,
            tx=tx,
        )

    @staticmethod
    def _train_step(state: train_state.TrainState, batch_x: jnp.ndarray, batch_y: jnp.ndarray, classification: bool):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch_x)
            if classification:
                labels = batch_y
                if labels.ndim == 1:
                    labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
                loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
            else:
                loss = jnp.mean((logits - batch_y) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    # ------------------------------------------------------------------ #
    # BaseModel API                                                      #
    # ------------------------------------------------------------------ #
    def train(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, Any]:
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        num_samples = X.shape[0]
        if num_samples != y.shape[0]:
            raise ValueError(f"Mismatched batch dimension: X {X.shape}, y {y.shape}")

        num_batches = num_samples // self.batch_size
        if num_batches == 0:
            raise ValueError(f"Dataset size {num_samples} is smaller than batch_size {self.batch_size}")

        X_pruned = X[: num_batches * self.batch_size]
        y_pruned = y[: num_batches * self.batch_size]
        X_batched = X_pruned.reshape((num_batches, self.batch_size) + X.shape[1:])
        y_batched = y_pruned.reshape((num_batches, self.batch_size) + y.shape[1:])

        rng = jax.random.PRNGKey(self.seed)
        init_key, _ = jax.random.split(rng)
        if self._state is None:
            self._state = self._init_train_state(init_key, X[:1])

        is_classification = self.classification

        @jax.jit
        def train_epoch(state, xs, ys):
            def body_fn(carry_state, batch_data):
                b_x, b_y = batch_data
                new_state, loss = BaseFlaxModel._train_step(carry_state, b_x, b_y, is_classification)
                return new_state, loss

            final_state, losses = jax.lax.scan(body_fn, state, (xs, ys))
            return final_state, jnp.mean(losses)

        loss_history = []
        for _ in range(self.epochs):
            self._state, epoch_loss = train_epoch(self._state, X_batched, y_batched)
            loss_history.append(float(epoch_loss))

        self.trained = True
        return {
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else None,
        }

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self._state is None:
            raise RuntimeError("Model not trained")
        X = jnp.asarray(X)
        @jax.jit
        def _predict_fn(params, inputs):
            return self._model_def.apply({"params": params}, inputs)

        return _predict_fn(self._state.params, X)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        y_arr = jnp.asarray(y)
        if self.classification:
            labels = y_arr
            if labels.ndim > 1:
                labels = jnp.argmax(labels, axis=-1)
            pred_labels = jnp.argmax(preds, axis=-1)
            acc = float(jnp.mean(pred_labels == labels))
            return {"accuracy": acc}
        mse = float(jnp.mean((preds - y_arr) ** 2))
        mae = float(jnp.mean(jnp.abs(preds - y_arr)))
        return {"mse": mse, "mae": mae}
