"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/base.py
Flax-based BaseModel adapter optimized with jax.lax.scan."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from reservoir.training.presets import TrainingConfig


class BaseModel(ABC):
    """Minimal training/evaluation contract shared by Flax adapters."""

    def train(self, inputs: Any, targets: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute internal pre-training phase (e.g., Distillation, Backprop). Returns metrics/logs.
        Defaults to a no-op so models without a pre-training stage can conform to the interface.
        """
        return {}

    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        ...

    @abstractmethod
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        ...

    def get_topology_meta(self) -> Dict[str, Any]:
        """Optional topology metadata for visualization."""
        return {}


class BaseFlaxModel(BaseModel, ABC):
    """Adapter that turns a flax.linen Module into a BaseModel."""

    def __init__(self, model_config: Dict[str, Any], training_config: TrainingConfig) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.learning_rate: float = float(training_config.learning_rate)
        self.epochs: int = int(training_config.epochs)
        self.batch_size: int = int(training_config.batch_size)
        self.classification: bool = bool(training_config.classification)
        self.seed: int = int(model_config.get("seed", training_config.seed))
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
    def train(self, inputs: Any, targets: Optional[Any] = None, **_: Any) -> Dict[str, Any]:
        """
        Train the model using a Python loop for batching to allow large datasets (OOM prevention).
        Input is expected to be numpy array or jax array.
        """
        if targets is None:
            raise ValueError("BaseFlaxModel.train requires 'targets' for supervised optimization.")
        
        # Check shapes (using numpy semantics if numpy)
        num_samples = inputs.shape[0]
        if num_samples != targets.shape[0]:
            raise ValueError(f"Mismatched batch dimension: inputs {inputs.shape}, targets {targets.shape}")
        
        if self.batch_size > num_samples:
             raise ValueError(f"Dataset size {num_samples} is smaller than batch_size {self.batch_size}")

        # Initialize State
        rng = jax.random.PRNGKey(self.seed)
        init_key, _ = jax.random.split(rng)
        
        # We need a sample input for init. Use first sample.
        sample_input = jnp.array(inputs[:1])
        if self._state is None:
            self._state = self._init_train_state(init_key, sample_input)

        is_classification = self.classification

        @jax.jit
        def train_step_jit(state, b_x, b_y):
            new_state, loss = BaseFlaxModel._train_step(state, b_x, b_y, is_classification)
            return new_state, loss

        loss_history = []
        
        # Training Loop
        for _ in range(self.epochs):
            batch_losses = []
            
            # Simple linear scan (no shuffle for now to match strict reproducibility of original if it didn't shuffle)
            # Original code did: X_batched = X_pruned.reshape((num_batches, batch_size) ... ) -> so it was sequential.
            
            # Drop last logic to match original's (num_batches * batch_size) behavior
            # or we can process all. Original was: X_pruned = X[: num_batches * self.batch_size]
            
            num_batches = num_samples // self.batch_size
            limit = num_batches * self.batch_size
            
            for i in range(0, limit, self.batch_size):
                end = i + self.batch_size
                # Slice on CPU (if numpy)
                b_x_np = inputs[i:end]
                b_y_np = targets[i:end]
                
                # Transfer to GPU
                b_x = jnp.array(b_x_np)
                b_y = jnp.array(b_y_np)
                
                self._state, loss = train_step_jit(self._state, b_x, b_y)
                batch_losses.append(float(loss))
            
            if batch_losses:
                import numpy as np  # Ensure numpy is available
                loss_history.append(float(np.mean(batch_losses)))

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

    def get_topology_meta(self) -> Dict[str, Any]:
        return getattr(self, "topology_meta", {})
