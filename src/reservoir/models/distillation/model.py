"""
Distill reservoir trajectories into a compact feed-forward network.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from reservoir.models.nn.base import BaseModel
from reservoir.models.nn.fnn import FNN
from reservoir.training.presets import TrainingConfig
from reservoir.models.distillation.config import DistillationConfig
from reservoir.models.reservoir.config import ReservoirConfig
from reservoir.models.reservoir.model import ReservoirModel


class DistillationModel(BaseModel):
    """Train a student FNN to mimic the state trajectory of a reservoir teacher."""

    def __init__(
        self,
        config: DistillationConfig,
        training_config: TrainingConfig,
        input_dim: int,
        teacher_model: ReservoirModel,
    ):
        config.validate(context="distillation")
        self.config = config
        self.training_config = training_config
        self.input_dim = int(input_dim)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")

        self.teacher_config: ReservoirConfig = config.teacher
        self.teacher_model = teacher_model
        self.teacher = teacher_model.reservoir
        if not hasattr(self.teacher, "projector"):
            raise AttributeError("Teacher reservoir must expose a 'projector' for distillation.")
        self.projector = self.teacher.projector
        self.teacher_output_dim: int = int(self.teacher.n_units)

        self.student_hidden: Tuple[int, ...] = tuple(int(v) for v in config.student_hidden_layers)
        self.student_input_dim: Optional[int] = None
        self.student: Optional[FNN] = None

        self.learning_rate: float = float(training_config.learning_rate)
        self.epochs: int = int(training_config.epochs)
        self.batch_size: int = int(training_config.batch_size)
        self._state: Optional[train_state.TrainState] = None
        self._rng = jax.random.PRNGKey(int(self.teacher_config.seed or 0) + 1)

    def _ensure_student(self, flat_dim: int) -> None:
        if self.student_input_dim == flat_dim and self.student is not None:
            return
        student_layers: Tuple[int, ...] = (
            flat_dim,
            *self.student_hidden,
            self.teacher_output_dim,
        )
        self.student = FNN(layer_dims=student_layers, return_hidden=False)
        self.student_input_dim = flat_dim
        self._state = None  # reset state because architecture changed

    def _init_train_state(self, key: jnp.ndarray, sample_input: jnp.ndarray) -> train_state.TrainState:
        if self.student is None:
            raise RuntimeError("Student network is not initialized.")
        variables = self.student.init(key, sample_input)
        params = variables["params"]
        tx = optax.adam(self.learning_rate)
        return train_state.TrainState.create(apply_fn=self.student.apply, params=params, tx=tx)

    @staticmethod
    @jax.jit
    def _train_step(
        state: train_state.TrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        def loss_fn(params):
            preds = state.apply_fn({"params": params}, batch_x)
            return jnp.mean((preds - batch_y) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def initialize(self, rng_key: jnp.ndarray, sample_input: jnp.ndarray) -> train_state.TrainState:
        sample = jnp.asarray(sample_input, dtype=jnp.float64)
        if sample.ndim != 2:
            raise ValueError(f"Student initializer expects 2D input, got {sample.shape}.")
        self._ensure_student(sample.shape[1])
        self._state = self._init_train_state(rng_key, sample)
        return self._state

    def _ensure_sequence(self, inputs: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(inputs)
        if arr.ndim == 2:
            if arr.shape[1] != self.input_dim:
                raise ValueError(
                    f"Input feature size mismatch: expected {self.input_dim}, received {arr.shape[1]}."
                )
            return arr[:, None, :]
        if arr.ndim != 3:
            raise ValueError(
                f"DistillationModel expects inputs with shape (batch, time, features), got {arr.shape}."
            )
        if arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input feature size mismatch: expected {self.input_dim}, received {arr.shape[-1]}."
            )
        return arr

    def _teacher_pass(self, inputs: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[0]
        init_state = self.teacher.initialize_state(batch_size)
        _, states = self.teacher.forward(init_state, inputs)
        teacher_states = states.states if hasattr(states, "states") else states
        return jnp.asarray(teacher_states, dtype=jnp.float64)

    def _aggregate_states(self, states: jnp.ndarray) -> jnp.ndarray:
        if states.ndim != 3:
            raise ValueError(f"Expected 3D states (batch, time, units), got {states.shape}.")
        mode = self.teacher_config.state_aggregation or "mean"
        if mode == "mean":
            return jnp.mean(states, axis=1)
        if mode == "last":
            return states[:, -1, :]
        if mode == "flatten":
            batch, time, feat = states.shape
            return states.reshape(batch, time * feat)
        raise ValueError(f"Unsupported aggregation mode '{mode}' for DistillationModel.")

    def train_student(self, inputs: jnp.ndarray) -> Dict[str, Any]:
        inputs_seq = self._ensure_sequence(jnp.asarray(inputs, dtype=jnp.float64))
        teacher_states = self._teacher_pass(inputs_seq)
        student_targets = self._aggregate_states(teacher_states)  # (B, N_res)

        batch_size = inputs_seq.shape[0]
        flat_dim = self.teacher_output_dim * inputs_seq.shape[1]
        self._ensure_student(flat_dim)

        num_samples = batch_size
        num_batches = num_samples // self.batch_size
        if num_batches == 0:
            raise ValueError(
                f"Dataset too small ({num_samples}) for batch_size={self.batch_size}."
            )

        usable = num_batches * self.batch_size
        batched_targets = student_targets[:usable].reshape(num_batches, self.batch_size, self.teacher_output_dim)
        batched_inputs_seq = inputs_seq[:usable].reshape(num_batches, self.batch_size, *inputs_seq.shape[1:])

        if self._state is None:
            self._rng, init_key = jax.random.split(self._rng)
            sample_proj = self.projector(batched_inputs_seq[0])
            sample_flat = sample_proj.reshape(sample_proj.shape[0], -1)
            self.initialize(init_key, sample_flat)

        @jax.jit
        def train_epoch(state, xs, ys):
            def body(carry, batch):
                seq_x, b_y = batch
                proj = self.projector(seq_x)
                flat = proj.reshape(proj.shape[0], -1)
                new_state, loss_val = DistillationModel._train_step(carry, flat, b_y)
                return new_state, loss_val

            new_state, losses = jax.lax.scan(body, state, (xs, ys))
            return new_state, jnp.mean(losses)

        loss_history = []
        for _ in range(self.epochs):
            self._state, epoch_loss = train_epoch(self._state, batched_inputs_seq, batched_targets)
            loss_history.append(float(epoch_loss))

        final_loss = loss_history[-1] if loss_history else None
        return {
            "loss_history": loss_history,
            "final_loss": final_loss,
            "final_mse": final_loss,
        }

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self._state is None:
            raise RuntimeError("Student network is not initialized. Call train_student or initialize first.")

        inputs_seq = self._ensure_sequence(jnp.asarray(inputs, dtype=jnp.float64))
        projected_inputs = self.projector(inputs_seq)
        flat_inputs = projected_inputs.reshape(projected_inputs.shape[0], -1)
        self._ensure_student(flat_inputs.shape[1])

        @jax.jit
        def _predict(params, x_batch):
            return self.student.apply({"params": params}, x_batch)

        preds = _predict(self._state.params, flat_inputs)
        return preds

    def predict(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.__call__(inputs)

    def train(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, **_: Any) -> Dict[str, Any]:
        return self.train_student(inputs)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        mse = self.score(X, y, metric="mse")
        return {"mse": mse}

    def score(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, metric: str = "mse") -> float:
        preds = self.predict(inputs)

        resolved_targets: Optional[jnp.ndarray] = None
        if targets is not None:
            candidate = jnp.asarray(targets)
            if candidate.shape == preds.shape:
                resolved_targets = candidate

        if resolved_targets is None:
            teacher_states = self._teacher_pass(self._ensure_sequence(jnp.asarray(inputs, dtype=jnp.float64)))
            teacher_targets = self._aggregate_states(teacher_states)
            if teacher_targets.shape != preds.shape:
                raise ValueError(
                    f"Unable to align targets for scoring: teacher produced {teacher_targets.shape}, predictions {preds.shape}."
                )
            resolved_targets = teacher_targets

        if metric != "mse":
            raise ValueError(f"Unsupported metric '{metric}' for DistillationModel.")
        return float(jnp.mean((preds - resolved_targets) ** 2))
