"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/rnn.py
RNN BaseModel wrapper using BaseFlaxModel."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from reservoir.models.nn.base import BaseFlaxModel
from reservoir.training.presets import TrainingConfig


class RNNModel(BaseFlaxModel):
    """Wrap SimpleRNN module with BaseModel API."""

    def __init__(self, model_config: Dict[str, Any], training_config: TrainingConfig):
        required = ("input_dim", "hidden_dim", "output_dim")
        missing = [key for key in required if key not in model_config]
        if missing:
            raise ValueError(f"RNNModel requires keys {missing} in model_config.")
        self.input_dim: int = int(model_config["input_dim"])
        self.hidden_dim: int = int(model_config["hidden_dim"])
        self.output_dim: int = int(model_config["output_dim"])
        self.return_sequences: bool = bool(model_config.get("return_sequences", False))
        self.return_hidden: bool = bool(model_config.get("return_hidden", False))
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("RNNModel dimensions must be positive.")
        super().__init__(model_config, training_config)

    def _create_model_def(self) -> nn.Module:
        return SimpleRNN(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            return_sequences=self.return_sequences,
            return_hidden=self.return_hidden,
        )


class SimpleRNN(nn.Module):
    """Single-layer tanh RNN with Dense readout."""

    hidden_dim: int
    output_dim: int
    return_sequences: bool = False
    return_hidden: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        inputs = jnp.asarray(inputs, dtype=jnp.float64)
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch, time, features), got shape {inputs.shape}"
            )
        if self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive.")

        batch_size, time_steps, input_dim = inputs.shape
        if time_steps == 0:
            raise ValueError("Input sequence must contain at least one timestep")
        if input_dim <= 0:
            raise ValueError("Input feature dimension must be positive.")

        input_kernel = self.param(
            "input_kernel",
            nn.initializers.lecun_normal(),
            (input_dim, self.hidden_dim),
        ).astype(inputs.dtype)
        hidden_kernel = self.param(
            "hidden_kernel",
            nn.initializers.lecun_normal(),
            (self.hidden_dim, self.hidden_dim),
        ).astype(inputs.dtype)
        hidden_bias = self.param("hidden_bias", nn.initializers.zeros, (self.hidden_dim,)).astype(
            inputs.dtype
        )
        output_kernel = self.param(
            "output_kernel",
            nn.initializers.lecun_normal(),
            (self.hidden_dim, self.output_dim),
        ).astype(inputs.dtype)
        output_bias = self.param("output_bias", nn.initializers.zeros, (self.output_dim,)).astype(
            inputs.dtype
        )

        def step(carry: jnp.ndarray, x_t: jnp.ndarray):
            pre_activation = (
                jnp.dot(x_t, input_kernel) + jnp.dot(carry, hidden_kernel) + hidden_bias
            )
            new_state = jnp.tanh(pre_activation)
            output_t = jnp.dot(new_state, output_kernel) + output_bias
            return new_state, (new_state, output_t)

        init_hidden = jnp.zeros((batch_size, self.hidden_dim), dtype=inputs.dtype)
        time_major_inputs = jnp.swapaxes(inputs, 0, 1)
        _, (hidden_states, outputs) = lax.scan(step, init_hidden, time_major_inputs)

        hidden_states = jnp.swapaxes(hidden_states, 0, 1)
        outputs = jnp.swapaxes(outputs, 0, 1)

        output_result = outputs if self.return_sequences else outputs[:, -1, :]
        hidden_result = hidden_states if self.return_sequences else hidden_states[:, -1, :]

        if self.return_hidden:
            return output_result, hidden_result
        return output_result
