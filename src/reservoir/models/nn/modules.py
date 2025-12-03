"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/modules.py
Pure flax.linen network definitions for FNN and SimpleRNN."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from jax import lax


class FNN(nn.Module):
    """Feed-forward network whose depth/width comes from layer_dims."""

    layer_dims: Sequence[int]
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions")

        hidden_output = None
        target_dims = self.layer_dims[1:]  # skip input dimension; flax infers input shape
        for idx, feat in enumerate(target_dims):
            x = nn.Dense(features=feat)(x)
            is_last = idx == len(target_dims) - 1
            if not is_last:
                x = nn.relu(x)
                hidden_output = x

        if hidden_output is None:
            hidden_output = x

        if self.return_hidden:
            return x, hidden_output
        return x


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

        batch_size, time_steps, input_dim = inputs.shape
        if time_steps == 0:
            raise ValueError("Input sequence must contain at least one timestep")

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
