"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/nn/fnn.py
FNN BaseModel wrapper using BaseFlaxModel.
Adapter is selected based on FNNConfig: Flatten (default) or TimeDelayEmbedding (if window_size is set).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from reservoir.models.config import FNNConfig
from reservoir.models.nn.base import BaseFlaxModel
from reservoir.models.generative import ClosedLoopGenerativeModel
from reservoir.training.presets import TrainingConfig
from reservoir.layers.adapters import Flatten, TimeDelayEmbedding


class FNNModel(BaseFlaxModel, ClosedLoopGenerativeModel):
    """
    FNN with configurable adapter (Step 4).
    
    If config.window_size is None: uses Flatten adapter (for classification)
    If config.window_size is set: uses TimeDelayEmbedding adapter (for time series regression)
    
    Also implements ClosedLoopGenerativeModel for autoregressive generation.
    """

    @property
    def input_window_size(self) -> int:
        return self.window_size or 0

    def __init__(self, model_config: FNNConfig, training_config: TrainingConfig, input_dim: int, output_dim: int, classification: bool = False):
        if not isinstance(model_config, FNNConfig):
            raise TypeError(f"FNNModel expects FNNConfig, got {type(model_config)}.")
        if int(input_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("input_dim and output_dim must be positive for FNNModel.")

        # Store window configuration
        self.window_size = model_config.window_size
        self._output_dim = output_dim
        
        # Select adapter based on config
        if self.window_size is not None:
            # TimeDelayEmbedding: input_dim becomes window_size * original_features
            self.adapter = TimeDelayEmbedding(window_size=self.window_size)
            effective_input_dim = self.window_size * int(input_dim)
        else:
            # Flatten: standard behavior
            self.adapter = Flatten()
            effective_input_dim = int(input_dim)

        hidden_layers = tuple(int(h) for h in (model_config.hidden_layers or ()))
        hidden_layers = tuple(h for h in hidden_layers if h > 0)

        self.layer_dims: Sequence[int] = (effective_input_dim, *hidden_layers, int(output_dim))

        super().__init__({"layer_dims": self.layer_dims}, training_config, classification=classification)

    def train(self, inputs: jnp.ndarray, targets: Optional[jnp.ndarray] = None, log_prefix: str = "4", **kwargs: Any) -> Dict[str, Any]:
        """Train with adapter-transformed inputs (and aligned targets if windowed)."""
        # Check if inputs are already adapted (Step 4 done externally)
        # Heuristic: if input feature dim matches the network's input layer dim
        if inputs.ndim == 2 and inputs.shape[-1] == self.layer_dims[0]:
            return super().train(inputs, targets, **kwargs)

        # Log Step 4 (Adapter) only during training
        adapter_name = self.adapter.__class__.__name__
        if self.window_size:
             adapter_name = f"TimeDelayEmbedding(k={self.window_size})"

        x_log_label = f"{log_prefix}:{adapter_name}:X:train"
        y_log_label = f"{log_prefix}:{adapter_name}:y:train"
        adapted_inputs = self.adapter(inputs, log_label=x_log_label)
        aligned_targets = self.adapter.align_targets(targets, log_label=y_log_label) if targets is not None else None

        return super().train(adapted_inputs, aligned_targets, **kwargs)

    def predict(self, X: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        """Predict with adapter-transformed inputs."""
        # Check if inputs are already adapted
        if X.ndim == 2 and X.shape[-1] == self.layer_dims[0]:
            return super().predict(X)
            
        adapted_inputs = self.adapter(X)
        return super().predict(adapted_inputs)

    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, float]:
        """Evaluate with adapter-transformed inputs (and aligned targets if windowed)."""
        # Check if inputs are already adapted
        if X.ndim == 2 and X.shape[-1] == self.layer_dims[0]:
            return super().evaluate(X, y)

        adapted_inputs = self.adapter(X)
        aligned_targets = self.adapter.align_targets(y)
        return super().evaluate(adapted_inputs, aligned_targets)

    def __call__(self, X: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        """Make model callable for batched_compute compatibility."""
        return self.predict(X)

    def _create_model_def(self) -> nn.Module:
        return FNN(layer_dims=self.layer_dims, return_hidden=False)

    # ------------------------------------------------------------------ #
    # ClosedLoopGenerativeModel implementation                           #
    # ------------------------------------------------------------------ #
    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        """
        For windowed FNN, state is the sliding window buffer of last `window_size` values.
        For non-windowed FNN, state is a dummy.
        """
        if self.window_size is not None:
            # State holds the sliding window buffer: (batch, window_size, features)
            return jnp.zeros((batch_size, self.window_size, self._output_dim))
        return jnp.zeros((batch_size, 1))

    def step(self, state: jnp.ndarray, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single step for closed-loop generation.
        For windowed FNN: concatenate window buffer to form input, predict, update buffer.
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        
        if self.window_size is not None:
            # state shape: (batch, window_size, features)
            # Flatten window to (batch, window_size * features) for FNN input
            batch_size = state.shape[0]
            windowed_input = state.reshape(batch_size, -1)  # (batch, window_size * features)
            
            # Predict using base class (no adapter, already windowed)
            output = super().predict(windowed_input)  # (batch, output_dim)
            
            # Update sliding window: shift left and append new prediction
            # new_state = [state[:, 1:, :], output[:, None, :]]
            new_state = jnp.concatenate([state[:, 1:, :], output[:, None, :]], axis=1)
            
            return new_state, output
        else:
            # Non-windowed: simple pass-through
            output = super().predict(inputs)
            return state, output

    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process seed sequence to initialize the sliding window state.
        For windowed FNN: fill the window buffer with the last `window_size` values from input.
        """
        # input_data shape: (batch, time, features)
        batch_size, seq_len, feat_dim = input_data.shape
        
        if self.window_size is not None:
            # Initialize state with last window_size values from seed
            # State shape: (batch, window_size, features)
            if seq_len >= self.window_size:
                initial_window = input_data[:, -self.window_size:, :]
            else:
                # Pad with zeros if seed is shorter than window
                padding = jnp.zeros((batch_size, self.window_size - seq_len, feat_dim))
                initial_window = jnp.concatenate([padding, input_data], axis=1)
            
            # Process seed through the windowed FNN to get outputs
            outputs = []
            current_state = initial_window
            for t in range(self.window_size - 1, seq_len):
                # Build window from input_data
                window = input_data[:, t - self.window_size + 1:t + 1, :]  # (batch, window_size, feat)
                windowed_input = window.reshape(batch_size, -1)
                output = super().predict(windowed_input)
                outputs.append(output)
            
            # Return final state (last window) and outputs
            output_seq = jnp.stack(outputs, axis=1) if outputs else jnp.zeros((batch_size, 0, self._output_dim))
            return initial_window, output_seq
        else:
            # Non-windowed: predict each timestep
            outputs = []
            for t in range(seq_len):
                step_input = input_data[:, t, :]
                output = super().predict(step_input)
                outputs.append(output)
            output_seq = jnp.stack(outputs, axis=1)
            return state, output_seq

    def generate_closed_loop(
        self,
        seed_data: jnp.ndarray,
        steps: int,
        readout: Any = None,
        projection_fn: Any = None,
        verbose: bool = True
    ) -> jnp.ndarray:
        """
        FNN-specific closed-loop generation using simple 2D arrays.
        Optimized with jax.lax.scan for speed.
        """
        import jax
        
        if self.window_size is None:
            # Non-windowed: just predict directly
            return self.predict(seed_data)
        
        # seed_data: (time, features) - use last window_size values as initial state
        seed = jnp.asarray(seed_data, dtype=jnp.float32)
        if seed.ndim == 3:
            seed = seed[0]  # Remove batch dim if present: (1, T, F) -> (T, F)
        
        W = self.window_size
        feat_dim = seed.shape[-1]
        
        # Initial window buffer: last W values from seed
        if seed.shape[0] >= W:
            window = seed[-W:, :]  # (W, features)
        else:
            padding = jnp.zeros((W - seed.shape[0], feat_dim))
            window = jnp.concatenate([padding, seed], axis=0)
        
        if verbose:
            print(f"    [FNN] Generating {steps} steps (JAX scan, window={W})...")
        
        def scan_step(window_state, _):
            # window_state: (W, features)
            # Flatten to (1, W * features) for FNN
            windowed_input = window_state.reshape(1, -1)
            output = super(FNNModel, self).predict(windowed_input)  # (1, output_dim)
            output = output.squeeze(0)  # (output_dim,)
            
            # Shift window: drop oldest, append new prediction
            new_window = jnp.concatenate([window_state[1:, :], output[None, :]], axis=0)
            return new_window, output
        
        _, predictions = jax.lax.scan(scan_step, window, None, length=steps)
        
        # predictions: (steps, output_dim)
        return predictions


class FNN(nn.Module):
    """Feed-forward network whose depth/width comes from layer_dims."""

    layer_dims: Sequence[int]
    return_hidden: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch, features), got shape {x.shape}")
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions")

        hidden_output = None
        # 入力次元(layer_dims[0])はスキップし、隠れ層以降の次元のみを使用する
        target_dims = self.layer_dims[1:]

        for idx, feat in enumerate(target_dims):
            # nn.Denseは入力次元を自動推論するため、features（出力次元）だけ指定すればOK
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