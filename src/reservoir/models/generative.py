"""
src/reservoir/models/generative.py
Base implementation for generative models providing closed-loop generation.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Callable

import jax
import jax.numpy as jnp




class ClosedLoopGenerativeModel(ABC):
    """
    Abstract base class for models that can generate autoregressive trajectories.
    Implements the GenerativeModel protocol methods for closed-loop generation.
    """

    @abstractmethod
    def initialize_state(self, batch_size: int = 1) -> jnp.ndarray:
        """Initialize hidden state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: jnp.ndarray, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single time step execution: (state, input) -> (next_state, output)."""
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, state: jnp.ndarray, input_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process sequence: (state, input_seq) -> (final_state, output_seq)."""
        raise NotImplementedError

    def generate_closed_loop(
        self,
        seed_data: jnp.ndarray,
        steps: int,
        readout: Optional[Any] = None,
        projection_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        verbose: bool = True
    ) -> jnp.ndarray:
        """
        Generate closed-loop predictions using Fast JAX Scan.
        
        Args:
            seed_data: 2D input (Time, Features) - will be converted to 3D internally
            steps: Number of steps to generate
            readout: Optional readout layer for prediction
            projection_fn: Optional projection function
            verbose: Print progress
            
        Returns:
            2D predictions (steps, Features)
        """
        # Convert 2D to 3D for internal processing
        # Cast to float32 to ensure scan carry consistency (even after x64 is enabled)
        history = jnp.asarray(seed_data)
        if history.ndim == 2:
            history = history[None, :, :]  # (T, F) -> (1, T, F)
        elif history.ndim == 1:
            history = history[None, None, :]
        
        batch_size = history.shape[0]
        
        if verbose:
            print(f"    [Generative] Generating {steps} steps (Fast JAX Scan)...")
        
        initial_state = self.initialize_state(batch_size)
        final_state, history_outputs = self.forward(initial_state, history)
        
        def predict_one(features):
            if readout is not None:
                # Readout expects features in correct shape
                # If features are (batch, feat), readout.predict usually handles it
                # Ensure input is suitable for readout
                f_in = features
                if f_in.ndim == 1: f_in = f_in[None, :]
                
                out = readout.predict(f_in)
                return out
            return features
        
        # Use the last output from history for the first prediction
        last_output = history_outputs[:, -1, :]  # (batch, features)
        first_prediction = predict_one(last_output)
        
        def scan_step(carry, _):
            h_prev, x_raw = carry
            x_proj = projection_fn(x_raw) if projection_fn else x_raw
            h_next, output = self.step(h_prev, x_proj)
            y_next = predict_one(output)
            return (h_next, y_next), y_next
        
        _, predictions = jax.lax.scan(scan_step, (final_state, first_prediction), None, length=steps)
        
        # predictions is (steps, batch, features) -> (batch, steps, features)
        predictions = jnp.swapaxes(predictions, 0, 1)
        
        # Return 2D (steps, features) for single batch
        if batch_size == 1:
            return predictions[0]
        return predictions
