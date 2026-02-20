"""
src/reservoir/models/generative.py
Base implementation for generative models providing closed-loop generation.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable

from beartype import beartype
import jax
import jax.numpy as jnp
from reservoir.core.types import JaxF64




@beartype
class ClosedLoopGenerativeModel(ABC):
    """
    Abstract base class for models that can generate autoregressive trajectories.
    Implements the GenerativeModel protocol methods for closed-loop generation.
    """

    @abstractmethod
    def initialize_state(self, batch_size: int = 1) -> JaxF64:
        """Initialize hidden state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: JaxF64, inputs: JaxF64) -> Tuple[JaxF64, JaxF64]:
        """Single time step execution: (state, input) -> (next_state, output)."""
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, state: JaxF64, input_data: JaxF64) -> Tuple[JaxF64, JaxF64]:
        """Process sequence: (state, input_seq) -> (final_state, output_seq)."""
        raise NotImplementedError

    def generate_closed_loop(
        self,
        seed_data: JaxF64,
        steps: int,
        readout: Optional[Callable] = None,
        projection_fn: Optional[Callable[[JaxF64], JaxF64]] = None,
        verbose: bool = True
    ) -> JaxF64:
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
        history = seed_data
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
        
        from reservoir.utils.reporting import print_feature_stats

        def scan_step(carry, _):
            h_prev, x_raw, step_idx = carry
            
            x_proj = projection_fn(x_raw) if projection_fn else x_raw
            h_next, output = self.step(h_prev, x_proj)
            y_next = predict_one(output)
            
            # Periodic Stats Logging (Every 50 steps)
            def log_stats(args):
                st_idx, x, h, y = args
                jax.debug.print("--- Step {i} ---", i=st_idx)
                jax.debug.print("Loop:Input | mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}", m=jnp.mean(x), s=jnp.std(x), mn=jnp.min(x), mx=jnp.max(x))
                jax.debug.print("Loop:State | mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}", m=jnp.mean(h), s=jnp.std(h), mn=jnp.min(h), mx=jnp.max(h))
                jax.debug.print("Loop:Pred  | mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}", m=jnp.mean(y), s=jnp.std(y), mn=jnp.min(y), mx=jnp.max(y))

            # Conditional Execution
            jax.lax.cond(
                step_idx % 800 == 0,
                log_stats,
                lambda _: None,
                (step_idx, x_raw, h_next, y_next)
            )
            
            return (h_next, y_next, step_idx + 1), y_next
        
        # Initial carry: (state, prediction, step_counter)
        init_carry = (final_state, first_prediction, 0)
        _, predictions = jax.lax.scan(scan_step, init_carry, None, length=steps)
        
        # predictions is (steps, batch, features) -> (batch, steps, features)
        predictions = jnp.swapaxes(predictions, 0, 1)
        
        # Return 2D (steps, features) for single batch
        if batch_size == 1:
            return predictions[0]
        return predictions
