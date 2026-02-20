
"""
src/reservoir/utils/data_prep.py
Data preparation utilities for Reservoir Computing pipelines.
"""
from beartype import beartype
from reservoir.core.types import NpF64

import numpy as np

@beartype
def apply_halo_padding(
    current_data: NpF64,
    context_data: NpF64 | None,
    window_size: int
) -> NpF64:
    """
    Apply Halo Padding (Time Delay Overlap) to the start of current_data 
    using the end of context_data.
    
    Args:
        current_data: Data to pad (e.g. Val or Test split). Shape (Time, Features) or (Batch, Time, Features)
        context_data: Preceding data (e.g. Train for Val).
        window_size: Model's input window size (k). Overlap will be k-1.
        
    Returns:
        Padded data if overlap > 0 and context exists, else original data.
    """
    overlap = window_size - 1
    if overlap <= 0 or context_data is None:
        return current_data
        
    # Validation for dimensions
    if current_data.ndim != context_data.ndim:
        # Mismatch dims (e.g. one has batch dim, one doesn't) - skip safely
        return current_data
        
    # Extract context (last overlap steps)
    # Assumes data is (Time, Features) or (Batch, Time, Features)
    # If 3D, assumes dim 1 is time? Or dim 0 is batch?
    # Standard format in this project seems to be (Time, Feat) for simple arrays, 
    # but batched_compute handles both.
    # Halo Padding in Executor used: context = processed.train_X[-overlap:] (implying Time is Dim 0)
    
    # Check if length is sufficient
    if context_data.shape[0] < overlap:
        return current_data
        
    context = context_data[-overlap:]
    
    # Concatenate along axis 0 (Time)
    # Numpy array assumed
    return np.concatenate([context, current_data], axis=0)
