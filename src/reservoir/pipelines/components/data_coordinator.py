
"""
src/reservoir/pipelines/components/data_coordinator.py

Implements DataCoordinator to handle data fetching, padding, and alignment,
freeing PipelineExecutor from low-level data manipulation.
"""
from beartype import beartype
from reservoir.core.types import NpF64


from reservoir.pipelines.config import DatasetMetadata, FrontendContext
from reservoir.utils.data_prep import apply_halo_padding
from reservoir.utils.reporting import print_feature_stats


@beartype
class DataCoordinator:
    """
    Coordinates data access and preparation for the pipeline.
    Handles Halo Padding and Target Alignment.
    """

    def __init__(self, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata):
        self.ctx = frontend_ctx
        self.meta = dataset_meta
        self.processed = frontend_ctx.processed_split

    def get_train_inputs(self) -> NpF64:
        """Get training inputs (no padding usually)."""
        return self.processed.train_X

    def get_val_inputs(self, window_size: int = 0) -> NpF64 | None:
        """Get validation inputs with Halo Padding applied."""
        if self.processed.val_X is None:
            return None
        
        # Apply Halo Padding using Train as context
        return apply_halo_padding(
            self.processed.val_X, 
            self.processed.train_X, 
            window_size
        )

    def get_test_inputs(self, window_size: int = 0) -> NpF64 | None:
        """Get test inputs with Halo Padding applied."""
        if self.processed.test_X is None:
            return None
            
        # Context is Val if exists, else Train
        context_source = self.processed.val_X if self.processed.val_X is not None else self.processed.train_X
        
        return apply_halo_padding(
            self.processed.test_X,
            context_source,
            window_size
        )

    def align_targets(
        self, 
        features: NpF64 | None, 
        split: str
    ) -> NpF64 | None:
        """
        Align targets for a specific split to match the given features.
        Trims targets from the start if they are longer than features (causal alignment).
        """
        if features is None:
            return None
            
        # Retrieve original targets based on split name
        targets = getattr(self.processed, f"{split}_y", None)
        if targets is None:
            return None

        len_f = features.shape[0]
        len_t = targets.shape[0]
        
        if len_t > len_f:
            diff = len_t - len_f
            aligned = targets[diff:]
            print_feature_stats(aligned, f"6.5:Aligned:y:{split} (Trimmed {diff})")
            return aligned
        elif len_t < len_f:
            # This is weird but we warn
            print(f"    [DataCoordinator] WARNING: Features ({len_f}) > Targets ({len_t}) for {split}. No alignment.")
            print_feature_stats(targets, f"6.5:Aligned:y:{split} (Mismatch)")
            return targets
        else:
            print_feature_stats(targets, f"6.5:Aligned:y:{split}")
            return targets
