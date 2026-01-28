import time
from functools import partial
from typing import Dict, Any, Tuple, Optional

import jax.numpy as jnp

from reservoir.models.presets import PipelineConfig
from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
from reservoir.pipelines.evaluation import Evaluator
from reservoir.pipelines.strategies import (
    EndToEndStrategy,
    ClassificationStrategy,
    ClosedLoopRegressionStrategy
)
from reservoir.utils.batched_compute import batched_compute
from reservoir.utils.reporting import print_feature_stats


class PipelineExecutor:
    """
    Executes the pipeline steps:
    1. Model Training (Warmup)
    2. Feature Extraction
    3. Readout Fitting
    """

    def __init__(self, stack: ModelStack, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata):
        self.stack = stack
        self.frontend_ctx = frontend_ctx
        self.dataset_meta = dataset_meta
        self.evaluator = Evaluator()

    def run(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Run the execution phase.
        """
        processed = self.frontend_ctx.processed_split
        
        # Step 5: Model Dynamics (Training/Warmup)
        train_logs = self.stack.model.train(processed.train_X, processed.train_y) or {}

        print("\n=== Step 6: Extract Features ===")
        # Always extract val/test features to ensure correct target alignment (Step 6.5)
        # and to simplify logs, even if readout fitting uses Closed-Loop only.
        train_Z, val_Z, test_Z = self._extract_states(self.stack.model, skip_val_test=False)
        
        if train_Z is not None:
             print_feature_stats(train_Z, "6:Z:train")
        if val_Z is not None:
             print_feature_stats(val_Z, "6:Z:val")
        if test_Z is not None:
             print_feature_stats(test_Z, "6:Z:test")

        # Step 6.5: Target Alignment (Auto-Align y to Z)
        print("\n=== Step 6.5: Target Alignment (Auto-Align) ===")
        
        train_y = self._auto_align_target(train_Z, processed.train_y, "train")
        val_y = self._auto_align_target(val_Z, processed.val_y, "val")
        test_y = self._auto_align_target(test_Z, processed.test_y, "test")

        # Step 7: Fit Readout (Strategy Pattern)
        readout_name = type(self.stack.readout).__name__ if self.stack.readout else "None"
        print(f"\n=== Step 7: Readout ({readout_name}) with val data ===")
        
        strategy = self._select_strategy()

        fit_result = strategy.fit_and_evaluate(
            self.stack.model, self.stack.readout,
            train_Z, val_Z, test_Z,
            train_y, val_y, test_y,
            self.frontend_ctx, self.dataset_meta,
            config
        )
        
        return {
            "fit_result": fit_result,
            "train_logs": train_logs,
        }

    def _get_model_window_size(self, model: Any) -> int:
        """Helper to find adapter window size from model structure."""
        # Case 1: DistillationModel -> student -> adapter
        if hasattr(model, 'student') and hasattr(model.student, 'adapter'):
             return getattr(model.student.adapter, 'window_size', 0) or 0
        # Case 2: Model with direct adapter (e.g. FNNModel)
        if hasattr(model, 'adapter'):
             return getattr(model.adapter, 'window_size', 0) or 0
        return 0

    def _extract_states(self, model: Any, skip_val_test: bool = False) -> Tuple[Optional[jnp.ndarray], ...]:
        """
        Extract features (Z) from model using batched computation.
        Applies Halo Padding (Context Overlap) for time series validation/test to preserve length.
        """
        processed = self.frontend_ctx.processed_split
        batch_size = self.dataset_meta.training.batch_size
        
        # Determine Window Size for Haloing
        window_size = 0
        if not self.dataset_meta.classification:
             window_size = self._get_model_window_size(self.stack.model) # Access model from stack or arg (arg is stack.model)
        
        overlap = window_size - 1 if window_size > 1 else 0
        if overlap > 0:
            print(f"    [Executor] Applying Halo Padding (Overlap={overlap}) for Time Series Continuity...")

        # Train (Standard: No padding)
        model_train = partial(model, split_name="train")
        train_Z = batched_compute(model_train, processed.train_X, batch_size, desc="[Extracting] train")
        
        val_Z = None
        test_Z = None

        if not skip_val_test:
            if processed.val_X is not None:
                val_in = processed.val_X
                # Apply Halo (Context)
                if overlap > 0 and processed.train_X is not None:
                    # Check dims to ensure safe concat (assuming (Time, Feats))
                    if val_in.ndim == processed.train_X.ndim:
                         context = processed.train_X[-overlap:]
                         val_in = jnp.concatenate([context, val_in], axis=0)

                model_val = partial(model, split_name="val")
                val_Z = batched_compute(model_val, val_in, batch_size, desc="[Extracting] val")
            
            # Test
            if processed.test_X is not None:
                test_in = processed.test_X
                # Apply Halo (Context)
                if overlap > 0:
                    context_source = processed.val_X if processed.val_X is not None else processed.train_X
                    if context_source is not None and test_in.ndim == context_source.ndim:
                        context = context_source[-overlap:]
                        test_in = jnp.concatenate([context, test_in], axis=0)

                model_test = partial(model, split_name="test")
                test_Z = batched_compute(model_test, test_in, batch_size, desc="[Extracting] test")
            
        return train_Z, val_Z, test_Z

    def _select_strategy(self):
        if self.stack.readout is None:
            return EndToEndStrategy(self.evaluator, self.stack.metric)
        elif self.dataset_meta.classification:
            return ClassificationStrategy(self.evaluator, self.stack.metric)
        else:
            return ClosedLoopRegressionStrategy(self.evaluator, self.stack.metric)

    @staticmethod
    def _auto_align_target(Z: Optional[jnp.ndarray], y: Optional[jnp.ndarray], label: str) -> Optional[jnp.ndarray]:
        """
        Automatically trim target (y) to match feature (Z) length.
        Assumes causal relationship (e.g. windowing removes first W-1 steps), so trims from start.
        """
        if Z is None or y is None:
            return y if y is not None else None
        
        len_z = Z.shape[0]
        len_y = y.shape[0]
        
        if len_y > len_z:
            diff = len_y - len_z
            y_aligned = y[diff:]
            print_feature_stats(y_aligned, f"6.5:Aligned:y:{label} (Trimmed {diff})")
            return y_aligned
        elif len_y < len_z:
            print(f"    [Executor] WARNING: Z ({len_z}) > y ({len_y}) for {label}. No alignment performed.")
            print_feature_stats(y, f"6.5:Aligned:y:{label} (WARNING: Mismatch)")
            return y
        else:
            # Lengths match
            print_feature_stats(y, f"6.5:Aligned:y:{label}")
            return y

