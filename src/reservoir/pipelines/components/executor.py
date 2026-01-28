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

        # Step 6: Extract Features
        # For Closed-Loop Regression with Readout, skip Val/Test feature extraction (they're unused if readout handles CL)
        skip_val_test = not self.dataset_meta.classification and self.stack.readout is not None
        
        train_Z, val_Z, test_Z = self._extract_states(self.stack.model, skip_val_test)

        # Step 7: Fit Readout (Strategy Pattern)
        readout_name = type(self.stack.readout).__name__ if self.stack.readout else "None"
        print(f"\n=== Step 7: Readout ({readout_name}) with val data ===")
        
        strategy = self._select_strategy()

        fit_result = strategy.fit_and_evaluate(
            self.stack.model, self.stack.readout,
            train_Z, val_Z, test_Z,
            processed.train_y, processed.val_y, processed.test_y,
            self.frontend_ctx, self.dataset_meta,
            config
        )
        
        return {
            "fit_result": fit_result,
            "train_logs": train_logs,
        }

    def _extract_states(
        self,
        model: Any,
        skip_val_test: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """
        Extract internal states (Z) from the reservoir model for all splits.
        Handles batching and split existence checks.
        """
        processed = self.frontend_ctx.processed_split
        batch_size = self.dataset_meta.training.batch_size
        
        # Train
        model_train = partial(model, split_name="train")
        train_Z = batched_compute(model_train, processed.train_X, batch_size, desc="[Extracting] train")

        val_Z = None
        test_Z = None

        if not skip_val_test:
            if processed.val_X is not None:
                model_val = partial(model, split_name="val")
                val_Z = batched_compute(model_val, processed.val_X, batch_size, desc="[Extracting] val")
            
            # Test
            model_test = partial(model, split_name="test")
            test_Z = batched_compute(model_test, processed.test_X, batch_size, desc="[Extracting] test")
            
        return train_Z, val_Z, test_Z

    def _select_strategy(self):
        if self.stack.readout is None:
            return EndToEndStrategy(self.evaluator, self.stack.metric)
        elif self.dataset_meta.classification:
            return ClassificationStrategy(self.evaluator, self.stack.metric)
        else:
            return ClosedLoopRegressionStrategy(self.evaluator, self.stack.metric)
