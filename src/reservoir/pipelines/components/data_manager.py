import time
from dataclasses import replace
from typing import Optional, Tuple, Any

import numpy as np

from reservoir.core.identifiers import Dataset
from reservoir.data.loaders import load_dataset_with_validation_split
from reservoir.data.presets import DATASET_REGISTRY
from reservoir.data.structs import SplitDataset
from reservoir.layers.preprocessing import create_preprocessor, apply_layers
from reservoir.layers.projection import InputProjection
from reservoir.pipelines.config import FrontendContext, DatasetMetadata
from reservoir.models.presets import PipelineConfig
from reservoir.training.presets import get_training_preset, TrainingConfig
from reservoir.utils.batched_compute import batched_compute
from reservoir.utils.reporting import print_feature_stats


class PipelineDataManager:
    """
    Handles data loading, preprocessing, and projection.
    Encapsulates memory management and stats logging for data preparation.
    """

    def __init__(self, dataset: Dataset, config: PipelineConfig, training_override: Optional[TrainingConfig] = None):
        self.config = config
        self.dataset = dataset
        self.training_override = training_override
        self.metadata: Optional[DatasetMetadata] = None

    def prepare(self) -> FrontendContext:
        """
        Step 1, 2, 3: Load, Preprocess, Project.
        """
        # --- Step 1: Load ---
        self.metadata, raw_split = self._load_dataset()
        
        # --- Step 2 & 3: Frontend Processing ---
        frontend_ctx = self._process_frontend(raw_split)
        
        # Explicitly release raw_split (though extracted method scope handles it mostly)
        del raw_split
        
        return frontend_ctx

    def _load_dataset(self) -> Tuple[DatasetMetadata, SplitDataset]:
        """Step 1: Resolve presets and load dataset without mutating inputs later."""
        print("=== Step 1: Loading Dataset ===")
        dataset_enum, dataset_preset = self.dataset, DATASET_REGISTRY.get(self.dataset)

        training_cfg = self.training_override or get_training_preset("standard")
        dataset_split = load_dataset_with_validation_split(
            self.dataset,
            training_cfg,
            require_3d=True,
        )

        self._log_dataset_stats(dataset_split, "1")

        # User Request: Use full 3D shape (Batch, Time, Feature) for topology logging
        input_shape = dataset_split.train_X.shape if dataset_split.train_X is not None else ()

        metadata = DatasetMetadata(
            dataset=dataset_enum,
            dataset_name=dataset_preset.name,
            preset=dataset_preset,
            training=training_cfg,
            classification=dataset_preset.classification,
            input_shape=input_shape,
        )
        return metadata, dataset_split

    def _process_frontend(self, raw_split: SplitDataset) -> FrontendContext:
        """
        Step 2 & 3: Apply preprocessing and projection.
        """
        batch_size = self.metadata.training.batch_size
        print(f"\n=== Step 2: Preprocessing ===")
        preprocessing_config = self.config.preprocess
        pre_layers, preprocess_labels = create_preprocessor(preprocessing_config.method, poly_degree=preprocessing_config.poly_degree)

        data_split = raw_split
        train_X = data_split.train_X
        val_X = data_split.val_X
        test_X = data_split.test_X

        if pre_layers:
            train_X = apply_layers(pre_layers, train_X, fit=True)
            if val_X is not None:
                val_X = apply_layers(pre_layers, val_X, fit=False)
            if test_X is not None:
                test_X = apply_layers(pre_layers, test_X, fit=False)
                
            # Fix: For Regression, targets (y) should also be scaled if they share the domain (Auto-Regression)
            if not self.metadata.classification:
                 print("    [Preprocessing] Applying transforms to targets (y) for REGRESSION task.")
                 # Note: fit=False to reuse scaler fitted on X
                 if data_split.train_y is not None:
                     data_split = replace(data_split, train_y=apply_layers(pre_layers, data_split.train_y, fit=False))
                 if data_split.val_y is not None:
                     data_split = replace(data_split, val_y=apply_layers(pre_layers, data_split.val_y, fit=False))
                 if data_split.test_y is not None:
                     data_split = replace(data_split, test_y=apply_layers(pre_layers, data_split.test_y, fit=False))

        # Re-package for stats logging
        preprocessed_split = replace(data_split, train_X=train_X, val_X=val_X, test_X=test_X)
        self._log_dataset_stats(preprocessed_split, "2")
        
        # Use full 3D shape
        preprocessed_shape = train_X.shape

        print("\n=== Step 3: Projection (for reservoir/distillation) ===")
        projection_config = self.config.projection

        if projection_config is None:
            processed_split = SplitDataset(
                train_X=train_X,
                train_y=data_split.train_y,
                test_X=test_X,
                test_y=data_split.test_y,
                val_X=val_X,
                val_y=data_split.val_y,
            )
            input_shape_for_meta = preprocessed_shape
            input_dim_for_factory = int(preprocessed_shape[-1])
            self._log_dataset_stats(processed_split, "3")
            
            return FrontendContext(
                processed_split=processed_split,
                preprocess_labels=preprocess_labels,
                preprocessors=pre_layers,
                preprocessed_shape=preprocessed_shape,
                projected_shape=None,
                input_shape_for_meta=input_shape_for_meta,
                input_dim_for_factory=input_dim_for_factory,
                scaler=pre_layers[0] if pre_layers else None,
            )

        projection = InputProjection(
            input_dim=int(preprocessed_shape[-1]),
            output_dim=int(projection_config.n_units),
            input_scale=float(projection_config.input_scale),
            input_connectivity=float(projection_config.input_connectivity),
            seed=int(projection_config.seed),
            bias_scale=float(projection_config.bias_scale),
        )

        desc = f"[Projection] (Batch: {batch_size})"
        print(f"Applying Projection in batches of {batch_size}...")
        projected_train = batched_compute(projection, train_X, batch_size, desc=desc + "train")
        # del train_X  # Managed by scope

        projected_val = None
        if val_X is not None:
            projected_val = batched_compute(projection, val_X, batch_size, desc=desc + "val")

        projected_test = None
        if test_X is not None:
            projected_test = batched_compute(projection, test_X, batch_size, desc=desc + "test")

        projected_shape = projected_train.shape # (Batch, Time, ProjUnits)
        input_dim_for_factory = int(projected_shape[-1])

        processed_split = SplitDataset(
            train_X=projected_train,
            train_y=data_split.train_y,
            test_X=projected_test,
            test_y=data_split.test_y,
            val_X=projected_val,
            val_y=data_split.val_y,
        )

        self._log_dataset_stats(processed_split, "3")

        return FrontendContext(
            processed_split=processed_split,
            preprocess_labels=preprocess_labels,
            preprocessors=pre_layers,
            preprocessed_shape=preprocessed_shape,
            projected_shape=projected_shape,
            input_shape_for_meta=projected_shape,
            input_dim_for_factory=input_dim_for_factory,
            scaler=pre_layers[0] if pre_layers else None,
            projection_layer=projection,
        )

    def apply_adapter(self, frontend_ctx: FrontendContext, adapter: Any) -> FrontendContext:
        """
        Step 4: Apply adapter (e.g., TimeDelayEmbedding) to all splits.
        This is called by the model builder after creating the model with adapter.
        """
        print("\n=== Step 4: Adapter ===")
        
        if adapter is None or not hasattr(adapter, '__call__'):
            return frontend_ctx
        
        processed = frontend_ctx.processed_split
        window_size = getattr(adapter, 'window_size', None)
        adapter_name = f"TimeDelayEmbedding(k={window_size})" if window_size else "Adapter"
        
        # Apply adapter to X (all splits)
        adapted_train_X = adapter(processed.train_X, log_label=f"4:{adapter_name}:X:train")
        adapted_val_X = adapter(processed.val_X, log_label=f"4:{adapter_name}:X:val") if processed.val_X is not None else None
        adapted_test_X = adapter(processed.test_X, log_label=f"4:{adapter_name}:X:test") if processed.test_X is not None else None
        
        # Align y (all splits) to match adapted X
        aligned_train_y = adapter.align_targets(processed.train_y, log_label=f"4:{adapter_name}:y:train") if processed.train_y is not None else None
        aligned_val_y = adapter.align_targets(processed.val_y, log_label=f"4:{adapter_name}:y:val") if processed.val_y is not None else None
        aligned_test_y = adapter.align_targets(processed.test_y, log_label=f"4:{adapter_name}:y:test") if processed.test_y is not None else None
        
        # Create new processed split with adapted data
        adapted_split = SplitDataset(
            train_X=adapted_train_X,
            train_y=aligned_train_y,
            val_X=adapted_val_X,
            val_y=aligned_val_y,
            test_X=adapted_test_X,
            test_y=aligned_test_y,
        )
        
        # Update input dimension for factory (now windowed)
        new_input_dim = int(adapted_train_X.shape[-1]) if adapted_train_X.ndim >= 2 else frontend_ctx.input_dim_for_factory
        
        from dataclasses import replace as dc_replace
        return dc_replace(
            frontend_ctx,
            processed_split=adapted_split,
            input_dim_for_factory=new_input_dim,
        )

    @staticmethod
    def _log_dataset_stats(dataset: SplitDataset, stage: str):
        """Centralized stats logging."""
        print_feature_stats(dataset.train_X, f"{stage}:X:train")
        if dataset.train_y is not None:
             print_feature_stats(dataset.train_y, f"{stage}:y:train")
             
        if dataset.val_X is not None:
            print_feature_stats(dataset.val_X, f"{stage}:X:val")
            if dataset.val_y is not None:
                print_feature_stats(dataset.val_y, f"{stage}:y:val")
                
        if dataset.test_X is not None:
            print_feature_stats(dataset.test_X, f"{stage}:X:test")
            if dataset.test_y is not None:
                print_feature_stats(dataset.test_y, f"{stage}:y:test")
