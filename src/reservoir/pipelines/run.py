"""
pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets.

V2 Architecture Compliance:
- Strict Configuration: No implicit defaults. Rely entirely on Presets + User Config.
- Canonical Names Only: No alias resolution (e.g., 'alpha' -> 'leak_rate') happens here.
- Fail Fast: Dictionary access raises KeyError if parameters are missing.
"""

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple
from typing import Any as _Any

import numpy as np

# Core Imports
from reservoir.models import ModelFactory
from reservoir.core.identifiers import Dataset, Model, TaskType
from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
from reservoir.utils.printing import print_topology
from reservoir.pipelines.generic_runner import UniversalPipeline
from reservoir.readout.factory import ReadoutFactory
from reservoir.data.loaders import load_dataset_with_validation_split
from reservoir.data.presets import DATASET_REGISTRY
from reservoir.data.structs import SplitDataset
from reservoir.training.presets import get_training_preset, TrainingConfig
from reservoir.layers.preprocessing import create_preprocessor
from reservoir.layers.projection import InputProjection
from reservoir.models.presets import PipelineConfig
from reservoir.utils.reporting import generate_report

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401

def _apply_layers(layers: list[_Any], data: np.ndarray, *, fit: bool = False) -> np.ndarray:
    """Sequentially apply preprocessing layers."""
    arr = data
    for layer in layers:
        if fit and hasattr(layer, "fit_transform"):
            arr = layer.fit_transform(arr)
            fit = False
        elif fit and hasattr(layer, "fit") and hasattr(layer, "transform"):
            layer.fit(arr)
            arr = layer.transform(arr)
            fit = False
        elif hasattr(layer, "transform"):
            arr = layer.transform(arr)
        else:
            arr = layer(arr)
    return np.asarray(arr)


def _log_split_stats(stage: str, train_X: np.ndarray, val_X: Optional[np.ndarray], test_X: Optional[np.ndarray]) -> None:
    """Lightweight stats logger for each split at a given processing stage."""
    def _stats(arr: np.ndarray) -> str:
        arr64 = np.asarray(arr)
        sample = arr64
        if sample.shape[0] > 2000:
            sample = sample[:2000]
        return (
            f"shape={arr64.shape}, mean={float(np.mean(sample)):.4f}, std={float(np.std(sample)):.4f}, "
            f"min={float(np.min(sample)):.4f}, max={float(np.max(sample)):.4f}, nans={int(np.isnan(sample).sum())}"
        )

    print(f"[FeatureStats:{stage}:train] {_stats(train_X)}")
    if val_X is not None:
        print(f"[FeatureStats:{stage}:val] {_stats(val_X)}")
    if test_X is not None:
        print(f"[FeatureStats:{stage}:test] {_stats(test_X)}")


def _prepare_dataset(dataset: Dataset, training_override: Optional[TrainingConfig] = None) -> Tuple[DatasetMetadata, SplitDataset]:
    """Step 1: Resolve presets and load dataset without mutating inputs later."""
    print("=== Step 1: Loading Dataset ===")
    dataset_enum, dataset_preset = dataset, DATASET_REGISTRY.get(dataset)

    training_cfg = training_override or replace(
        get_training_preset("standard"), classification=dataset.task_type is TaskType.CLASSIFICATION
    )
    dataset_split = load_dataset_with_validation_split(
        dataset,
        training_cfg,
        require_3d=True,
    )

    _log_split_stats("raw", dataset_split.train_X, dataset_split.val_X, dataset_split.test_X)

    input_shape = tuple(dataset_split.train_X.shape[1:]) if dataset_split.train_X is not None else ()

    metadata = DatasetMetadata(
        dataset=dataset_enum,
        dataset_name=dataset_preset.name,
        preset=dataset_preset,
        training=training_cfg,
        task_type=dataset_preset.task_type,
        input_shape=input_shape,
    )
    return metadata, dataset_split


def _process_frontend(config: PipelineConfig, raw_split: SplitDataset, dataset_meta: DatasetMetadata) -> FrontendContext:
    """
    Step 2 & 3: Apply preprocessing and projection without mutating raw splits.
    Returns processed datasets and shape metadata for model creation.
    """
    _ = dataset_meta  # metadata retained for symmetry; data flow uses raw_split directly
    print("\n=== Step 2: Preprocessing ===")
    preprocessing_config = config.preprocess
    pre_layers, preprocess_labels = create_preprocessor(preprocessing_config.method, poly_degree=preprocessing_config.poly_degree)

    data_split = raw_split
    train_X = data_split.train_X
    val_X = data_split.val_X
    test_X = data_split.test_X

    if pre_layers:
        train_X = _apply_layers(pre_layers, train_X, fit=True)
        if val_X is not None:
            val_X = _apply_layers(pre_layers, val_X, fit=False)
        if test_X is not None:
            test_X = _apply_layers(pre_layers, test_X, fit=False)
    _log_split_stats("preprocess", train_X, val_X, test_X)
    preprocessed_shape = train_X.shape[1:]

    print("\n=== Step 3: Projection (for reservoir/distillation) ===")
    projection_config = config.projection

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
        input_dim_for_factory = (
            int(np.prod(input_shape_for_meta)) if config.model_type is Model.FNN else int(input_shape_for_meta[-1])
        )
        _log_split_stats("projection", train_X, val_X, test_X)
        return FrontendContext(
            processed_split=processed_split,
            preprocess_labels=preprocess_labels,
            preprocessed_shape=preprocessed_shape,
            projected_shape=None,
            input_shape_for_meta=input_shape_for_meta,
            input_dim_for_factory=input_dim_for_factory,
        )

    projection = InputProjection(
        input_dim=int(preprocessed_shape[-1]),
        output_dim=int(projection_config.n_units),
        input_scale=float(projection_config.input_scale),
        input_connectivity=float(projection_config.input_connectivity),
        seed=int(projection_config.seed),
        bias_scale=float(projection_config.bias_scale),
    )


    projected_train = projection(train_X)
    projected_test = projection(test_X)

    projected_val = val_X
    if val_X is not None:
        projected_val = projection(val_X)
    projected_shape = projected_train.shape[1:]

    transformed_shape = projected_shape or preprocessed_shape
    input_shape_for_meta: tuple[int, ...] = transformed_shape
    input_dim_for_factory = (
        int(np.prod(transformed_shape)) if config.model_type is Model.FNN else int(transformed_shape[-1])
    )

    processed_split = SplitDataset(
        train_X=projected_train,
        train_y=data_split.train_y,
        test_X=projected_test,
        test_y=data_split.test_y,
        val_X=projected_val,
        val_y=data_split.val_y,
    )

    _log_split_stats("projection", projected_train, projected_val, projected_test)
    return FrontendContext(
        processed_split=processed_split,
        preprocess_labels=preprocess_labels,
        preprocessed_shape=preprocessed_shape,
        projected_shape=projected_shape,
        input_shape_for_meta=input_shape_for_meta,
        input_dim_for_factory=input_dim_for_factory,
    )


def _build_model_stack(
    config: PipelineConfig,
    dataset_meta: DatasetMetadata,
    frontend_ctx: FrontendContext,
) -> ModelStack:

    print("\n=== Step 4: Build Model Stack ===")


    """Step 4: Instantiate model + readout and enrich topology metadata."""
    processed = frontend_ctx.processed_split
    if dataset_meta.preset.config.n_output is not None:
        meta_n_outputs = int(dataset_meta.preset.config.n_output)
    else:
        target_sample = processed.train_y if processed.train_y is not None else processed.test_y
        if target_sample is None:
            raise ValueError("Unable to infer output dimension without targets.")
        meta_n_outputs = (
            int(target_sample.shape[-1]) if hasattr(target_sample, "shape") and len(target_sample.shape) > 1 else 1
        )

    model = ModelFactory.create_model(
        config=config,
        training=dataset_meta.training,
        input_dim=frontend_ctx.input_dim_for_factory,
        output_dim=meta_n_outputs,
        input_shape=frontend_ctx.input_shape_for_meta,
    )

    topo_meta = model.get_topology_meta() if hasattr(model, "get_topology_meta") else {}
    if not isinstance(topo_meta, dict):
        topo_meta = {}
    shapes_meta = topo_meta.get("shapes", {}) or {}
    shapes_meta["input"] = dataset_meta.input_shape
    shapes_meta["preprocessed"] = frontend_ctx.preprocessed_shape
    shapes_meta["projected"] = frontend_ctx.projected_shape
    shapes_meta["output"] = (meta_n_outputs,)
    topo_meta["shapes"] = shapes_meta
    details_meta = topo_meta.get("details", {}) or {}
    details_meta["preprocess"] = "-".join(frontend_ctx.preprocess_labels) if frontend_ctx.preprocess_labels else None
    topo_meta["details"] = details_meta
    print_topology(topo_meta)

    metric = "accuracy" if dataset_meta.task_type is TaskType.CLASSIFICATION else "mse"
    readout = ReadoutFactory.create_readout(config.readout)
    return ModelStack(
        model=model,
        readout=readout,
        topo_meta=topo_meta,
        metric=metric,
        model_label=config.model_type.value,
    )


def run_pipeline(config: PipelineConfig, dataset: Dataset, training_config: Optional[TrainingConfig]=None) -> Dict[str, Any]:
    """
    Declarative orchestrator for the unified pipeline.
    """

    #Step1.
    dataset_meta, raw_split = _prepare_dataset(dataset, training_config)

    #Step2&3.
    frontend_ctx = _process_frontend(config, raw_split, dataset_meta)
    del raw_split

    #Step 4: Build Model Stack
    stack = _build_model_stack(config, dataset_meta, frontend_ctx)

    runner = UniversalPipeline(stack, config)
    results = runner.run(frontend_ctx, dataset_meta)

    report_payload = dict(
        runner=runner,
        readout=stack.readout,
        train_X=frontend_ctx.processed_split.train_X,
        train_y=frontend_ctx.processed_split.train_y,
        test_X=frontend_ctx.processed_split.test_X,
        test_y=frontend_ctx.processed_split.test_y,
        val_X=frontend_ctx.processed_split.val_X,
        val_y=frontend_ctx.processed_split.val_y,
        training_obj=dataset_meta.training,
        dataset_name=dataset_meta.dataset_name,
        model_type_str=stack.model_label,
    )
    generate_report(
        results,
        config,
        stack.topo_meta,
        **report_payload,
    )

    return results
