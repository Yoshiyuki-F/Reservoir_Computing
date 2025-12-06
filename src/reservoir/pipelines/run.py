"""
pipelines/run.py
Unified Pipeline Runner for JAX-based Models and Datasets.

V2 Architecture Compliance:
- Strict Configuration: No implicit defaults. Rely entirely on Presets + User Config.
- Canonical Names Only: No alias resolution (e.g., 'alpha' -> 'leak_rate') happens here.
- Fail Fast: Dictionary access raises KeyError if parameters are missing.
"""

from dataclasses import replace
from typing import Dict, Any, Optional, Tuple
from typing import Any as _Any

import numpy as np

# Core Imports
from reservoir.models import ModelFactory
from reservoir.core.identifiers import Dataset, Pipeline, TaskType, RunConfig, Preprocessing
from reservoir.utils.printing import print_topology
from reservoir.pipelines.generic_runner import UniversalPipeline
from reservoir.readout.ridge import RidgeRegression
from reservoir.data.loaders import load_dataset_with_validation_split
from reservoir.data.presets import DATASET_REGISTRY, DatasetPreset
from reservoir.training.presets import get_training_preset, TrainingConfig
from reservoir.layers.preprocessing import create_preprocessor
from reservoir.layers.projection import InputProjection
from reservoir.models.presets import get_model_preset, DistillationConfig
from reservoir.utils.reporting import generate_report

# Ensure dataset loaders are registered
from reservoir.data import loaders as _data_loaders  # noqa: F401


def _get_strict_dataset_meta(dataset: Dataset) -> Tuple[Dataset, DatasetPreset]:
    """
    Retrieve dataset metadata.
    Raises ValueError if the dataset is unknown or metadata is incomplete.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Configuration Error: 'dataset' must be a Dataset Enum, got {type(dataset)}.")

    preset = DATASET_REGISTRY.get(dataset)
    if preset is None:
        raise ValueError(f"Configuration Error: Dataset '{dataset}' is not registered in DATASET_REGISTRY.")

    if preset.config.n_input is None:
        raise ValueError(f"Preset Error: Dataset '{dataset}' is missing 'n_input' in its definition.")
    if preset.config.n_output is None:
        raise ValueError(f"Preset Error: Dataset '{dataset}' is missing 'n_output' in its definition.")

    return dataset, preset


def _resolve_training_config(task_type: TaskType) -> TrainingConfig:
    """
    Resolve training configuration strictly from presets.
    """
    preset = get_training_preset("standard")
    classification = task_type is TaskType.CLASSIFICATION
    return replace(preset, classification=classification)


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


def run_pipeline(config: RunConfig) -> Dict[str, Any]:
    """
    The Unified Entry Point (V2 Strict Mode).
    """
    if not isinstance(config, RunConfig):
        raise TypeError(f"run_pipeline requires RunConfig, got {type(config)}.")

    print("=== Step 1: Loading Dataset ===")
    dataset_enum, dataset_preset = _get_strict_dataset_meta(config.dataset)
    dataset_name = dataset_enum.value
    task_type = config.task_type

    training_obj = _resolve_training_config(task_type)

    preset_model = get_model_preset(config.model_type)
    if preset_model is None or preset_model.config is None:
        raise ValueError(f"Model preset for {config.model_type} is missing configuration.")

    dataset_split = load_dataset_with_validation_split(
        config,
        training_obj,
        model_type=config.model_type.value,
        require_3d=True,
    )
    train_X = dataset_split.train_X
    train_y = dataset_split.train_y
    test_X = dataset_split.test_X
    test_y = dataset_split.test_y
    val_X = dataset_split.val_X
    val_y = dataset_split.val_y

    input_shape_original = train_X.shape[1:]

    print("\n=== Step 2: Preprocessing ===")
    poly_degree = int(getattr(preset_model.config, "poly_degree", 1))
    preprocessing_enum = config.preprocessing
    preprocess_labels: list[str] = []
    if preprocessing_enum == Preprocessing.STANDARD_SCALER:
        preprocess_labels.append("scaler")
    elif preprocessing_enum == Preprocessing.DESIGN_MATRIX:
        preprocess_labels.extend(["scaler", f"poly{poly_degree}"])
    pre_layers = create_preprocessor(preprocessing_enum, poly_degree=poly_degree)
    if pre_layers:
        train_X = _apply_layers(pre_layers, train_X, fit=True)
        if val_X is not None:
            val_X = _apply_layers(pre_layers, val_X, fit=False)
        if test_X is not None:
            test_X = _apply_layers(pre_layers, test_X, fit=False)
    preprocessed_shape = train_X.shape[1:]


    print("\n=== Step 3: Projection (for reservoir/distillation) ===")
    projected_shape: Optional[tuple[int, ...]] = None
    if config.model_type in {Pipeline.CLASSICAL_RESERVOIR, Pipeline.QUANTUM_GATE_BASED, Pipeline.QUANTUM_ANALOG}:
        res_cfg = preset_model.config
        projection = InputProjection(
            input_dim=int(preprocessed_shape[-1]),
            output_dim=int(res_cfg.n_units),
            input_scale=float(res_cfg.input_scale),
            input_connectivity=float(res_cfg.input_connectivity),
            seed=int(res_cfg.seed or 0),
            bias_scale=float(res_cfg.bias_scale),
        )
        train_X = projection(train_X)
        if val_X is not None:
            val_X = projection(val_X)
        if test_X is not None:
            test_X = projection(test_X)
        projected_shape = train_X.shape[1:]
    elif config.model_type is Pipeline.FNN_DISTILLATION:
        distill_cfg = preset_model.config
        if not isinstance(distill_cfg, DistillationConfig):
            raise ValueError("FNN_DISTILLATION preset must define DistillationConfig.")
        teacher_cfg = distill_cfg.teacher
        projection = InputProjection(
            input_dim=int(preprocessed_shape[-1]),
            output_dim=int(teacher_cfg.n_units),
            input_scale=float(teacher_cfg.input_scale),
            input_connectivity=float(teacher_cfg.input_connectivity),
            seed=int(teacher_cfg.seed or 0),
            bias_scale=float(teacher_cfg.bias_scale),
        )
        train_X = projection(train_X)
        if val_X is not None:
            val_X = projection(val_X)
        if test_X is not None:
            test_X = projection(test_X)
        projected_shape = train_X.shape[1:]

    # Determine model input dimensions after preprocessing/projection
    transformed_shape = projected_shape or preprocessed_shape
    input_dim_for_factory = int(np.prod(transformed_shape)) if config.model_type is Pipeline.FNN else int(transformed_shape[-1])
    input_shape_for_meta: tuple[int, ...] = transformed_shape

    if dataset_preset.config.n_output is not None:
        meta_n_outputs = int(dataset_preset.config.n_output)
    else:
        target_sample = train_y if train_y is not None else test_y
        if target_sample is None:
            raise ValueError("Unable to infer output dimension without targets.")
        meta_n_outputs = int(target_sample.shape[-1]) if hasattr(target_sample, "shape") and len(target_sample.shape) > 1 else 1

    print(f"Data Shapes -> Train: {train_X.shape}, Val: {getattr(val_X, 'shape', None)}, Test: {test_X.shape}")

    # Build model via Factory (all domain logic lives there)
    model = ModelFactory.create_model(
        config=config,
        dataset_preset=dataset_preset,
        training=training_obj,
        input_dim=input_dim_for_factory,
        output_dim=meta_n_outputs,
        input_shape=input_shape_for_meta,
    )

    model_type_str = config.model_type.value
    topo_meta = model.get_topology_meta() if hasattr(model, "get_topology_meta") else {}
    if not isinstance(topo_meta, dict):
        topo_meta = {}
    shapes_meta = topo_meta.get("shapes", {}) or {}
    shapes_meta["input"] = input_shape_original
    shapes_meta["preprocessed"] = preprocessed_shape
    shapes_meta["projected"] = projected_shape
    shapes_meta["output"] = (meta_n_outputs,)
    topo_meta["shapes"] = shapes_meta
    details_meta = topo_meta.get("details", {}) or {}
    details_meta["preprocess"] = "-".join(preprocess_labels) if preprocess_labels else None
    topo_meta["details"] = details_meta
    print_topology(topo_meta)

    readout = RidgeRegression(ridge_lambda=float(training_obj.ridge_lambda), use_intercept=True)

    metric = "accuracy" if task_type is TaskType.CLASSIFICATION else "mse"
    runner = UniversalPipeline(model, readout, None, metric=metric)
    validation_tuple = (val_X, val_y) if (val_X is not None and val_y is not None) else None
    results = runner.run(
        train_X,
        train_y,
        test_X,
        test_y,
        validation=validation_tuple,
        training_cfg=training_obj,
        training_extras={},
        model_label=model_type_str,
    )

    generate_report(
        results,
        config,
        topo_meta,
        runner=runner,
        readout=readout,
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        val_X=val_X,
        val_y=val_y,
        training_obj=training_obj,
        dataset_name=dataset_name,
        model_type_str=model_type_str,
    )

    return results
