"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/cli/main.py
Unified CLI that delegates to reservoir.pipelines.run.run_pipeline.

Launch: uv run python -m reservoir.cli.main --model fnn --dataset mnist
NOTE: Do NOT use reservoir-cli entry point — JAX 0.9 x64 will silently fail.
"""

from __future__ import annotations

import argparse
import sys

from reservoir.training import get_training_preset
from reservoir.utils import check_gpu_available
from reservoir.core.identifiers import Model, Dataset
from reservoir.pipelines import run_pipeline
from reservoir.models.presets import get_model_preset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import ConfigDict

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ML Framework CLI")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[p.value for p in Model],
        help="Model preset name (Pipeline enum value, e.g., classical_reservoir, fnn_distillation)",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--force-cpu", action="store_true")

    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except Exception as exc:
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    # Build Config (strict preset + dataset only)
    model_enum = Model(args.model)
    dataset = Dataset(args.dataset)
    pipeline_config = get_model_preset(model_enum, dataset)
    
    # Auto-select "quantum" training config for Quantum Reservoir to prevent OOM
    training_preset_name = "standard"
    if pipeline_config.model_type == Model.QUANTUM_RESERVOIR:
        training_preset_name = "quantum"
        
    training_config = get_training_preset(training_preset_name)

    print(f"[Unified] Running {pipeline_config.name} pipeline on {dataset.name}...")
    print(f"With training preset: '{training_config.name}'")
    print("\n=== Configuration ===")
    print(f"Pipeline    : {pipeline_config.name}")
    print(f"Description : {pipeline_config.description}")
    print(f"Dataset     : {dataset.name}")
    print(f"Model Class : {type(pipeline_config.model).__name__}")
    if pipeline_config.readout:
        print(f"Readout Class: {type(pipeline_config.readout).__name__}")
    
    import json
    def print_section(name: str, data: ConfigDict):
        if not data:
            return
        print(f"\n[{name}]")
        # filter out None values for cleaner output
        clean_data: ConfigDict = {k: v for k, v in data.items() if v is not None}
        print(json.dumps(clean_data, indent=2))

    print_section("Preprocessing", pipeline_config.preprocess.to_dict())
    
    if pipeline_config.projection:
        print_section("Projection", pipeline_config.projection.to_dict())
        
    model_dict = pipeline_config.model.to_dict()
    model_dict["_config_type"] = type(pipeline_config.model).__name__
    print_section("Model Dynamics", model_dict)
    
    if pipeline_config.readout:
        readout_dict = pipeline_config.readout.to_dict()
        readout_dict["_config_type"] = type(pipeline_config.readout).__name__
        print_section("Readout", readout_dict)

    print_section("Training", training_config.to_dict())
    print("=" * 21 + "\n")

    # Run Pipeline
    run_pipeline(pipeline_config, dataset, training_config)

    sys.exit(0)


if __name__ == "__main__":
    main()
