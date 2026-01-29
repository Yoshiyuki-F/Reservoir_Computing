"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/cli/main.py
Unified CLI that delegates to reservoir.pipelines.run.run_pipeline."""

from __future__ import annotations

import argparse
import sys

from reservoir.training import get_training_preset
from reservoir.utils import check_gpu_available
from reservoir.core.identifiers import Model, Dataset
from reservoir.pipelines import run_pipeline
from reservoir.models.presets import get_model_preset

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
    training_config = get_training_preset("standard")

    print(f"[Unified] Running {pipeline_config.name} pipeline on {dataset.name}...")
    print(f"With training preset: '{training_config.name}'")
    print("\n=== Configuration ===")
    print(f"Pipeline    : {pipeline_config.name}")
    print(f"Description : {pipeline_config.description}")
    print(f"Dataset     : {dataset.name}")
    
    import json
    def print_section(name: str, data: dict):
        if not data: return
        print(f"\n[{name}]")
        # filter out None values for cleaner output
        clean_data = {k: v for k, v in data.items() if v is not None}
        print(json.dumps(clean_data, indent=2))

    print_section("Preprocessing", pipeline_config.preprocess.to_dict())
    
    if pipeline_config.projection:
        print_section("Projection", pipeline_config.projection.to_dict())
        
    print_section("Model Dynamics", pipeline_config.model.to_dict())
    
    if pipeline_config.readout:
        print_section("Readout", pipeline_config.readout.to_dict())

    print_section("Training", training_config.to_dict())
    print("=" * 21 + "\n")

    # Run Pipeline
    results = run_pipeline(pipeline_config, dataset, training_config)

    # Output Results
    print("[Unified] Results:")
    if isinstance(results, dict):
        for split, metrics in results.items():
            if split in ("outputs", "training_logs", "readout"):
                continue
            if isinstance(metrics, dict):
                pretty = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics.items()
                )
                print(f"  {split}: {pretty}")
            else:
                print(f"  {split}: {metrics}")

    sys.exit(0)


if __name__ == "__main__":
    main()
