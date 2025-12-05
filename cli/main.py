"""Unified CLI that delegates to pipelines.run.run_pipeline."""

from __future__ import annotations

import argparse
import sys

from reservoir.utils.jax_config import ensure_x64_enabled
ensure_x64_enabled()

from reservoir.utils import check_gpu_available
from reservoir.core.config_builder import build_run_config
from pipelines import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ML Framework CLI")

    # Required dont add more!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! other things should be defined in presets.py(training/ or models/ or data/)
    parser.add_argument("--model", type=str, required=True, help="Model preset name (e.g., classical, fnn-distillation)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--force-cpu", action="store_true")

    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except Exception as exc:
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    # Build Config (strict preset + dataset only)
    config = build_run_config(preset_name=args.model, dataset_name=args.dataset)

    print(f"[Unified] Running {config['model_type']} pipeline on {config['dataset']}...")

    # Run Pipeline
    results = run_pipeline(config)

    # Output Results
    print("[Unified] Results:")
    if isinstance(results, dict):
        for split, metrics in results.items():
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
