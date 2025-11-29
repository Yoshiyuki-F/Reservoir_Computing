"""/home/yoshi/PycharmProjects/Reservoir/cli/main.py
Unified CLI that delegates to pipelines.run.run_pipeline.

Legacy positional CLI has been removed; use --unified-model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from reservoir.utils import check_gpu_available
from pipelines import run_pipeline


def _run_unified_pipeline_cli(args) -> None:
    dataset_name = (args.dataset or args.dataset_pos or "sine_wave").lower()
    model_type = args.unified_model.lower()

    config: Dict[str, Any] = {
        "model_type": model_type,
        "dataset": dataset_name,
        "hidden_dim": args.unified_hidden,
        "training": {
            "num_epochs": args.unified_epochs,
            "batch_size": 32,
            "learning_rate": 1e-3,
        },
        "use_design_matrix": args.use_design_matrix,
        "poly_degree": args.poly_degree,
    }

    # Reservoir config merging
    if model_type == "reservoir":
        cfg_path: Path
        if args.reservoir_config_path:
            cfg_path = Path(args.reservoir_config_path)
        else:
            cfg_path = Path(__file__).resolve().parents[1] / "presets" / "models" / "shared_reservoir_params.json"
        if not cfg_path.exists():
            raise SystemExit(f"Reservoir config not found: {cfg_path}")
        with cfg_path.open("r") as f:
            reservoir_cfg = json.load(f)
        params = reservoir_cfg.setdefault("params", {})
        params.setdefault("n_hidden_layer", int(args.unified_hidden))
        config["reservoir_config"] = reservoir_cfg
        config["reservoir_type"] = "classical"

    print(f"[Unified] Running {model_type} pipeline on {dataset_name}...")
    results = run_pipeline(config)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ML Framework CLI")
    parser.add_argument("--unified-model", choices=["fnn", "rnn", "reservoir"], required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("dataset_pos", nargs="?", type=str)
    parser.add_argument("--unified-hidden", type=int, default=32)
    parser.add_argument("--unified-epochs", type=int, default=3)
    parser.add_argument("--unified-seq-len", type=int, default=50)
    parser.add_argument("--reservoir-config-path", type=str, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--use-design-matrix", action="store_true", help="Enable polynomial feature expansion")
    parser.add_argument("--poly-degree", type=int, default=2, help="Polynomial expansion degree")

    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except Exception as exc:  # pragma: no cover - env specific
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    _run_unified_pipeline_cli(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
