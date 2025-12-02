"""Unified CLI that delegates to pipelines.run.run_pipeline."""

from __future__ import annotations

import argparse
import sys

from reservoir.utils import check_gpu_available
from reservoir.core.config_builder import build_run_config
from pipelines import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ML Framework CLI")

    # Required
    parser.add_argument("--unified-model", choices=["fnn", "rnn", "reservoir"], required=True)

    # Dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("dataset_pos", nargs="?", type=str, help="Positional alias for dataset")

    # Hyperparameters
    parser.add_argument("--unified-hidden", type=int, default=None)
    parser.add_argument("--unified-epochs", type=int, default=None)
    parser.add_argument("--unified-batch-size", type=int, default=None)
    parser.add_argument("--unified-lr", type=float, default=None)
    parser.add_argument("--unified-seq-len", type=int, default=50)
    parser.add_argument(
        "--nn-hidden",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer dimensions for FNN (Student in distillation mode).",
    )

    # Presets
    parser.add_argument(
        "--reservoir-preset",
        type=str,
        default="classical",
        help="Reservoir preset name (classical, quantum_gate_based, etc.)",
    )
    parser.add_argument(
        "--training-preset",
        type=str,
        default="standard",
        help="Training preset name (standard, quick_test, heavy)",
    )

    # Features
    parser.add_argument(
        "--use-design-matrix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable polynomial feature expansion (override preset)",
    )
    parser.add_argument("--poly-degree", type=int, default=None, help="Polynomial expansion degree (override preset)")

    # Environment
    parser.add_argument("--force-cpu", action="store_true")

    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except Exception as exc:
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    # Build Config
    dataset_name = args.dataset or args.dataset_pos
    config = build_run_config(
        model_type=args.unified_model,
        dataset=dataset_name,
        hidden_dim=args.unified_hidden,
        epochs=args.unified_epochs,
        batch_size=args.unified_batch_size,
        learning_rate=args.unified_lr,
        seq_len=args.unified_seq_len,
        reservoir_preset=args.reservoir_preset,
        training_preset=args.training_preset,
        use_design_matrix=args.use_design_matrix,
        poly_degree=args.poly_degree,
        nn_hidden=args.nn_hidden,
    )

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
