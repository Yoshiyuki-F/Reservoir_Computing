"""CLI implementation for the reservoir-cli command.

This module lives inside the ``core_lib`` package so that the
console_script entry point can be wired to ``core_lib.cli:main``.
"""

from __future__ import annotations

import argparse
import json
import sys

from core_lib.utils import check_gpu_available


def main() -> None:
    """Main function for the project CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-model ML framework with Reservoir Computing and other models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Usage examples:
            reservoir-cli --dataset sine_wave --model classic_standard --n-reservoir 600
            reservoir-cli --dataset lorenz --model gatebased_quantum --show-training
            reservoir-cli --dataset mnist --model analog_quantum --force-cpu
            reservoir-cli --dataset mackey_glass --model reservoir_large --training windowed
            reservoir-cli --dataset mnist --model classic_standard --n-reservoir 300 --force-cpu
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "classic_standard",
            "fnn_pretrained",
            "fnn_pretrained_b_dash",
            "reservoir_large",
            "reservoir_complex",
            "gatebased_quantum",
            "analog_quantum",
            "quantum_advanced",
        ],
        default="classic_standard",
        help="Model configuration to use (default: classic_standard)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["sine_wave", "lorenz", "mackey_glass", "mnist"],
        required=True,
        help="Dataset to use for the experiment",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage (by default GPU is required)",
    )

    parser.add_argument(
        "--training",
        type=str,
        choices=["standard", "robust", "windowed", "raw_standard"],
        default="standard",
        help="Training configuration to use (default: standard)",
    )

    parser.add_argument(
        "--show-training",
        action="store_true",
        help="Show training data in visualization",
    )

    parser.add_argument(
        "--n-reservoir",
        type=int,
        default=None,
        help="Override reservoir size for classical models",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (used for FNN models)",
    )

    args = parser.parse_args()

    model_name_lower = args.model.lower()
    is_fnn_model = "fnn" in model_name_lower
    requires_reservoir = "quantum" not in model_name_lower and not is_fnn_model
    if requires_reservoir and args.n_reservoir is None:
        parser.error("--n-reservoir is required for classical reservoir models")

    if is_fnn_model and args.dataset != "mnist":
        parser.error("fnn_pretrained model is currently supported only for the 'mnist' dataset")
    if is_fnn_model and args.config is None:
        parser.error("--config is required when using the fnn_pretrained model")

    if args.dataset == "mnist":
        print(" MNIST dataset detected; classification mode will be applied automatically.")

    print("Multi-model ML Framework")
    print("=" * 60)
    if args.n_reservoir is not None:
        print(f"Reservoir override: {args.n_reservoir}")
        print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Training: {args.training}")
    print("=" * 60)

    # Display experiment info
    experiment_name = f"{args.dataset}_{args.model}_{args.training}"
    print(f"Experiment: {experiment_name}")

    # Run dynamic experiment
    try:
        # GPU check
        if not args.force_cpu:
            try:
                check_gpu_available()
            except Exception as exc:  # pragma: no cover - env specific
                print(f"❌ GPU check failed: {exc}")
                sys.exit(1)

        backend = "cpu" if args.force_cpu else "gpu"

        if is_fnn_model:
            from configs.fnn_config import FNNPipelineConfig
            from pipelines.datasets.mnist_loader import get_mnist_dataloaders
            if args.model == "fnn_pretrained_b_dash":
                from pipelines.fnn_b_dash_pipeline import (
                    pretrain_fnn_b_dash as pretrain_fnn,
                    run_fnn_fixed_feature_pipeline_b_dash as run_fnn_fixed_feature_pipeline,
                )
            else:
                from pipelines.fnn_pipeline import (
                    pretrain_fnn,
                    run_fnn_fixed_feature_pipeline,
                )

            if args.config is None:
                raise ValueError("FNN models require --config pointing to a JSON config file")

            with open(args.config, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            fnn_config = FNNPipelineConfig(**cfg_dict)

            train_loader, test_loader = get_mnist_dataloaders(
                batch_size=fnn_config.training.batch_size,
                shuffle_train=True,
                num_workers=0,
            )

            print("[Phase 1] Running FNN pretraining phase...")
            epochs, train_hist, test_hist = pretrain_fnn(
                fnn_config,
                train_loader,
                test_loader,
            )
            print("[Phase 2] Running FNN fixed-feature ridge pipeline...")
            results = run_fnn_fixed_feature_pipeline(
                fnn_config,
                train_loader,
                test_loader,
                epochs,
                train_hist,
                test_hist,
            )

            print("Experiment completed successfully!")
            print(
                "📊 Results: "
                f"Train MSE: {results['train_mse']:.6f}, "
                f"Test MSE: {results['test_mse']:.6f}, "
                f"Train Acc: {results['train_accuracy']:.4f}, "
                f"Test Acc: {results['test_accuracy']:.4f}"
            )
            return

        from experiments.dynamic_runner import run_dynamic_experiment

        print("Running dynamic experiment...")

        result = run_dynamic_experiment(
            dataset_name=args.dataset,
            model_name=args.model,
            training_name=args.training,
            show_training=args.show_training,
            backend=backend,
            n_reservoir_override=args.n_reservoir,
        )

        print("Experiment completed successfully!")
        train_mse, test_mse, train_mae, test_mae = result
        if train_mse is not None:
            print(f"📊 Results: Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        else:
            print(f"📊 Results: Test MSE: {test_mse:.6f}")

    except Exception as exc:  # pragma: no cover - top-level error
        print(f"❌ Error running experiment: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

