"""CLI entry point for reservoir-cli command."""

import argparse
import sys
from pathlib import Path

def main():
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
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'classic_standard',
            'reservoir_large',
            'reservoir_complex',
            'gatebased_quantum',
            'analog_quantum',
            'quantum_advanced'
        ],
        default='classic_standard',
        help='Model configuration to use (default: classic_standard)'
    )

    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=['sine_wave', 'lorenz', 'mackey_glass', 'mnist'],
        required=True,
        help='Dataset to use for the experiment'
    )

    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage (by default GPU is required)'
    )

    parser.add_argument(
        '--training',
        type=str,
        choices=['standard', 'robust', 'windowed', 'raw_standard'],
        default='standard',
        help='Training configuration to use (default: standard)'
    )

    parser.add_argument(
        '--show-training',
        action='store_true',
        help='Show training data in visualization'
    )

    parser.add_argument(
        '--n-reservoir',
        type=int,
        default=None,
        help='Override reservoir size for classical models'
    )

    args = parser.parse_args()

    model_name_lower = args.model.lower()
    requires_reservoir = "quantum" not in model_name_lower
    if requires_reservoir and args.n_reservoir is None:
        parser.error("--n-reservoir is required for classical reservoir models")

    if args.dataset == 'mnist':
        print(" MNIST dataset detected; classification mode will be applied automatically.")

    print(f"Multi-model ML Framework")
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
                from pipelines.gpu_utils import check_gpu_available
                check_gpu_available()
            except Exception as e:
                print(f"❌ GPU check failed: {e}")
                sys.exit(1)

        backend = 'cpu' if args.force_cpu else 'gpu'

        # Import and run dynamic experiment
        from experiments.dynamic_runner import run_dynamic_experiment

        print(f"Running dynamic experiment...")

        result = run_dynamic_experiment(
            dataset_name=args.dataset,
            model_name=args.model,
            training_name=args.training,
            show_training=args.show_training,
            backend=backend,
            n_reservoir_override=args.n_reservoir,
        )

        print(f"Experiment completed successfully!")
        train_mse, test_mse, train_mae, test_mae = result
        if train_mse is not None:
            print(f"📊 Results: Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        else:
            print(f"📊 Results: Test MSE: {test_mse:.6f}")

    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
