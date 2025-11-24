"""CLI implementation for the ``reservoir-cli`` command.

This module lives in the top-level ``cli`` package so that the
core library (``src/core_lib``) remains UI-agnostic. The console
script entry point is wired to ``cli.main:main``.
"""

from __future__ import annotations

import argparse
import sys

from core_lib.utils import check_gpu_available


def _parse_preprocessing(value: str) -> str:
    """Normalize preprocessing argument, supporting 'DefaultRaw' alias."""
    v = value.lower()
    aliases = {
        "default": "default",       # scaled + design matrix
        "scaled": "default",
        "preprocessed": "default",
        "raw": "raw",               # raw reservoir states
        "defaultraw": "raw",
        "default_raw": "raw",
    }
    if v in aliases:
        return aliases[v]
    raise SystemExit(
        f"Unknown preprocessing '{value}'. "
        "Use one of: default|scaled, raw|DefaultRaw."
    )


def _parse_dataset(value: str) -> str:
    """Normalize dataset argument, supporting short aliases."""
    v = value.lower()
    aliases = {
        "sine_wave": "sine_wave",
        "sine": "sine_wave",
        "sw": "sine_wave",
        "lorenz": "lorenz",
        "lz": "lorenz",
        "mackey_glass": "mackey_glass",
        "mackey": "mackey_glass",
        "mg": "mackey_glass",
        "mnist": "mnist",
        "m": "mnist",
    }
    if v in aliases:
        return aliases[v]
    raise SystemExit(
        f"Unknown dataset '{value}'. "
        "Use one of: sine_wave|sine|sw, lorenz|lz, mackey_glass|mackey|mg, mnist|m."
    )


def _parse_model(value: str) -> str:
    """Normalize model argument, supporting short aliases."""
    v = value.lower()
    aliases = {
        "classical_reservoir": "classical_reservoir",
        "cr": "classical_reservoir",
        "classical": "classical_reservoir",
        "gate_based_quantum": "gate_based_quantum",
        "gq": "gate_based_quantum",
        "gate_based": "gate_based_quantum",
    }
    if v in aliases:
        return aliases[v]
    raise SystemExit(
        f"Unknown model '{value}'. Use a full name such as "
        "classical_reservoir, gate_based_quantum, analog_quantum, "
        "quantum_advanced, fnn_pretrained, or their aliases "
        "(cr, qr, fnn, gq, aq, qa, ...)."
    )


def main() -> None:
    """Main function for the project CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-model ML framework with Reservoir Computing and other models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Usage examples:
            reservoir-cli --dataset sine_wave --model classical_reservoir --n-hidden_layer 600
            reservoir-cli sine cr 600
            reservoir-cli --dataset lorenz --model gate_based_quantum --show-training
            reservoir-cli --dataset mnist --model analog_quantum --force-cpu
            reservoir-cli --dataset mackey_glass --model reservoir_large --training windowed
            reservoir-cli --dataset mnist --model classical_reservoir --n-hidden_layer 300 --force-cpu
        """,
    )

    parser.add_argument(
        "--model",
        type=_parse_model,
        choices=[
            "classical_reservoir",
            "gate_based_quantum",
        ],
        default="classical_reservoir",
        help="Model configuration to use (default: classical_reservoir)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=_parse_dataset,
        choices=["sine_wave", "lorenz", "mackey_glass", "mnist"],
        required=False,
        default=None,
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
        choices=["standard"],
        default="standard",
        help="Training configuration to use (default: standard)",
    )

    parser.add_argument(
        "--preprocessing",
        type=_parse_preprocessing,
        choices=["default", "raw"],
        default="raw",
        help="Preprocessing mode for classical reservoir features "
             "(raw: disable scaler/design matrix; default: apply them)",
    )

    parser.add_argument(
        "--show-training",
        action="store_true",
        help="Show training data in visualization",
    )

    parser.add_argument(
        "--n-hidden_layer",
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

    # Optional positional shorthand: DATASET MODEL [N_RESERVOIR]
    parser.add_argument(
        "dataset_pos",
        nargs="?",
        type=_parse_dataset,
        help="Optional dataset shorthand (e.g. sine, lorenz, mg, mnist)",
    )
    parser.add_argument(
        "model_pos",
        nargs="?",
        type=_parse_model,
        help="Optional model shorthand (e.g. cr, qr, fnn)",
    )
    parser.add_argument(
        "n_reservoir_pos",
        nargs="?",
        type=int,
        help="Optional reservoir size shorthand for classical models",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional positional arguments (e.g., 'reservoir 100 fnn 1' for comparison mode)",
    )

    args = parser.parse_args()

    # Resolve dataset/model/n_reservoir from flags or positionals
    dataset = args.dataset or args.dataset_pos
    if dataset is None:
        parser.error("Dataset is required; pass -d/--dataset or use positional DATASET.")

    comparison_mode = False
    comparison_reservoir_size = None
    n_fnn_hidden = None
    extra_args = args.extra_args or []
    extra_args_lower = [str(x).lower() for x in extra_args]

    model = args.model_pos or args.model
    n_hidden_layer = args.n_hidden_layer if args.n_hidden_layer is not None else args.n_reservoir_pos

    if "fnn" in extra_args_lower:
        comparison_mode = True
        try:
            fnn_idx = extra_args_lower.index("fnn")
            n_fnn_hidden = int(extra_args[fnn_idx + 1])
        except (ValueError, IndexError):
            parser.error("Invalid syntax. Use: DATASET reservoir [N_RES] fnn [H]")

        if "reservoir" in extra_args_lower:
            try:
                res_idx = extra_args_lower.index("reservoir")
                comparison_reservoir_size = int(extra_args[res_idx + 1])
            except (ValueError, IndexError):
                parser.error("Comparison mode requires reservoir size after 'reservoir'.")
        else:
            comparison_reservoir_size = args.n_reservoir_pos or args.n_hidden_layer

        if comparison_reservoir_size is None:
            parser.error("Comparison mode requires reservoir size before 'fnn'.")

        model = "fnn_pretrained"
        n_hidden_layer = n_fnn_hidden

    model_name_lower = model.lower()
    is_fnn_model = "fnn" in model_name_lower
    requires_reservoir = "quantum" not in model_name_lower and not is_fnn_model
    if requires_reservoir and n_hidden_layer is None:
        parser.error("--n-hidden_layer is required for classical reservoir models")

    if is_fnn_model and dataset != "mnist":
        parser.error("fnn_pretrained model is currently supported only for the 'mnist' dataset")

    # For FNN models, either config or n_hidden_layer must be provided
    if is_fnn_model and args.config is None and n_hidden_layer is None:
        parser.error("fnn_pretrained model requires either --config or --n-hidden_layer (hidden layer size)")

    if dataset == "mnist":
        print(" MNIST dataset detected; classification mode will be applied automatically.")

    training_name = args.training
    if args.training == "standard" and args.preprocessing == "raw":
        training_name = "raw_standard"

    print("Multi-model ML Framework")
    print("=" * 60)
    if comparison_mode:
        print(
            f"Comparison mode: FNN hidden={n_hidden_layer} "
            f"vs Reservoir baseline N={comparison_reservoir_size}"
        )
        print("=" * 60)
    elif n_hidden_layer is not None:
        print(f"Reservoir override: {n_hidden_layer}")
        print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Training: {training_name}")
    if args.preprocessing != "default":
        print(f"Preprocessing: {args.preprocessing}")
    print("=" * 60)

    # Display experiment info
    experiment_name = (
        f"{dataset}_fnn_h{n_hidden_layer}_vs_res{comparison_reservoir_size}"
        if comparison_mode
        else f"{dataset}_{model}_{training_name}"
    )
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
            from pipelines.builders import build_fnn_config

            if comparison_mode and args.config is not None:
                parser.error("Comparison mode ignores external --config; remove it to proceed.")

            fnn_config = build_fnn_config(
                dataset_name=dataset,
                n_hidden=int(n_hidden_layer),
                reservoir_size=comparison_reservoir_size if comparison_mode else None,
                training_name="standard",
            )

            # Log architecture and trainable weight count (excluding biases)
            dims = fnn_config.model.layer_dims
            num_weights = sum(d_in * d_out for d_in, d_out in zip(dims[:-1], dims[1:]))
            arch_str = " → ".join(str(d) for d in dims)
            print(
                "  FNN config (preset: standard): "
                f"{arch_str} "
                f"| Weights: {num_weights}"
            )
            print(
                "  Training preset: "
                f"lr={fnn_config.training.learning_rate}, "
                f"batch={fnn_config.training.batch_size}, "
                f"epochs={fnn_config.training.num_epochs}"
            )

            if not comparison_mode:
                from pipelines.datasets.mnist_loader import get_mnist_dataloaders
                from pipelines.fnn_pipeline import (
                    pretrain_fnn,
                    run_fnn_fixed_feature_pipeline,
                )

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
        from pipelines.dynamic_runner import run_dynamic_experiment

        print("Running dynamic experiment...")

        comparison_payload = None
        if comparison_mode:
            comparison_payload = {
                "reservoir_size": comparison_reservoir_size,
                "n_hidden": n_hidden_layer,
            }

        runner_training_name = "standard" if comparison_mode and is_fnn_model else training_name

        result = run_dynamic_experiment(
            dataset_name=dataset,
            model_name=model,
            training_name=runner_training_name,
            show_training=args.show_training,
            backend=backend,
            n_hidden_layer_override=n_hidden_layer,
            comparison_config=comparison_payload,
        )

        print("Experiment completed successfully!")
        train_mse, test_mse, train_mae, test_mae = result
        if train_mse is not None:
            print(f"📊 Results: Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        else:
            print(f"📊 Results: Test MSE: {test_mse:.6f}")
        if train_mae is not None or test_mae is not None:
            print(f"📈 Acc: Train={train_mae or 0:.4f}, Test={test_mae or 0:.4f}")

    except Exception as exc:  # pragma: no cover - top-level error
        print(f"❌ Error running experiment: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
