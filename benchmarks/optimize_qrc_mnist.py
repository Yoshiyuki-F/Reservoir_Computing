#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Quantum Reservoir Computing on MNIST.

Optimizes:
- Projection: BoundedAffinePCA (scale, relative_shift)
- Quantum Reservoir (n_layers, feedback_scale, leak_rate)
- Readout (Ridge/Poly)

Target Preset: QUANTUM_RESERVOIR_PRESET (Classification)

Usage:
uv run python benchmarks/optimize_qrc_mnist.py
uv run python benchmarks/optimize_qrc_mnist.py --trials 100
Visualization:
uv run optuna-dashboard sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optimize_qrc_mnist.db
"""

import os
# Force 64-bit precision before ANY other imports
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import dataclasses
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import optuna

from reservoir.utils import check_gpu_available
from reservoir.pipelines import run_pipeline
from reservoir.pipelines.strategies import DivergenceError
from reservoir.models.presets import (
    QUANTUM_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (
    BoundedAffinePCAConfig,
    PolyRidgeReadoutConfig,
)
from reservoir.data.identifiers import Dataset

# ---------------------------------------------------------------------------
# Readout Map
# ---------------------------------------------------------------------------
_POLY_LAMBDAS = tuple(np.logspace(-12, 3, 30).tolist())

READOUT_MAP = {
    "ridge":       DEFAULT_RIDGE_READOUT,
    "poly_full":   PolyRidgeReadoutConfig(
        use_intercept=True, lambda_candidates=_POLY_LAMBDAS, degree=2, mode="full",
    ),
    "poly_square": PolyRidgeReadoutConfig(
        use_intercept=True, lambda_candidates=_POLY_LAMBDAS, degree=2, mode="square_only",
    ),
}

VALID_BASES = ("Z", "ZZ", "Z+ZZ")


def build_config(
        scale: float,
        relative_shift: float,
        n_layers: int,
        feedback_scale: float,
        leak_rate: float,
        use_reuploading: bool,
        measurement_basis: str,
        readout_config,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    Preprocessing from preset (StandardScaler).
    Projection: BoundedAffinePCA (scale, relative_shift).
    """
    base = QUANTUM_RESERVOIR_PRESET

    # Update Projection (BoundedAffinePCA with tuned scale/relative_shift)
    base_proj = base.projection
    n_units = int(getattr(base_proj, 'n_units', 6))
    new_proj = BoundedAffinePCAConfig(
        n_units=n_units,
        scale=scale,
        relative_shift=relative_shift,
    )

    # Update Reservoir (n_layers, feedback_scale, leak_rate, measurement_basis)
    new_model = dataclasses.replace(
        base.model,
        n_layers=n_layers,
        feedback_scale=feedback_scale,
        leak_rate=leak_rate,
        measurement_basis=measurement_basis,
        use_reuploading=use_reuploading,
    )

    # Construct final config (preprocess from preset)
    return dataclasses.replace(
        base,
        projection=new_proj,
        model=new_model,
        readout=readout_config,
    )


def make_objective(measurement_basis: str, readout_config, use_reuploading: bool, dataset_enum: Dataset):
    """Factory that returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Searches for optimal QRC parameters to maximize Accuracy.
        """

        # === 1. Suggest Parameters ===

        # Projection (BoundedAffinePCA — controls QC input range)
        scale = trial.suggest_float("scale", 0.00000001, 1.0)
        relative_shift = trial.suggest_float("relative_shift", -1.0, 1.0)

        # Reservoir
        n_layers = trial.suggest_int("n_layers", 1, 1)
        feedback_scale = trial.suggest_float("feedback_scale", 0.0, 3.5)
        leak_rate = trial.suggest_float("leak_rate", 0.0, 1.0)

        # === 2. Build Config ===
        config = build_config(
            scale=scale,
            relative_shift=relative_shift,
            n_layers=n_layers,
            feedback_scale=feedback_scale,
            leak_rate=leak_rate,
            use_reuploading=use_reuploading,
            measurement_basis=measurement_basis,
            readout_config=readout_config,
        )

        # === 3. Run Pipeline ===
        try:
            from typing import Any
            results: dict[str, Any] = run_pipeline(config, dataset_enum)

            # === 4. Extract Metrics ===
            test_results = results.get("test", {})
            train_results = results.get("train", {})

            accuracy = test_results.get("accuracy", 0.0)
            best_lambda = train_results.get("best_lambda", None)

            # Store best_lambda
            if best_lambda is not None:
                trial.set_user_attr("best_lambda", float(best_lambda))

            # Store extra metrics
            for key in ["accuracy", "precision", "recall", "f1"]:
                val = test_results.get(key, None)
                if val is not None:
                    trial.set_user_attr(key, float(val))

            if accuracy <= 0.11:  # Random guess threshold roughly
                trial.set_user_attr("status", "low_acc")

            print(f"Trial {trial.number}: Acc={accuracy:.4f}, λ={best_lambda} "
                  f"(scale={scale:.4f}, shift={relative_shift:.4f}, "
                  f"L={n_layers}, fb={feedback_scale:.4f}, lr={leak_rate:.4f})")

            trial.set_user_attr("status", "success")

        except DivergenceError as e:
            # Retrieve stats attached to DivergenceError (from strategies.py)
            if hasattr(e, "stats") and isinstance(e.stats, dict):
                print(f"    [Divergence Stats] {e.stats}")
                for key, val in e.stats.items():
                    trial.set_user_attr(key, float(val))
            
            print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
            trial.set_user_attr("status", "diverged")
            trial.set_user_attr("error", str(e))
            return 0.0

        except ValueError as e:
            if "nan detected" in str(e).lower() or "diverged" in str(e).lower():
                print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
                trial.set_user_attr("status", "diverged")
                trial.set_user_attr("error", str(e))
                return 0.0
            else:
                print(f"Trial {trial.number}: EXCEPTION (ValueError) - {e}")
                trial.set_user_attr("status", "exception")
                trial.set_user_attr("error", str(e))
                return 0.0
        except RuntimeError as e:
            import traceback
            traceback.print_exc()
            print(f"Trial {trial.number}: EXCEPTION - {e}")
            trial.set_user_attr("status", "exception")
            trial.set_user_attr("error", str(e))
            return 0.0
        else:
            return accuracy

    return objective


def derive_names(dataset_name: str, measurement_basis: str, readout_key: str, n_qubits: int, use_reuploading: bool):
    """Derive DB filename and study name from the variant combination."""
    reupload_str = "reupTrue" if use_reuploading else "reupFalse"
    
    base = QUANTUM_RESERVOIR_PRESET
    # Projection
    proj = base.projection
    n_units = int(getattr(proj, 'n_units', 0))
    proj_tag = f"BAPCA{n_units}"

    # Preprocessing from preset
    scaler_tag = type(base.preprocess).__name__.replace("Config", "").lower()

    study_name = f"qrc_{dataset_name}_{scaler_tag}_{proj_tag}_q{n_qubits}_{measurement_basis}_{readout_key}_{reupload_str}_kai4"
    db_name = "optimize_qrc_mnist.db"
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search (MNIST)")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist"],
                        help="Dataset to optimize on (default: mnist)")
    parser.add_argument("--measurement-basis", type=str, default=None,
                        choices=list(VALID_BASES),
                        help="Measurement basis (default: from preset)")
    parser.add_argument("--readout", type=str, default=None,
                        choices=list(READOUT_MAP.keys()),
                        help="Readout type (default: from preset)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Override Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Override Optuna storage URL")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except RuntimeError as exc:
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    # --- Dataset ---
    dataset_name = args.dataset
    if dataset_name == "mnist":
        dataset_enum = Dataset.MNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Resolve variant from args or preset defaults ---
    base = QUANTUM_RESERVOIR_PRESET

    # Measurement basis
    if args.measurement_basis is not None:
        measurement_basis = args.measurement_basis
    else:
        measurement_basis = base.model.measurement_basis
    
    # Qubits
    n_qubits = base.model.n_qubits

    # Re-uploading
    use_reuploading = base.model.use_reuploading

    # Readout
    if args.readout is not None:
        readout_key = args.readout
        readout_config = READOUT_MAP[readout_key]
    else:
        readout_config = base.readout
        # Reverse-lookup key from preset
        if isinstance(readout_config, PolyRidgeReadoutConfig):
            readout_key = "poly_full" if readout_config.mode == "full" else "poly_square"
        else:
            readout_key = "ridge"

    # --- Derive study / DB names ---
    study_name, db_name = derive_names(dataset_name, measurement_basis, readout_key, n_qubits, use_reuploading)

    if args.study_name is not None:
        study_name = args.study_name
    
    if args.storage is None:
        db_path = Path(__file__).parent / db_name
        storage = f"sqlite:///{db_path}"
    else:
        storage = args.storage

    # --- Create or load study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )

    print("=" * 60)
    print("Optuna QRC Hyperparameter Search (MNIST)")
    print("=" * 60)
    print(f"  Study            : {study_name}")
    print(f"  Storage          : {storage}")
    print(f"  Dataset          : {dataset_name} ({dataset_enum})")
    print(f"  Trials           : {args.trials}")
    print(f"  Measurement Basis: {measurement_basis}")
    print(f"  Readout          : {readout_key}")
    print(f"  Re-uploading     : {use_reuploading}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(measurement_basis, readout_config, use_reuploading, dataset_enum)
    study.optimize(objective_fn, n_trials=args.trials)
    
    # --- Report ---
    print("" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.6f}")
    if study.best_value is not None:
        print(f"  {'Best Accuracy':20s}: {study.best_value:.4f}")
    print("-" * 60)
    print("  [Stored Metrics]")
    for k, v in study.best_trial.user_attrs.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
