#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Classical Reservoir Computing on MNIST.

Optimizes:
- Projection input_scale
- Reservoir spectral_radius
- Reservoir leak_rate
- Reservoir rc_connectivity
- Readout Ridge Alpha (if Ridge)

Target Preset: CLASSICAL_RESERVOIR_PRESET (Classification)

Usage:
    uv run python benchmarks/optimize_rc_mnist.py
    uv run python benchmarks/optimize_rc_mnist.py --n-trials 100
Visualization:
    uv run optuna-dashboard  sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optimize_rc.db
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import optuna
import jax

jax.config.update("jax_enable_x64", True)

from reservoir.pipelines import run_pipeline  # noqa: E402
from reservoir.utils import check_gpu_available  # noqa: E402
from reservoir.models.presets import (  # noqa: E402
    CLASSICAL_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (  # noqa: E402
    MinMaxScalerConfig,
    PolyRidgeReadoutConfig,
    RandomProjectionConfig,
)
from reservoir.core.identifiers import Dataset  # noqa: E402


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


def build_config(
        input_scale: float,
        input_connectivity: float,
        bias_scale: float,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        readout_config,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    """
    base = CLASSICAL_RESERVOIR_PRESET




    # Ensure base projection is RandomProjectionConfig
    if isinstance(base.projection, RandomProjectionConfig):
        new_proj = dataclasses.replace(
            base.projection, 
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale
        )
    else:
        # Fallback
        new_proj = base.projection

    # Update Reservoir (spectral_radius, leak_rate, connectivity)
    new_model = dataclasses.replace(
        base.model,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        rc_connectivity=rc_connectivity,
    )

    # Construct final config
    return dataclasses.replace(
        base,
        projection=new_proj,
        model=new_model,
        readout=readout_config,
    )



def make_objective(readout_config, dataset_enum: Dataset):
    """Factory that returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        """

        
        # === 1. Suggest Parameters ===

        # Projection
        input_scale = trial.suggest_float("input_scale", 0.05, 5.0, log=True)
        # input_scale = trial.suggest_float("input_scale", 3.1537235606199965, 3.1537235606199965)


        input_connectivity = trial.suggest_float("input_connectivity", 0.01, 1.0)
        # input_connectivity = trial.suggest_float("input_connectivity", 0.7789498820486052, 0.7789498820486052)

        bias_scale = trial.suggest_float("bias_scale", 0.0, 2.0)
        # bias_scale = trial.suggest_float("bias_scale", 0.6664704836440828, 0.6664704836440828)


        # Reservoir
        spectral_radius = trial.suggest_float("spectral_radius", 0.1, 2.0)

        leak_rate = trial.suggest_float("leak_rate", 0.1, 1.0)

        rc_connectivity = trial.suggest_float("rc_connectivity", 0.1, 1.0)
        # rc_connectivity = trial.suggest_float("rc_connectivity",  0.6213282741686085,  0.6213282741686085)

        # Readout Alpha (Logic to override preset lambda candidates if we want to optimize single alpha)
        # For now, we rely on RidgeReadoutConfig's list of candidates which are scanned automatically via CV/LOO.
        # But we could optimize other readout params here if needed.

        # === 2. Build Config ===
        config = build_config(
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            rc_connectivity=rc_connectivity,
            readout_config=readout_config,
        )

        # === 3. Run Pipeline ===
        try:
            from typing import Any
            results: dict[str, Any] = run_pipeline(config, dataset_enum)

            # === 4. Extract & Store ALL Metrics ===
            test_results = results.get("test", {})
            train_results = results.get("train", {})
            
            # Accuracy metric keys might vary, check reporter logic
            # Typically 'accuracy' or 'test_score'
            # In reporter.py: results["test"] = {metric_name: test_score, ...}
            # metric_name comes from ModelStack.metric
            
            # For classification, look for 'accuracy'
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

            if accuracy <= 0.11: # Random guess threshold roughly
                 trial.set_user_attr("status", "low_acc")
                 # We still return valid accuracy, but mark it.

            print(f"Trial {trial.number}: Acc={accuracy:.4f}, λ={best_lambda} "
                  f"(in={input_scale:.2f}, ic={input_connectivity:.2f}, bs={bias_scale:.2f}, sr={spectral_radius:.2f}, lr={leak_rate:.2f}, rc={rc_connectivity:.2f})")

            trial.set_user_attr("status", "success")
            return accuracy

        except ValueError as e:
            if "diverged" in str(e).lower():
                print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
                trial.set_user_attr("status", "diverged")
                return 0.0
            else:
                print(f"Trial {trial.number}: EXCEPTION (ValueError) - {e}")
                trial.set_user_attr("status", "exception")
                return 0.0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Trial {trial.number}: EXCEPTION - {e}")
            trial.set_user_attr("status", "exception")
            return 0.0

    return objective


def derive_names(readout_key: str, dataset_name: str):
    """Derive DB filename and study name from config components."""
    base = CLASSICAL_RESERVOIR_PRESET
    
    # Preprocess
    preprocess_name = type(base.preprocess).__name__.replace("Config", "")
    if "Standard" in preprocess_name:
        prep_tag = "Standard"
    elif "MinMax" in preprocess_name:
        prep_tag = "MinMaX"
    else:
        prep_tag = preprocess_name
    
    # Custom MinMax tag
    if isinstance(base.preprocess, MinMaxScalerConfig):
        prep_tag = f"Min{int(base.preprocess.feature_min)}Max{int(base.preprocess.feature_max)}"

    # Projection
    proj = base.projection
    if isinstance(proj, RandomProjectionConfig):
        proj_tag = f"Random{proj.n_units}"
    else:
        proj_tag = type(proj).__name__.replace("Config", "")

    # Study Name: optimize_rc_{Dataset}_{Preprocess}_{Projection}_{Readout}
    study_name = f"optimize_rc_{dataset_name.upper()}_{prep_tag}_{proj_tag}_{readout_key}"
    db_name = "optimize_rc.db" # Shared DB for RC optimization
    
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna RC Hyperparameter Search (MNIST)")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist"],
                        help="Dataset to optimize on (default: mnist)")
    parser.add_argument("--readout", type=str, default=None,
                        choices=list(READOUT_MAP.keys()),
                        help="Readout type (default: ridge)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Override Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Override Optuna storage URL")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except Exception as exc:
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    # --- Dataset ---
    dataset_name = args.dataset
    if dataset_name == "mnist":
        dataset_enum = Dataset.MNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Readout ---
    if args.readout is not None:
        readout_key = args.readout
        readout_config = READOUT_MAP[readout_key]
    else:
        readout_config = CLASSICAL_RESERVOIR_PRESET.readout
        readout_key = "ridge"
        if isinstance(readout_config, PolyRidgeReadoutConfig):
             readout_key = "poly_full" if readout_config.mode == "full" else "poly_square"

    # --- Derive Names ---
    study_name, db_name = derive_names(readout_key, dataset_name)

    if args.study_name is not None:
        study_name = args.study_name
    
    if args.storage is None:
        db_path = Path(__file__).parent / db_name
        storage = f"sqlite:///{db_path}"
    else:
        storage = args.storage

    # --- Create Study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize", # Maximize Accuracy
        load_if_exists=True
    )

    print("=" * 60)
    print("Optuna RC Hyperparameter Search (MNIST)")
    print("=" * 60)
    print(f"  Study:   {study_name}")
    print(f"  Storage: {storage}")
    print(f"  Dataset: {dataset_name} ({dataset_enum})")
    print(f"  Trials:  {args.n_trials}")
    print(f"  Readout: {readout_key}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(readout_config, dataset_enum)
    study.optimize(objective_fn, n_trials=args.n_trials)

    # --- Report ---
    print("\n" + "=" * 60)
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
