#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Classical Reservoir Computing.

Optimizes:
- Projection input_scale
- Reservoir spectral_radius
- Reservoir leak_rate
- Reservoir rc_connectivity

Target Preset: TIME_CLASSICAL_RESERVOIR_PRESET

Usage:
    uv run python benchmarks/optimize_rc.py
    uv run python benchmarks/optimize_rc.py --n-trials 100
"""

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna

from reservoir.pipelines import run_pipeline
from reservoir.models.presets import (
    TIME_CLASSICAL_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (
    PolyRidgeReadoutConfig,
    RandomProjectionConfig,
)
from reservoir.core.identifiers import Dataset


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
    base = TIME_CLASSICAL_RESERVOIR_PRESET

    # Update Projection (input_scale)
    # Ensure base projection is RandomProjectionConfig
    if isinstance(base.projection, RandomProjectionConfig):
        new_proj = dataclasses.replace(
            base.projection, 
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale
        )
    else:
        # Fallback if changed, but expected to be RP
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


def make_objective(readout_config):
    """Factory that returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        """
        # === 1. Suggest Parameters ===
        
        # Projection
        input_scale = trial.suggest_float("input_scale", 0.5, 4.0, log=True)
        input_connectivity = trial.suggest_float("input_connectivity", 0.1, 1.0, log=True)
        bias_scale = trial.suggest_float("bias_scale", 0.3, 1.8)
        
        # Reservoir
        spectral_radius = 1.616 # trial.suggest_float("spectral_radius", 0.6, 1.8)
        leak_rate = trial.suggest_float("leak_rate", 0.0, 0.8)
        rc_connectivity = 0.677 # trial.suggest_float("rc_connectivity", 0.01, 1, log=True)

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
            results: Dict[str, Any] = run_pipeline(config, Dataset.MACKEY_GLASS)

            # === 4. Extract Metrics ===
            test_results = results.get("test", {})
            vpt_lt = test_results.get("vpt_lt", 0.0)
            var_ratio = test_results.get("var_ratio", 0.0)
            mse = test_results.get("mse", float('inf'))

            trial.set_user_attr("var_ratio", var_ratio)
            trial.set_user_attr("mse", mse)

            if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
                print(f"Trial {trial.number}: FAILED (VPT=0)")
                trial.set_user_attr("status", "failed")
                return 0.0

            print(f"Trial {trial.number}: VPT={vpt_lt:.2f} LT, MSE={mse:.5f} "
                  f"(in={input_scale:.2f}, ic={input_connectivity:.2f}, bs={bias_scale:.2f}, sr={spectral_radius:.2f}, lr={leak_rate:.2f}, rc={rc_connectivity:.2f})")

            trial.set_user_attr("status", "success")
            return vpt_lt

        except Exception as e:
            print(f"Trial {trial.number}: EXCEPTION - {e}")
            trial.set_user_attr("status", "exception")
            return 0.0

    return objective


def derive_names(readout_key: str):
    """Derive DB filename and study name from config components."""
    base = TIME_CLASSICAL_RESERVOIR_PRESET
    
    # Preprocess
    preprocess_name = type(base.preprocess).__name__.replace("Config", "")
    if "Standard" in preprocess_name:
        prep_tag = "Standard"
    elif "MinMax" in preprocess_name:
        prep_tag = "MinMaX"
    else:
        prep_tag = preprocess_name

    # Projection
    proj = base.projection
    if isinstance(proj, RandomProjectionConfig):
        proj_tag = f"Random{proj.n_units}"
    else:
        proj_tag = type(proj).__name__.replace("Config", "")

    # Study Name: optimize_rc_{Preprocess}_{Projection}_{Readout}
    study_name = f"optimize_rc_{prep_tag}_{proj_tag}_{readout_key}"
    db_name = "optimize_rc.db" # Shared DB for RC optimization
    
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna RC Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--readout", type=str, default=None,
                        choices=list(READOUT_MAP.keys()),
                        help="Readout type (default: ridge)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Override Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Override Optuna storage URL")
    args = parser.parse_args()

    # --- Readout ---
    if args.readout is not None:
        readout_key = args.readout
        readout_config = READOUT_MAP[readout_key]
    else:
        readout_config = TIME_CLASSICAL_RESERVOIR_PRESET.readout
        readout_key = "ridge"
        if isinstance(readout_config, PolyRidgeReadoutConfig):
             readout_key = "poly_full" if readout_config.mode == "full" else "poly_square"

    # --- Derive Names ---
    study_name, db_name = derive_names(readout_key)

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
        direction="maximize",
        load_if_exists=True
    )

    print("=" * 60)
    print("Optuna RC Hyperparameter Search")
    print("=" * 60)
    print(f"  Study:   {study_name}")
    print(f"  Storage: {storage}")
    print(f"  Trials:  {args.n_trials}")
    print(f"  Readout: {readout_key}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(readout_config)
    study.optimize(objective_fn, n_trials=args.n_trials)

    # --- Report ---
    print("\n" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.4f}")
    if study.best_value is not None:
        print(f"  {'Best VPT':20s}: {study.best_value:.2f} LT")
    print("=" * 60)


if __name__ == "__main__":
    main()
