#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Quantum Reservoir Computing.

Optimizes input_scale (projection & R-gate) and feedback_scale for Mackey-Glass prediction,
maximizing VPT (Valid Prediction Time).

Supports separate studies for different measurement_basis / readout combinations:

Usage:
    # Default (uses preset's measurement_basis + readout)
    uv run python benchmarks/optimize_qrc.py

    # Specific readout: ridge / poly_full / poly_square
    uv run python benchmarks/optimize_qrc.py --readout poly_full
    uv run python benchmarks/optimize_qrc.py --readout poly_square

    # Z+ZZ measurement basis with poly_full readout
    uv run python benchmarks/optimize_qrc.py --measurement-basis Z+ZZ --readout poly_full

    # Custom trial count
    uv run python benchmarks/optimize_qrc.py --n-trials 10

Visualization:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///benchmarks/optuna_qrc.db
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
    TIME_QUANTUM_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (
    PolyRidgeReadoutConfig,
    RidgeReadoutConfig,
)
from reservoir.core.identifiers import Dataset


# ---------------------------------------------------------------------------
# Readout Lookup  (ridge / poly_full / poly_square)
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
        input_scale: float,
        feedback_scale: float,
        use_reuploading: bool,
        *,
        measurement_basis: str,
        readout_config,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    
    input_scale is applied via MinMaxScalerConfig (preprocessing).
    projection is None (Step 3 skipped).
    """
    from reservoir.models.config import CustomRangeScalerConfig
    base = TIME_QUANTUM_RESERVOIR_PRESET

    # Update preprocessing (input_scale via CustomRangeScaler)
    new_preprocess = CustomRangeScalerConfig(
        input_scale=input_scale, 
        centering=getattr(base.preprocess, "centering")
    )

    # Update model (feedback_scale, measurement_basis)
    new_model = dataclasses.replace(
        base.model,
        feedback_scale=feedback_scale,
        measurement_basis=measurement_basis,
        use_reuploading=use_reuploading,
    )

    # Construct final config (projection=None, no Step 3)
    return dataclasses.replace(
        base,
        preprocess=new_preprocess,
        model=new_model,
        readout=readout_config,
    )


def make_objective(measurement_basis: str, readout_config):
    """Factory that returns an Optuna objective closed over the study variant."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Searches for optimal QRC parameters to maximize VPT.
        """
        # === 1. Suggest Parameters ===

        # ======================== Preprocessing =============================
        # input_scale = trial.suggest_float("input_scale", 3, 4.5)
        input_scale = trial.suggest_float("input_scale", 0.5, 5)

        # ======================== Reservoir ==================================
        feedback_scale = trial.suggest_float("feedback_scale", 0.5, 5)
        use_reuploading = trial.suggest_categorical("use_reuploading", [True])


        # === 2. Build Config ===
        config = build_config(
            input_scale,
            feedback_scale,
            use_reuploading,
            measurement_basis=measurement_basis,
            readout_config=readout_config,
        )

        # === 3. Run Pipeline ===
        try:
            results: Dict[str, Any] = run_pipeline(config, Dataset.MACKEY_GLASS)

            # === 4. Extract Metrics ===
            test_results = results.get("test", {})
            train_results = results.get("train", {})
            chaos = test_results.get("chaos_metrics", {})

            vpt_lt = test_results.get("vpt_lt", 0.0)
            best_lambda = train_results.get("best_lambda", None)
            
            # Store best_lambda
            if best_lambda is not None:
                trial.set_user_attr("best_lambda", float(best_lambda))

            # Store ALL chaos metrics
            for key in ["mse", "nmse", "nrmse", "mase", "ndei",
                        "var_ratio", "correlation", "vpt_steps", "vpt_lt", "vpt_threshold"]:
                val = chaos.get(key, None)
                if val is not None:
                    trial.set_user_attr(key, float(val))
            
            # For backward compatibility / printing convenience, ensure these locals exist
            var_ratio = chaos.get("var_ratio", 0.0)
            mse = chaos.get("mse", float('inf'))

            if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
                print(f"Trial {trial.number}: FAILED (VPT=0) "
                      f"(in={input_scale:.3f}, fb={feedback_scale:.3f}, reupload={use_reuploading})")
                trial.set_user_attr("status", "failed")
                return -1.0  # Return a negative value to indicate failure

            print(f"Trial {trial.number}: VPT={vpt_lt:.2f} LT, Var={var_ratio:.3f} "
                  f"(in={input_scale:.3f}, fb={feedback_scale:.3f}, reupload={use_reuploading})")

            trial.set_user_attr("status", "success")
            return vpt_lt

        except Exception as e:
            print(f"Trial {trial.number}: EXCEPTION (VPT=0) - {e}")
            trial.set_user_attr("status", "exception")
            trial.set_user_attr("error", str(e))
            return -1.0  # Return a negative value to indicate failure

    return objective


def derive_names(measurement_basis: str, readout_key: str, proj_type: str, n_qubits: int, scaler_type: str):
    """Derive DB filename and study name from the variant combination."""
    study_name = f"qrc_vpt_{scaler_type}_{proj_type}_q{n_qubits}_{measurement_basis}_{readout_key}"
    db_name = f"optuna_qrc_{proj_type}.db"          # one DB per projection type
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
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
    args = parser.parse_args()

    # --- Resolve variant from args or preset defaults ---
    base = TIME_QUANTUM_RESERVOIR_PRESET

    # Measurement basis
    if args.measurement_basis is not None:
        measurement_basis = args.measurement_basis
    else:
        measurement_basis = base.model.measurement_basis
    
    # Qubits
    n_qubits = base.model.n_qubits

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
    proj_type_name = type(base.projection).__name__
    proj_tag = proj_type_name.lower().replace("config", "")
    scaler_tag = "crs"

    study_name, db_name = derive_names(measurement_basis, readout_key, proj_tag, n_qubits, scaler_tag)

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
    print("Optuna QRC Hyperparameter Search")
    print("=" * 60)
    print(f"  Study            : {study_name}")
    print(f"  Storage          : {storage}")
    print(f"  Trials           : {args.trials}")
    print(f"  Measurement Basis: {measurement_basis}")
    print(f"  Readout          : {readout_key}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(measurement_basis, readout_config)
    study.optimize(objective_fn, n_trials=args.trials)

    # --- Report ---
    print("\n" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  {'Best VPT':20s}: {study.best_value:.2f} LT")
    print("=" * 60)


if __name__ == "__main__":
    main()
