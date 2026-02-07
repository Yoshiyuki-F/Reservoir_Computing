#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Quantum Reservoir Computing.

Optimizes input_scale, leak_rate, and feedback_scale for Mackey-Glass prediction,
maximizing VPT (Valid Prediction Time).

Usage:
    uv run python benchmarks/optimize_qrc.py                # Run 50 trials
    uv run python benchmarks/optimize_qrc.py --n-trials 10  # Run 10 trials

Visualization:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///benchmarks/optuna_qrc.db
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Any, Dict

import optuna



from reservoir.pipelines import run_pipeline
from reservoir.models.presets import TIME_QUANTUM_RESERVOIR_PRESET
from reservoir.models.config import RandomProjectionConfig, CoherentDriveProjectionConfig
from reservoir.core.identifiers import Dataset


def build_config(
        input_scale: float, 
        leak_rate: float, 
        feedback_scale: float,
        bias_scale: float,
        input_connectivity: float
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    
    Uses dataclasses.replace to modify the frozen preset config.
    """
    base = TIME_QUANTUM_RESERVOIR_PRESET
    
    # Update projection (input_scale, bias_scale, connectivity)
    new_projection = dataclasses.replace(
        base.projection,
        input_scale=input_scale,
        bias_scale=bias_scale,
        input_connectivity=input_connectivity
    )
    
    # Update model (leak_rate, feedback_scale)
    new_model = dataclasses.replace(
        base.model,
        leak_rate=leak_rate,
        feedback_scale=feedback_scale
    )
    
    # Construct final config
    return dataclasses.replace(
        base,
        projection=new_projection,
        model=new_model
    )


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    
    Searches for optimal QRC parameters to maximize VPT.
    """
    # === 1. Suggest Parameters ===

    #========================Projection===============================================================

    # input_scale: Signal strength (Amplitude)
    input_scale = trial.suggest_float("input_scale", 0.001, 1.5, log=True)
    
    # bias_scale: Operating point variety (Quality/Non-linearity)
    bias_scale = trial.suggest_float("bias_scale", 0.0, 2.0)
    
    # input_connectivity: Sparsity (Information mixing)
    input_connectivity = 0.8

    #========================Reservoir===============================================================
    # Reservoir dynamics
    leak_rate = trial.suggest_float("leak_rate", 0.0, 1.0)
    feedback_scale = trial.suggest_float("feedback_scale", 0.0, 2.0)



    # === 2. Build Config ===
    config = build_config(
        input_scale, 
        leak_rate, 
        feedback_scale,
        bias_scale,
        input_connectivity
    )
    
    # === 3. Run Pipeline ===
    try:
        results: Dict[str, Any] = run_pipeline(config, Dataset.MACKEY_GLASS)
        
        # === 4. Extract Metrics ===
        test_results = results.get("test", {})
        vpt_lt = test_results.get("vpt_lt", 0.0)
        var_ratio = test_results.get("var_ratio", 0.0)
        ndei = test_results.get("ndei", float('inf'))
        mse = test_results.get("mse", float('inf'))
        
        # Save additional metrics to DB
        trial.set_user_attr("var_ratio", var_ratio)
        trial.set_user_attr("ndei", ndei)
        trial.set_user_attr("mse", mse)
        
        # Guard: NaN or invalid VPT - return 0 so Optuna learns to avoid this region
        if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
            print(f"Trial {trial.number}: FAILED (VPT=0) "
                  f"(in={input_scale:.3f}, bias={bias_scale:.3f}, conn={input_connectivity:.2f}, "
                  f"leak={leak_rate:.3f}, fb={feedback_scale:.3f})")
            trial.set_user_attr("status", "failed")
            return 0.0  # Bad score - Optuna will learn to avoid this region
        
        print(f"Trial {trial.number}: VPT={vpt_lt:.2f} LT, Var={var_ratio:.3f} "
              f"(in={input_scale:.3f}, bias={bias_scale:.3f}, conn={input_connectivity:.2f}, "
              f"leak={leak_rate:.3f}, fb={feedback_scale:.3f})")
        
        trial.set_user_attr("status", "success")
        return vpt_lt  # Maximize VPT directly
        
    except Exception as e:
        print(f"Trial {trial.number}: EXCEPTION (VPT=0) - {e}")
        trial.set_user_attr("status", "exception")
        trial.set_user_attr("error", str(e))
        return 0.0  # Bad score - Optuna will learn to avoid this region


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=50, 
                        help="Number of optimization trials (default: 50)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Optuna study name (default: derived from projection type)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (default: benchmarks/optuna_qrc.db)")
    args = parser.parse_args()
    
    # Determine DB name and Study name based on projection type
    proj_config = TIME_QUANTUM_RESERVOIR_PRESET.projection
    config_type_name = type(proj_config).__name__
    
    if config_type_name == "RandomProjectionConfig":
        db_name = "optuna_qrc_random_projection.db"
        default_study_name = "qrc_mackey_glass_vpt_random_projection"
    elif config_type_name == "CoherentDriveProjectionConfig":
        db_name = "optuna_qrc_coherent_drive.db"
        default_study_name = "qrc_mackey_glass_vpt_coherent_drive"
    else:
        db_name = "optuna_qrc.db"
        default_study_name = "qrc_mackey_glass_vpt"
    
    # Default storage in benchmarks folder
    if args.storage is None:
        db_path = Path(__file__).parent / db_name
        args.storage = f"sqlite:///{db_path}"
    
    # Use default study name if not provided
    if args.study_name is None:
        args.study_name = default_study_name
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize VPT directly
        load_if_exists=True
    )
    
    print("=" * 60)
    print("Optuna QRC Hyperparameter Search")
    print("=" * 60)
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.n_trials}")
    print("=" * 60)
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Report results
    print("\n" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    print(f"input_scale    : {study.best_params['input_scale']:.4f}")
    print(f"leak_rate      : {study.best_params['leak_rate']:.4f}")
    print(f"feedback_scale : {study.best_params['feedback_scale']:.4f}")
    print(f"Best VPT       : {study.best_value:.2f} LT")
    print("=" * 60)


if __name__ == "__main__":
    main()
