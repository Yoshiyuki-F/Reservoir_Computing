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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reservoir.pipelines import run_pipeline
from reservoir.models.presets import TIME_QUANTUM_RESERVOIR_PRESET
from reservoir.core.identifiers import Dataset


def build_config(input_scale: float, leak_rate: float, feedback_scale: float):
    """
    Build a PipelineConfig with dynamically updated parameters.
    
    Uses dataclasses.replace to modify the frozen preset config.
    """
    base = TIME_QUANTUM_RESERVOIR_PRESET
    
    # Update projection (input_scale)
    new_projection = dataclasses.replace(
        base.projection,
        input_scale=input_scale
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
    Returns negative VPT (since Optuna minimizes by default).
    """
    # === 1. Suggest Parameters ===
    input_scale = trial.suggest_float("input_scale", 0.1, 5.0, log=True)
    leak_rate = trial.suggest_float("leak_rate", 0.01, 1.0)
    feedback_scale = trial.suggest_float("feedback_scale", 0.0, 2.0)
    
    # === 2. Build Config ===
    config = build_config(input_scale, leak_rate, feedback_scale)
    
    # === 3. Run Pipeline ===
    try:
        results: Dict[str, Any] = run_pipeline(config, Dataset.MACKEY_GLASS)
        
        # === 4. Extract VPT ===
        closed_loop_metrics = results.get("closed_loop_metrics", {})
        vpt_lt = closed_loop_metrics.get("vpt_lt", 0.0)
        
        # Guard: NaN or invalid VPT
        if math.isnan(vpt_lt) or vpt_lt <= 0:
            print(f"Trial {trial.number}: FAILED (VPT={vpt_lt})")
            return float('inf')
        
        print(f"Trial {trial.number}: VPT={vpt_lt:.2f} LT "
              f"(in={input_scale:.3f}, leak={leak_rate:.3f}, fb={feedback_scale:.3f})")
        
        # Return negative VPT (minimize → maximize VPT)
        return -vpt_lt
        
    except Exception as e:
        print(f"Trial {trial.number}: EXCEPTION - {e}")
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=50, 
                        help="Number of optimization trials (default: 50)")
    parser.add_argument("--study-name", type=str, default="qrc_mackey_glass_vpt",
                        help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (default: benchmarks/optuna_qrc.db)")
    args = parser.parse_args()
    
    # Default storage in benchmarks folder
    if args.storage is None:
        db_path = Path(__file__).parent / "optuna_qrc.db"
        args.storage = f"sqlite:///{db_path}"
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",  # Minimizing -VPT = Maximizing VPT
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
    print(f"Best VPT       : {-study.best_value:.2f} LT")
    print("=" * 60)


if __name__ == "__main__":
    main()
