"""
Extract and analyze Optuna results from a SQLite database using Pandas.
Usage:
uv run python benchmarks/extract_optuna_results.py --db benchmarks/optuna_qrc_nonetype.db --study qrc_lorenz_...
"""

import argparse
import sqlite3
from pathlib import Path

import optuna
import pandas as pd


def get_available_studies(db_path: str) -> list[str]:
    """Helper function to list all study names in the SQLite database."""
    # Remove the 'sqlite:///' prefix if present for raw sqlite connection
    raw_path = db_path.replace("sqlite:///", "")
    if not Path(raw_path).exists():
        return []
    
    try:
        conn = sqlite3.connect(raw_path)
        cursor = conn.cursor()
        cursor.execute("SELECT study_name FROM studies")
        studies = [row[0] for row in cursor.fetchall()]
        conn.close()
        return studies
    except sqlite3.Error:
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract Optuna study results to Pandas/CSV")
    parser.add_argument("--db", type=str, default="benchmarks/optuna_qrc_nonetype.db",
                        help="Path to the SQLite database file (e.g., benchmarks/optuna_qrc_nonetype.db)")
    parser.add_argument("--study", type=str, default=None,
                        help="Name of the study to extract. If omitted, lists available studies.")
    parser.add_argument("--out", type=str, default="optuna_results.csv",
                        help="Output CSV file name (default: optuna_results.csv)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top trials to print to console (default: 10)")

    args = parser.parse_args()

    # Format the storage URL
    db_path = args.db
    if not db_path.startswith("sqlite:///"):
        # Make it absolute to avoid path issues
        abs_db_path = Path(db_path).resolve()
        storage = f"sqlite:///{abs_db_path}"
    else:
        storage = db_path

    # If study name is not provided, list them and exit
    if args.study is None:
        print(f"Checking database: {storage}")
        studies = get_available_studies(db_path)
        if not studies:
            print("No studies found or database does not exist.")
            return
        
        print("\nAvailable studies in this database:")
        for s in studies:
            print(f"  - {s}")
        print("\nPlease run again providing a study name using --study <study_name>")
        return

    study_name = args.study

    try:
        # Load the study
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"Error: Study '{study_name}' not found in {storage}.")
        return

    # Extract all trials as a Pandas DataFrame
    print(f"\nExtracting results for study: {study_name}")
    df = study.trials_dataframe()

    if df.empty:
        print("The study contains no trials.")
        return

    # Filter only COMPLETED trials
    if 'state' in df.columns:
        completed_df = df[df['state'] == 'COMPLETE'].copy()
    else:
        completed_df = df.copy()

    # Sort by the objective value (VPT). Since we maximized, sort descending.
    # Note: Optuna's default column for the objective is 'value'
    if 'value' in completed_df.columns:
        completed_df.sort_values(by='value', ascending=False, inplace=True)
        
        print(f"\nTop {args.top} Trials (Sorted by Objective Value / VPT):")
        
        # Select columns to display clearly
        # Get parameter columns (they start with 'params_')
        param_cols = [col for col in completed_df.columns if col.startswith('params_')]
        display_cols = ['number', 'value'] + param_cols
        
        # Display the top N rows
        print(completed_df[display_cols].head(args.top).to_string(index=False))

    # Save the full dataframe to CSV
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"\nFull results (including failed trials and all metadata) saved to: {out_path.absolute()}")


if __name__ == "__main__":
    main()