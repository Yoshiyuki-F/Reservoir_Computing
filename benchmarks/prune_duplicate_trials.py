import optuna
import sys
import os
import json

def prune_duplicate_trials(db_path, dry_run=True):
    url = f"sqlite:///{db_path}"
    print(f"Connecting to {url}...")
    
    try:
        storage = optuna.storages.RDBStorage(url=url)
        summaries = optuna.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"Failed to connect/read DB: {e}")
        return

    for summary in summaries:
        print(f"\nScanning study: {summary.study_name}")
        try:
            study = optuna.load_study(study_name=summary.study_name, storage=storage)
        except Exception as e:
            print(f"  Failed to load study: {e}")
            continue

        # Group trials by parameters (using hash of sorted params)
        trials_by_params = {}
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            # Create a unique key for parameters
            # Sort keys to ensure consistent order
            params_key = json.dumps(trial.params, sort_keys=True)
            
            if params_key not in trials_by_params:
                trials_by_params[params_key] = []
            trials_by_params[params_key].append(trial)

        trials_to_delete = []

        for params_key, trials in trials_by_params.items():
            if len(trials) > 1:
                # Sort by trial ID (latest ID is most recent)
                trials.sort(key=lambda t: t.number)
                
                # Keep the last one, delete the rest
                keep_trial = trials[-1]
                del_trials = trials[:-1]
                
                print(f"  Duplicate Params found ({len(trials)} trials):")
                print(f"    Keep: ID#{keep_trial.number} (Value: {keep_trial.value})")
                for t in del_trials:
                    print(f"    Delete: ID#{t.number} (Value: {t.value})")
                    trials_to_delete.append(t.number)

        if not trials_to_delete:
            print("  No duplicates found to prune.")
            continue

        print(f"  Found {len(trials_to_delete)} trials to prune in this study.")
        
        if dry_run:
            print("  [DRY RUN] No changes made. Run without --dry-run to execute.")
        else:
            print("  Deleting trials...")
            # Use storage API to delete trials
            # RDBStorage.delete_trial is not exposed in high-level API usually, check version
            # If not available, we can set state to PRUNED or FAIL to hide them from best_trial
            # But users usually want them GONE.
            # Optuna's delete_study deletes the whole study. duplicated trials are hard to delete cleanly via public API.
            # However, storage.delete_trial(trial_id) exists in RDBStorage.
            
            for trial_id in trials_to_delete:
                try:
                    storage.delete_trial(trial_id)
                    print(f"    Deleted ID#{trial_id}")
                except Exception as e:
                    print(f"    Failed to delete ID#{trial_id}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmarks/prune_duplicate_trials.py <path_to_db_file> [--no-dry-run]")
        sys.exit(1)
        
    db_file = sys.argv[1]
    dry_run = "--no-dry-run" not in sys.argv
    
    if not os.path.exists(db_file):
        # check cwd
        if os.path.exists(os.path.join(os.getcwd(), db_file)):
            db_file = os.path.join(os.getcwd(), db_file)
        else:
             print(f"File not found: {db_file}")
             sys.exit(1)
        
    prune_duplicate_trials(os.path.abspath(db_file), dry_run=dry_run)
