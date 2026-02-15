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
        import json
        
        # Consistent normalization (same as requeue_trials.py)
        def normalize_params(params):
            norm = {}
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    norm[k] = round(float(v), 8) # Force float and round
                else:
                    norm[k] = str(v)
            return json.dumps(norm, sort_keys=True)

        for trial in study.trials:
            # Check all states. Duplicates might span any state.
            
            # Use normalized key for grouping (fuzzy match for float precision)
            params_key = normalize_params(trial.params)
            
            if params_key not in trials_by_params:
                trials_by_params[params_key] = []
            trials_by_params[params_key].append(trial)

        trials_to_delete = []

        for params_key, trials in trials_by_params.items():
            if len(trials) > 1:
                # Priority:
                # 1. State must be WAITING (keep for re-run)
                # 2. State must be COMPLETE
                # 3. Prefer Latest (ID)
                
                def trial_priority(t):
                    is_waiting = 1 if t.state == optuna.trial.TrialState.WAITING else 0
                    is_complete = 1 if t.state == optuna.trial.TrialState.COMPLETE else 0
                    return (is_waiting, is_complete, t.number)
                
                trials.sort(key=trial_priority)
                
                # Keep the last one (highest priority), delete the rest
                keep_trial = trials[-1]
                del_trials = trials[:-1]
                
                print(f"  Duplicate Params found ({len(trials)} trials):")
                print(f"    Keep: ID#{keep_trial.number} (State: {keep_trial.state.name}, Value: {keep_trial.value})")
                for t in del_trials:
                    print(f"    Delete: ID#{t.number} (State: {t.state.name}, Value: {t.value})")
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
            
            # Use sqlite3 directly for reliable deletion
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Need study_id to target correctly
            # study_id = summary.study_id # Attribute might be missing in some versions
            
            # Fetch from DB safely
            cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (summary.study_name,))
            row = cursor.fetchone()
            if not row:
                print(f"    [ERROR] Could not find study_id for name: {summary.study_name}")
                continue
            study_id = row[0]
            
            for trial_number in trials_to_delete:
                try:
                    # Get the actual trial_id first
                    cursor.execute("SELECT trial_id FROM trials WHERE study_id = ? AND number = ?", (study_id, trial_number))
                    row = cursor.fetchone()
                    if not row:
                        print(f"    [WARNING] Could not find trial_id for study_id={study_id}, number={trial_number}")
                        continue
                    
                    real_trial_id = row[0]
                    
                    # Delete using global trial_id (now correct)
                    cursor.execute("DELETE FROM trial_user_attributes WHERE trial_id = ?", (real_trial_id,))
                    cursor.execute("DELETE FROM trial_params WHERE trial_id = ?", (real_trial_id,))
                    cursor.execute("DELETE FROM trial_values WHERE trial_id = ?", (real_trial_id,))
                    cursor.execute("DELETE FROM trial_intermediate_values WHERE trial_id = ?", (real_trial_id,))
                    cursor.execute("DELETE FROM trials WHERE trial_id = ?", (real_trial_id,))
                    conn.commit()
                    
                    # Verify deletion
                    cursor.execute("SELECT count(*) FROM trials WHERE trial_id = ?", (real_trial_id,))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        print(f"    Deleted Trial #{trial_number} (Global ID#{real_trial_id}) (Verified)")
                    else:
                        print(f"    [WARNING] Failed to delete Trial #{trial_number} (Global ID#{real_trial_id}) (Row still exists!)")
                        
                except Exception as e:
                    print(f"    Failed to delete Trial #{trial_number}: {e}")
            
            conn.close()

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
