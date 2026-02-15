import optuna
import sys
import os

def patch_db(db_path):
    url = f"sqlite:///{db_path}"
    print(f"Connecting to {url}...")
    
    # We need to access storage directly to update values of finished trials
    try:
        storage = optuna.storages.RDBStorage(url=url)
    except Exception as e:
        print(f"Failed to connect to storage: {e}")
        return

    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"Failed to get study summaries: {e}")
        return

    for study_summary in summaries:
        print(f"Scanning study: {study_summary.study_name}")
        try:
            study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        except Exception as e:
            print(f"  Failed to load study: {e}")
            continue
        
        updated_count = 0
        for trial in study.trials:
            # We are interested in trials that have an error message in user_attrs
            error_msg = trial.user_attrs.get("error", "")
            if not isinstance(error_msg, str) or not error_msg:
                continue
                
            error_msg_lower = error_msg.lower()
            new_value = None
            
            if "pred std" in error_msg_lower:
                new_value = -0.5
            elif "pred max" in error_msg_lower:
                new_value = -0.4
                
            if new_value is not None:
                # Check if update is needed
                current_value = trial.value
                
                # Check if current_value is already correct (approximate float comparison)
                if current_value is not None and abs(current_value - new_value) < 1e-6:
                    continue

                print(f"  Trial {trial.number} (Current: {current_value}): Updating to {new_value} (Error: {error_msg[:60]}...)")
                
                try:
                    # Update value
                    # Note: set_trial_values takes a list of values
                    storage.set_trial_values(trial._trial_id, [new_value])
                    updated_count += 1
                    
                    # Also ensure the user_attr "status" is set to "diverged" if not already
                    if trial.user_attrs.get("status") != "diverged":
                        storage.set_trial_user_attr(trial._trial_id, "status", "diverged")
                        
                except Exception as e:
                    print(f"    Failed to update trial {trial.number}: {e}")

        print(f"  Updated {updated_count} trials in study {study_summary.study_name}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmarks/patch_optuna_db.py <path_to_db_file>")
        print("Example: uv run python benchmarks/patch_optuna_db.py benchmarks/optuna_qrc_coherent_drive.db")
        sys.exit(1)
        
    db_file = sys.argv[1]
    # Allow relative path resolution
    if not os.path.exists(db_file):
        # Try relative to CWD if not found
        if os.path.exists(os.path.join(os.getcwd(), db_file)):
            db_file = os.path.join(os.getcwd(), db_file)
        else:
             print(f"File not found: {db_file}")
             sys.exit(1)
        
    patch_db(os.path.abspath(db_file))
