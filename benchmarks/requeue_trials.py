import optuna
import sys
import os

def requeue_successful_trials(db_path, min_score=0.0):
    url = f"sqlite:///{db_path}"
    print(f"Connecting to {url}...")
    
    try:
        storage = optuna.storages.RDBStorage(url=url)
        summaries = optuna.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"Failed to connect/read DB: {e}")
        return

    for study_summary in summaries:
        print(f"Scanning study: {study_summary.study_name}")
        try:
            study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        except Exception as e:
            print(f"  Failed to load study: {e}")
            continue

        requeue_count = 0
        trials_to_enqueue = []

        # Helper to normalize params for consistent hashing
        def normalize_params(params):
            norm = {}
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    norm[k] = round(float(v), 8) # Force float and round
                else:
                    norm[k] = str(v) # Force str for others
            return json.dumps(norm, sort_keys=True)

        # First, collect valid trials and check for existing waiting
        import json
        waiting_params = set()
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.WAITING:
                waiting_params.add(normalize_params(trial.params))

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            val = trial.value
            if val is not None and val >= min_score:
                # Check if already waiting
                params_key = normalize_params(trial.params)
                if params_key in waiting_params:
                    # print(f"  [DEBUG] Skipping already waiting params: {params_key}")
                    pass
                else:
                    trials_to_enqueue.append(trial.params)
                    waiting_params.add(params_key) # Avoid double enqueue in same loop

        if not trials_to_enqueue:
            # print("  No new successful trials to requeue.")
            continue

        if not trials_to_enqueue:
           # print("  No new successful trials to requeue.")
           continue
        
        # New robust fuzzy matching logic
        final_enqueue_list = []
        for params in trials_to_enqueue:
            tgt_json = normalize_params(params)
            tgt_dict = json.loads(tgt_json)
            
            is_duplicate = False
            for wp_json in waiting_params:
                wp = json.loads(wp_json)
                
                # Check fuzzy match
                common_keys = set(tgt_dict.keys()) & set(wp.keys())
                if not common_keys: continue
                if len(common_keys) != len(tgt_dict): continue # Keys must match exactly

                diff = 0
                for k in common_keys:
                    v1 = tgt_dict[k]
                    v2 = wp[k]
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        diff += abs(v1 - v2)
                    elif str(v1) != str(v2):
                        diff += 1.0 
                
                # If very close, consider it a duplicate
                if diff < 1e-9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_enqueue_list.append(params)
                # Add to waiting_params so we don't add duplicates within this loop
                waiting_params.add(tgt_json)

        if not final_enqueue_list:
            print("  All candidate trials are already in queue (fuzzy match).")
            continue

        print(f"  Found {len(final_enqueue_list)} successful trials not yet in queue. Enqueuing them...")

        # Enqueue them
        for params in final_enqueue_list:
            study.enqueue_trial(params)
            requeue_count += 1
        
        print(f"  Successfully requeued {requeue_count} trials.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmarks/requeue_trials.py <path_to_db_file>")
        sys.exit(1)
        
    db_file = sys.argv[1]
    if not os.path.exists(db_file):
         if os.path.exists(os.path.join(os.getcwd(), db_file)):
            db_file = os.path.join(os.getcwd(), db_file)
         else:
             print(f"File not found: {db_file}")
             sys.exit(1)
        
    requeue_successful_trials(os.path.abspath(db_file))
