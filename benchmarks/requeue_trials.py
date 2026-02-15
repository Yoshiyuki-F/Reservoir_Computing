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

        # First, collect valid trials
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            val = trial.value
            if val is not None and val >= min_score:
                # Add to list to queue
                trials_to_enqueue.append(trial.params)

        if not trials_to_enqueue:
            print("  No trials found with score >= 0.0")
            continue

        print(f"  Found {len(trials_to_enqueue)} successful trials. Enqueuing them...")

        # Enqueue them
        for params in trials_to_enqueue:
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
