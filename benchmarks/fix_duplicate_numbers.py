import optuna
import sys
import os
import sqlite3

def fix_duplicate_numbers(db_path, dry_run=True):
    url = f"sqlite:///{db_path}"
    print(f"Connecting to {url}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Identify studies with duplicates
    cursor.execute("SELECT study_id, number, count(*) FROM trials GROUP BY study_id, number HAVING count(*) > 1")
    dups = cursor.fetchall()
    
    if not dups:
        print("No duplicate trial numbers found.")
        return

    print(f"Found {len(dups)} duplicate number groups.")
    
    trials_to_delete = []

    for study_id, number, count in dups:
        # Get all trials with this number
        # Join with trial_values to get values
        # Note: trial_values might be empty for some failures/running
        cursor.execute("""
            SELECT t.trial_id, t.state, v.value 
            FROM trials t
            LEFT JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = ? AND t.number = ?
        """, (study_id, number))
        trials = cursor.fetchall()
        
        candidates = []
        for t_id, state, value in trials:
            # Check user attrs for detailed status if needed
            cursor.execute("SELECT value_json FROM trial_user_attributes WHERE trial_id = ? AND key = 'status'", (t_id,))
            status_row = cursor.fetchone()
            status = status_row[0] if status_row else "unknown"
            
            candidates.append({
                'id': t_id,
                'state': state,
                'value': value, # Can be None
                'status': status
            })
            
        # Sort candidates
        # Priority:
        # 1. State == 'COMPLETE' (or 1 in some DBs?)
        # 2. Status != 'diverged'
        # 3. ID (Latest)
        
        def priority(c):
            p_state = 1 if c['state'] == 'COMPLETE' else 0
            p_status = 1 if c['status'] != '"diverged"' and c['status'] != 'diverged' else 0 # json string might have quotes
            return (p_state, p_status, c['id'])
            
        candidates.sort(key=priority)
        
        keep = candidates[-1]
        delete_list = candidates[:-1]
        
        print(f"Duplicate Trial #{number} (Study {study_id}):")
        print(f"  Keep: ID#{keep['id']} (State: {keep['state']}, Val: {keep['value']})")
        for d in delete_list:
            print(f"  Delete: ID#{d['id']} (State: {d['state']}, Val: {d['value']})")
            trials_to_delete.append(d['id'])

    if not trials_to_delete:
        print("Nothing to delete.")
        conn.close()
        return

    if dry_run:
        print("\n[DRY RUN] No changes made.")
    else:
        print(f"\nDeleting {len(trials_to_delete)} trials...")
        for t_id in trials_to_delete:
            try:
                cursor.execute("DELETE FROM trial_user_attributes WHERE trial_id = ?", (t_id,))
                cursor.execute("DELETE FROM trial_params WHERE trial_id = ?", (t_id,))
                cursor.execute("DELETE FROM trial_values WHERE trial_id = ?", (t_id,))
                cursor.execute("DELETE FROM trial_intermediate_values WHERE trial_id = ?", (t_id,))
                cursor.execute("DELETE FROM trials WHERE trial_id = ?", (t_id,))
                conn.commit()
                print(f"  Deleted ID#{t_id}")
            except Exception as e:
                print(f"  Failed: {e}")
                
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmarks/fix_duplicate_numbers.py <db> [--no-dry-run]")
        sys.exit(1)
    
    db = sys.argv[1]
    dry = "--no-dry-run" not in sys.argv
    fix_duplicate_numbers(db, dry)
