import sqlite3
import json
import sys
import os

def patch_db(db_path):
    print(f"Connecting to {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    try:
        # Get all trials with status 'FAIL' or 'PRUNED' or 'COMPLETE' (though failed ones are target)
        # We need to filter by user_attrs error message.
        # Structure:
        # trials: trial_id, state
        # trial_user_attributes: trial_user_attribute_id, trial_id, key, value_json
        # trial_values: trial_value_id, trial_id, objective, value, value_type

        # 1. Select trials that have an error attribute containing "Pred STD" or "Pred Max"
        query = """
        SELECT t.trial_id, t.number, ta.value_json, tv.value
        FROM trials t
        JOIN trial_user_attributes ta ON t.trial_id = ta.trial_id
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE ta.key = 'error'
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        updated_count = 0
        
        for row in rows:
            trial_id, number, error_json, current_value = row
            try:
                error_msg = json.loads(error_json)
            except:
                error_msg = str(error_json)
                
            if not isinstance(error_msg, str):
                 continue

            error_msg_lower = error_msg.lower()
            new_value = None
            
            if "pred std" in error_msg_lower:
                new_value = -0.5
            elif "pred max" in error_msg_lower:
                new_value = -0.4
            
            if new_value is not None:
                # Check need to update
                if current_value is not None and abs(current_value - new_value) < 1e-6:
                    continue

                print(f"  Trial {number}: Updating value to {new_value} (Error: {error_msg[:60]}...)")
                
                # Update trial_values
                # Check if value exists
                cursor.execute("SELECT 1 FROM trial_values WHERE trial_id = ?", (trial_id,))
                exists = cursor.fetchone()
                
                if exists:
                    cursor.execute("UPDATE trial_values SET value = ? WHERE trial_id = ?", (new_value, trial_id))
                else:
                    # Insert if missing (e.g. if it failed before reporting value?)
                    # Generally optimization returns value, so simple fail might have it.
                    # But if Exception caught in optimize_qrc, it returns -1.0 so it should be there.
                    cursor.execute("INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (?, 0, ?, 'FINITE')", (trial_id, new_value))
                
                # Ensure state is COMPLETE
                cursor.execute("UPDATE trials SET state = 'COMPLETE' WHERE trial_id = ?", (trial_id,))
                
                # Ensure status attribute is 'diverged'
                # Check if 'status' attr exists
                cursor.execute("SELECT trial_user_attribute_id, value_json FROM trial_user_attributes WHERE trial_id = ? AND key = 'status'", (trial_id,))
                status_row = cursor.fetchone()
                
                diverged_json = json.dumps("diverged")
                
                if status_row:
                    if status_row[1] != diverged_json:
                        cursor.execute("UPDATE trial_user_attributes SET value_json = ? WHERE trial_id = ? AND key = 'status'", (diverged_json, trial_id))
                else:
                    cursor.execute("INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, 'status', ?)", (trial_id, diverged_json))

                updated_count += 1
        
        conn.commit()
        print(f"  Updated {updated_count} trials.")
        
    except Exception as e:
        print(f"Error during patching: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python benchmarks/patch_optuna_db.py <path_to_db_file>")
        sys.exit(1)
        
    db_file = sys.argv[1]
    if not os.path.exists(db_file):
         if os.path.exists(os.path.join(os.getcwd(), db_file)):
            db_file = os.path.join(os.getcwd(), db_file)
         else:
             print(f"File not found: {db_file}")
             sys.exit(1)
        
    patch_db(os.path.abspath(db_file))
