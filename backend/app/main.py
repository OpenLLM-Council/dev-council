import os
import sys
import sqlite3
import uuid
from app.agents.manager import get_manager
from langgraph.checkpoint.sqlite import SqliteSaver


def run():
    print("\n[INFO] Starting dev-council...")
    project_path = input("Project path: ").strip()
    
    if not project_path:
        print("Project path is required.")
        sys.exit(1)

    checkpoint_dir = os.path.join(project_path, ".checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    db_path = os.path.join(checkpoint_dir, "checkpoint.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    thread_id = "2"
    config = {"configurable": {"thread_id": thread_id}}

    manager = get_manager(checkpointer=saver)

    # Check if there is an existing checkpoint
    try:
        existing_state = saver.get_tuple(config)
    except Exception:
        existing_state = None

    if existing_state:
        print(f"\n[INFO] Found an existing coding session checkpoint in {project_path}.")
        choice = input("Resume previous session? (y)es / (n)ew / (r)eset checkpoint: ").strip().lower()
        if choice in ('y', 'yes'):
            try:
                manager.process_request(user_query=None, project_path=project_path, config=config, resume=True)
                return
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user.")
                return
            except Exception as e:
                print(f"Error resuming: {e}")
                sys.exit(1)
        elif choice in ('r', 'reset'):
            print("[INFO] Resetting checkpoint...")
            conn.close()
            os.remove(db_path)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            manager = get_manager(checkpointer=saver)
            print("[INFO] Checkpoint cleared. Starting fresh.")
        else:
            # New session — use a fresh thread ID so old checkpoint doesn't interfere
            thread_id = uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print(f"[INFO] Starting a new session (thread: {thread_id}).")

    user_input = input("User request: ")

    try:
        manager.process_request(user_input, project_path, config=config, resume=False)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Progress saved to checkpoint.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
