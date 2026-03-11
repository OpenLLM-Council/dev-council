import os
import sys
import sqlite3
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
    conn = sqlite3.connect(os.path.join(checkpoint_dir, "checkpoint.db"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    manager = get_manager(checkpointer=saver)

    config = {"configurable": {"thread_id": "2"}}
    # Check if there is an existing checkpoint
    try:
        existing_state = saver.get_tuple(config)
    except Exception:
        existing_state = None

    if existing_state:
        print(f"\n[INFO] Found an existing coding session checkpoint in {project_path}.")
        resume = input("Do you want to resume the previous session? (y/n): ").strip().lower()
        if resume == 'y':
            try:
                manager.process_request(user_query=None, project_path=project_path, config=config, resume=True)
                return
            except Exception as e:
                print(f"Error resuming: {e}")
                sys.exit(1)
        else:
            print("[INFO] Starting a new session.")

    user_input = input("User request: ")

    try:
        manager.process_request(user_input, project_path, config=config, resume=False)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
