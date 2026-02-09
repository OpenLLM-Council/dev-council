import os
import sys
from app.agents.manager import get_manager


def run():
    manager = get_manager()

    user_input = input("User request: ")

    try:
        manager.process_request(user_input)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
