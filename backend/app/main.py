import os
import sys
from app.agents.project_lead import get_project_lead_agent


def run():
    agent = get_project_lead_agent()

    user_input = input("User request: ")

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        output = result["messages"][-1].content
        print(output)

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/output.md", "w", encoding="utf-8") as f:
            f.write(output)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
