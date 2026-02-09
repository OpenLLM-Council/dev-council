from app.agents.project_lead import get_project_lead_agent
from app.agents.milestone import get_milestone_agent
from langchain_core.messages import HumanMessage
from app.tools.save_file import save_file


class ManagerAgent:
    def __init__(self):
        self.project_lead = get_project_lead_agent()
        self.milestone_agent = get_milestone_agent()

    def process_request(self, user_query: str):
        print(f"--- Manager: Received request: {user_query} ---")

        # Step 1: Invoke Project Lead
        print("--- Manager: Delegating to Project Lead ---")
        project_lead_response = self.project_lead.invoke(
            {"messages": [HumanMessage(content=user_query)]}
        )

        # Try to extract content based on common patterns

        print("--- Manager: Received Project Plan ---")
        print(project_lead_response)

        # Step 2: Invoke Milestone Agent
        print("--- Manager: Delegating to Milestone Agent ---")
        milestone_response = self.milestone_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Create a milestone table based on this plan:\n\n{project_lead_response}"
                    )
                ]
            }
        )
        print("--- Manager: Received Milestone Table ---")
        print(milestone_response)
        return


def get_manager():
    return ManagerAgent()
