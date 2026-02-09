import sys
import time
from langchain_core.messages import HumanMessage, AIMessage
from app.tools.mermaid import generate_flow_diagram
from app.agents.project_lead import get_project_lead_agent
from app.agents.milestone import get_milestone_agent
from app.agents.flow_diagram import get_flow_diagram_agent
from app.tools.save_file import save_file, markdown_to_pdf


def extract_text(response) -> str:
    if isinstance(response, AIMessage):
        return response.content

    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        if messages and hasattr(messages[-1], "content"):
            return messages[-1].content

    return str(response)


def loader(message: str, delay: float = 0.9, steps: int = 3):
    for i in range(steps):
        dots = "." * (i + 1)
        sys.stdout.write(f"\r{message}{dots}")
        sys.stdout.flush()
        time.sleep(delay)
    print(" ✓")


def stage(title: str):
    print(f"\n> {title}")


class ManagerAgent:
    def __init__(self):
        self.project_lead = get_project_lead_agent()
        self.milestone_agent = get_milestone_agent()
        self.flow_diagram_agent = get_flow_diagram_agent()

    def process_request(self, user_query: str):
        stage("Manager received request")
        print(user_query)

        # -------- Project Plan --------
        stage("Generating Project Plan")
        loader("Thinking")

        project_plan = extract_text(
            self.project_lead.invoke({"messages": [HumanMessage(content=user_query)]})
        )

        save_file("project_plan.md", project_plan)
        markdown_to_pdf("project_plan.md", "project_plan.pdf")
        print("Project plan saved")

        # -------- Milestones --------
        stage("Generating Milestones")
        loader("Analyzing plan")

        milestones = extract_text(
            self.milestone_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Create a milestone table based on this plan:\n\n{project_plan}"
                        )
                    ]
                }
            )
        )

        save_file("milestone.md", milestones)
        markdown_to_pdf("milestone.md", "milestone.pdf")
        print("Milestones saved")

        # -------- Flow Diagram --------
        stage("Generating Flow Diagram")
        loader("Designing system flow", steps=4)

        flow_diagram_code = extract_text(
            self.flow_diagram_agent.invoke(
                {
                    "input": (
                        "Create a flow diagram based on this:\n\n" f"{project_plan}"
                    )
                }
            )
        )

        if not flow_diagram_code.strip().startswith("graph"):
            raise ValueError("Invalid Mermaid code returned by Flow Diagram Agent")

        generate_flow_diagram(flow_diagram_code)
        print("Flow diagram generated")

        stage("Process completed successfully")


def get_manager():
    return ManagerAgent()
