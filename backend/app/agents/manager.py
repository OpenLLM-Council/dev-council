import sys
import time
from typing import TypedDict, Literal

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START

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


class ManagerState(TypedDict):
    input: str
    project_plan: str
    milestones: str
    flow_diagram_code: str
    feedback: str
    revision_needed: bool


def call_project_lead(state: ManagerState):
    """Generates or revises the project plan."""
    project_lead = get_project_lead_agent()
    user_query = state.get("input")
    revision_needed = state.get("revision_needed", False)
    current_plan = state.get("project_plan", "")
    feedback = state.get("feedback", "")

    if revision_needed:
        stage("Revising SRS based on feedback")
        loader("Revising")
        revision_prompt = (
            f"You previously generated the following SRS document for this request:\n\n"
            f"--- ORIGINAL USER REQUEST ---\n{user_query}\n\n"
            f"--- CURRENT SRS DOCUMENT ---\n{current_plan}\n\n"
            f"--- USER FEEDBACK ---\n{feedback}\n\n"
            f"Please revise the SRS document to incorporate the user's feedback. "
            f"Output ONLY the complete revised SRS document."
        )
        response = project_lead.invoke(
            {"messages": [HumanMessage(content=revision_prompt)]}
        )
    else:
        stage("Generating Project Plan")
        loader("Thinking")
        response = project_lead.invoke({"messages": [HumanMessage(content=user_query)]})

    plan_text = extract_text(response)

    save_file("project_plan.md", plan_text)
    markdown_to_pdf("project_plan.md", "project_plan.pdf")
    print("Project plan saved")

    return {"project_plan": plan_text, "revision_needed": False}


def human_review(state: ManagerState):
    """Asks the user for review and sets the next path."""
    stage("SRS Review")
    print("\n📄 SRS document saved. Please review it before approving:")
    print(f"   Markdown: outputs/project_plan.md")
    print(f"   PDF:      outputs/project_plan.pdf\n")

    while True:
        decision = input("Do you approve this SRS? (approve/edit): ").strip().lower()
        if decision == "approve":
            print("SRS approved ✓")
            return {"revision_needed": False}
        elif decision == "edit":
            feedback = input("What changes should be made? ").strip()
            if not feedback:
                print("No feedback provided. Please try again.")
                continue
            return {"revision_needed": True, "feedback": feedback}
        else:
            print("Invalid input. Please enter 'approve' or 'edit'.")


def call_milestone(state: ManagerState):
    """Generates milestones based on the plan."""
    stage("Generating Milestones")
    loader("Analyzing plan")

    milestone_agent = get_milestone_agent()
    project_plan = state["project_plan"]

    response = milestone_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Create a milestone table based on this plan:\n\n{project_plan}"
                )
            ]
        }
    )
    milestones = extract_text(response)

    save_file("milestone.md", milestones)
    markdown_to_pdf("milestone.md", "milestone.pdf")
    print("Milestones saved")

    return {"milestones": milestones}


def call_flow_diagram(state: ManagerState):
    """Generates a flow diagram based on the plan."""
    stage("Generating Flow Diagram")
    loader("Designing system flow", steps=4)

    flow_diagram_agent = get_flow_diagram_agent()
    project_plan = state["project_plan"]

    response = flow_diagram_agent.invoke(
        {"input": ("Create a flow diagram based on this:\n\n" f"{project_plan}")}
    )
    flow_diagram_code = extract_text(response)

    if not flow_diagram_code.strip().startswith("graph"):
        print("Warning: unexpected Mermaid code format")

    generate_flow_diagram(flow_diagram_code)
    print("Flow diagram generated")

    stage("Process completed successfully")
    return {"flow_diagram_code": flow_diagram_code}


def check_review(state: ManagerState) -> Literal["call_project_lead", "call_milestone"]:
    if state.get("revision_needed"):
        return "call_project_lead"
    return "call_milestone"


class ManagerAgent:
    def __init__(self):
        workflow = StateGraph(ManagerState)

        workflow.add_node("call_project_lead", call_project_lead)
        workflow.add_node("human_review", human_review)
        workflow.add_node("call_milestone", call_milestone)
        workflow.add_node("call_flow_diagram", call_flow_diagram)

        workflow.add_edge(START, "call_project_lead")
        workflow.add_edge("call_project_lead", "human_review")

        workflow.add_conditional_edges("human_review", check_review)

        workflow.add_edge("call_milestone", "call_flow_diagram")
        workflow.add_edge("call_flow_diagram", END)

        self.app = workflow.compile()

    def process_request(self, user_query: str):
        stage("Manager received request")
        print(user_query)

        initial_state = {"input": user_query}

        self.app.invoke(initial_state)


def get_manager():
    return ManagerAgent()
