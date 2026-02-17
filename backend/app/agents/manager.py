import sys
import time
from typing import TypedDict, Literal, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
import pyfiglet

from app.tools.mermaid import generate_flow_diagram
from app.agents.project_lead import get_project_lead_agent
from app.agents.milestone import get_milestone_agent
from app.agents.flow_diagram import get_flow_diagram_agent
from app.tools.save_file import save_file, markdown_to_pdf
from app.agents.tech_stack_agent import get_tech_stack_agent
from app.agents.consensus_agent import get_consensus_agent, get_manager_decision_agent
from app.tools.llm_resources import get_available_llms

console = Console()


def extract_text(response) -> str:
    if isinstance(response, AIMessage):
        return response.content

    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        if messages and hasattr(messages[-1], "content"):
            return messages[-1].content

    return str(response)



def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: merges two dicts so parallel branches combine their proposals."""
    merged = a.copy() if a else {}
    if b:
        merged.update(b)
    return merged


class ManagerState(TypedDict):
    input: str
    project_plan: str
    milestones: str
    flow_diagram_code: str
    feedback: str
    tech_stack: str
    revision_needed: bool
    current_milestone: str
    milestone_folder: str
    llm_proposals: Annotated[dict, merge_dicts]
    chosen_approach: str


def call_project_lead(state: ManagerState):
    """Generates or revises the project plan."""
    project_lead = get_project_lead_agent()
    user_query = state.get("input")
    revision_needed = state.get("revision_needed", False)
    current_plan = state.get("project_plan", "")
    feedback = state.get("feedback", "")

    if revision_needed:
        console.rule("[bold yellow]Revising SRS[/bold yellow]")
        with console.status("[bold green]Incorporating feedback...", spinner="dots"):
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
        console.rule("[bold cyan]Generating Project Plan[/bold cyan]")
        with console.status("[bold green]Thinking...", spinner="dots"):
            response = project_lead.invoke(
                {"messages": [HumanMessage(content=user_query)]}
            )

    plan_text = extract_text(response)

    # Save artifacts
    save_file("project_plan.md", plan_text)
    markdown_to_pdf("project_plan.md", "project_plan.pdf")
    console.print("[bold green]✓ Project plan saved[/bold green]")

    return {"project_plan": plan_text, "revision_needed": False}


def human_review(state: ManagerState):
    """Asks the user for review and sets the next path."""
    console.rule("[bold magenta]SRS Review[/bold magenta]")
    console.print(
        Panel(
            f"[bold]Markdown:[/bold] outputs/project_plan.md\n[bold]PDF:[/bold]      outputs/project_plan.pdf",
            title="[bold blue]Documents Ready for Review[/bold blue]",
            border_style="green",
        )
    )

    while True:
        decision = (
            console.input(
                "[bold yellow]Do you approve this SRS? (approve/edit): [/bold yellow]"
            )
            .strip()
            .lower()
        )
        if decision == "approve":
            console.print("[bold green]SRS approved ✓[/bold green]")
            return {"revision_needed": False}
        elif decision == "edit":
            feedback = console.input(
                "[bold cyan]What changes should be made? [/bold cyan]"
            ).strip()
            if not feedback:
                console.print(
                    "[bold red]No feedback provided. Please try again.[/bold red]"
                )
                continue
            return {"revision_needed": True, "feedback": feedback}
        else:
            console.print(
                "[bold red]Invalid input. Please enter 'approve' or 'edit'.[/bold red]"
            )


def call_milestone(state: ManagerState):
    """Generates milestones based on the plan."""
    console.rule("[bold cyan]Generating Milestones[/bold cyan]")
    with console.status("[bold green]Analyzing plan...", spinner="dots"):
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

    console.print("[bold green]✓ Milestones saved[/bold green]")
    return {"milestones": milestones}


def call_flow_diagram(state: ManagerState):
    """Generates a flow diagram based on the plan."""
    console.rule("[bold cyan]Generating Flow Diagram[/bold cyan]")
    with console.status("[bold green]Designing system flow...", spinner="dots"):
        flow_diagram_agent = get_flow_diagram_agent()
        project_plan = state["project_plan"]

        response = flow_diagram_agent.invoke(
            {"input": ("Create a flow diagram based on this:\n\n" f"{project_plan}")}
        )
        flow_diagram_code = extract_text(response)

        if not flow_diagram_code.strip().startswith("graph"):
            console.print(
                "[bold red]Warning: unexpected Mermaid code format[/bold red]"
            )

        generate_flow_diagram(flow_diagram_code)

    console.print("[bold green]✓ Flow diagram generated[/bold green]")
    return {"flow_diagram_code": flow_diagram_code}


def call_tech_stack(state: ManagerState):
    """Generates or revises a tech stack based on the SRS document."""
    tech_stack_agent = get_tech_stack_agent()
    project_plan = state["project_plan"]
    revision_needed = state.get("revision_needed", False)
    current_tech_stack = state.get("tech_stack", "")
    feedback = state.get("feedback", "")

    if revision_needed:
        console.rule("[bold yellow]Revising Tech Stack[/bold yellow]")
        with console.status(
            "[bold green]Revising based on feedback...", spinner="dots"
        ):
            prompt = (
                f"Project Plan:\n\n{project_plan}\n\n"
                f"--- EXISTING TECH STACK ---\n{current_tech_stack}\n\n"
                f"--- USER FEEDBACK ---\n{feedback}\n\n"
                f"Please revise the Tech Stack table based on the feedback. Output ONLY the table."
            )
            response = tech_stack_agent.invoke(
                {"messages": [HumanMessage(content=prompt)]}
            )
    else:
        console.rule("[bold cyan]Generating Tech Stack[/bold cyan]")
        with console.status("[bold green]Designing tech stack...", spinner="dots"):
            response = tech_stack_agent.invoke(
                {"messages": [HumanMessage(content=f"Project Plan:\n\n{project_plan}")]}
            )

    tech_stack = extract_text(response)

    save_file("tech_stack.md", tech_stack)
    markdown_to_pdf("tech_stack.md", "tech_stack.pdf")
    console.print("[bold green]✓ Tech stack saved[/bold green]")

    return {"tech_stack": tech_stack, "revision_needed": False}


def tech_stack_review(state: ManagerState):
    """Asks the user for review of the tech stack."""
    console.rule("[bold magenta]Tech Stack Review[/bold magenta]")
    console.print(
        Panel(
            f"[bold]Markdown:[/bold] outputs/tech_stack.md\n[bold]PDF:[/bold]      outputs/tech_stack.pdf",
            title="[bold blue]Tech Stack Ready for Review[/bold blue]",
            border_style="green",
        )
    )

    while True:
        decision = (
            console.input(
                "[bold yellow]Do you approve this Tech Stack? (approve/edit): [/bold yellow]"
            )
            .strip()
            .lower()
        )
        if decision == "approve":
            console.print("[bold green]Tech Stack approved ✓[/bold green]")
            return {"revision_needed": False}
        elif decision == "edit":
            feedback = console.input(
                "[bold cyan]What changes should be made? [/bold cyan]"
            ).strip()
            if not feedback:
                console.print(
                    "[bold red]No feedback provided. Please try again.[/bold red]"
                )
                continue
            return {"revision_needed": True, "feedback": feedback}
        else:
            console.print(
                "[bold red]Invalid input. Please enter 'approve' or 'edit'.[/bold red]"
            )


def check_review(state: ManagerState) -> Literal["call_project_lead", "call_milestone"]:
    if state.get("revision_needed"):
        return "call_project_lead"
    return "call_milestone"


def check_tech_stack_review(
    state: ManagerState,
) -> Literal["call_tech_stack", "pick_milestone"]:
    if state.get("revision_needed"):
        return "call_tech_stack"
    return "pick_milestone"


def pick_milestone(state: ManagerState):
    """Extracts the first milestone from the milestone table."""
    console.rule("[bold cyan]Picking First Milestone[/bold cyan]")
    milestones_text = state.get("milestones", "")

    # Parse the markdown table to get the first milestone description
    lines = milestones_text.strip().split("\n")
    first_milestone = ""
    for line in lines:
        # Skip header rows and separator rows
        if line.startswith("|") and "---" not in line and "Milestone" not in line and "Description" not in line:
            cols = [col.strip() for col in line.split("|") if col.strip()]
            if len(cols) >= 2:
                first_milestone = cols[1]  # Description column
                break

    if not first_milestone:
        first_milestone = "First milestone from the project plan"

    console.print(f"[bold]Selected Milestone:[/bold] {first_milestone}")
    return {"current_milestone": first_milestone, "milestone_folder": "milestone_1", "llm_proposals": {}}


def make_proposal_node(llm_name: str, model_name: str):
    """Factory: returns a node function that generates a proposal for the given LLM."""
    def propose(state: ManagerState):
        console.print(f"[bold blue]  ➤ {llm_name} ({model_name}) proposing...[/bold blue]")
        agent = get_consensus_agent(model_name)
        milestone = state["current_milestone"]
        project_plan = state.get("project_plan", "")
        tech_stack = state.get("tech_stack", "")

        input_text = (
            f"## Milestone\n{milestone}\n\n"
            f"## Project Plan (SRS)\n{project_plan}\n\n"
            f"## Tech Stack\n{tech_stack}"
        )

        with console.status(f"[bold green]{llm_name} thinking...", spinner="dots"):
            response = agent.invoke({"input": input_text})

        proposal = response.content if hasattr(response, "content") else str(response)
        console.print(f"[bold green]  ✓ {llm_name} done[/bold green]")

        # Save individual proposal
        folder = state.get("milestone_folder", "milestone_1")
        safe_name = llm_name.lower().replace(" ", "_")
        file_path = f"{folder}/proposal_{safe_name}.md"
        save_file(file_path, f"# Proposal from {llm_name}\n\n{proposal}")
        console.print(f"[dim]  Saved to outputs/{file_path}[/dim]")

        # Merge into existing proposals
        existing = state.get("llm_proposals", {})
        updated = {**existing, llm_name: proposal}
        return {"llm_proposals": updated}

    return propose


def manager_decision(state: ManagerState):
    """The manager evaluates all LLM proposals and picks the best one."""
    console.rule("[bold cyan]Manager Decision[/bold cyan]")
    proposals = state.get("llm_proposals", {})
    milestone = state.get("current_milestone", "")
    revision_needed = state.get("revision_needed", False)
    feedback = state.get("feedback", "")

    # Build a summary of all proposals
    proposals_text = ""
    for llm_name, proposal in proposals.items():
        proposals_text += f"\n### Proposal from {llm_name}\n{proposal}\n"

    if revision_needed and feedback:
        input_text = (
            f"## Milestone\n{milestone}\n\n"
            f"## All LLM Proposals\n{proposals_text}\n\n"
            f"## Previous Decision\n{state.get('chosen_approach', '')}\n\n"
            f"## User Feedback\n{feedback}\n\n"
            f"Revise your decision based on the feedback."
        )
    else:
        input_text = (
            f"## Milestone\n{milestone}\n\n"
            f"## Proposals\n{proposals_text}"
        )

    manager_agent = get_manager_decision_agent()

    with console.status("[bold green]Manager evaluating proposals...", spinner="dots"):
        response = manager_agent.invoke({"input": input_text})

    chosen = response.content if hasattr(response, "content") else str(response)

    # Wrap with heading to satisfy markdown-pdf hierarchy requirement
    folder = state.get("milestone_folder", "milestone_1")
    md_content = f"# Consensus Decision\n\n{chosen}"
    save_file(f"{folder}/consensus_decision.md", md_content)
    try:
        markdown_to_pdf(f"{folder}/consensus_decision.md", f"{folder}/consensus_decision.pdf")
    except Exception as e:
        console.print(f"[bold yellow]⚠ PDF generation skipped: {e}[/bold yellow]")
    console.print("[bold green]✓ Manager decision saved[/bold green]")

    return {"chosen_approach": chosen, "revision_needed": False}


def consensus_review(state: ManagerState):
    """HITL review for the manager's consensus decision."""
    console.rule("[bold magenta]Consensus Review[/bold magenta]")
    folder = state.get("milestone_folder", "milestone_1")
    console.print(
        Panel(
            f"[bold]Markdown:[/bold] outputs/{folder}/consensus_decision.md\n[bold]PDF:[/bold]      outputs/{folder}/consensus_decision.pdf",
            title="[bold blue]Consensus Decision Ready for Review[/bold blue]",
            border_style="green",
        )
    )

    while True:
        decision = (
            console.input(
                "[bold yellow]Do you approve this decision? (approve/edit): [/bold yellow]"
            )
            .strip()
            .lower()
        )
        if decision == "approve":
            console.print("[bold green]Consensus decision approved ✓[/bold green]")
            return {"revision_needed": False}
        elif decision == "edit":
            feedback = console.input(
                "[bold cyan]What changes should be made? [/bold cyan]"
            ).strip()
            if not feedback:
                console.print(
                    "[bold red]No feedback provided. Please try again.[/bold red]"
                )
                continue
            return {"revision_needed": True, "feedback": feedback}
        else:
            console.print(
                "[bold red]Invalid input. Please enter 'approve' or 'edit'.[/bold red]"
            )


def check_consensus_review(
    state: ManagerState,
) -> Literal["manager_decision", "__end__"]:
    if state.get("revision_needed"):
        return "manager_decision"
    return END


class ManagerAgent:
    def __init__(self):
        workflow = StateGraph(ManagerState)

        # --- Existing nodes ---
        workflow.add_node("call_project_lead", call_project_lead)
        workflow.add_node("human_review", human_review)
        workflow.add_node("call_milestone", call_milestone)
        workflow.add_node("call_flow_diagram", call_flow_diagram)
        workflow.add_node("call_tech_stack", call_tech_stack)
        workflow.add_node("tech_stack_review", tech_stack_review)

        # --- Consensus nodes ---
        workflow.add_node("pick_milestone", pick_milestone)
        workflow.add_node("manager_decision", manager_decision)
        workflow.add_node("consensus_review", consensus_review)

        # Dynamically add one proposal node per available LLM
        available_llms = get_available_llms()
        proposal_node_names = []
        for llm_info in available_llms:
            node_name = f"propose_{llm_info['name']}"
            proposal_node_names.append(node_name)
            workflow.add_node(node_name, make_proposal_node(llm_info["name"], llm_info["model"]))

        # --- Edges ---
        workflow.add_edge(START, "call_project_lead")
        workflow.add_edge("call_project_lead", "human_review")
        workflow.add_conditional_edges("human_review", check_review)

        workflow.add_edge("call_milestone", "call_flow_diagram")
        workflow.add_edge("call_flow_diagram", "call_tech_stack")
        workflow.add_edge("call_tech_stack", "tech_stack_review")
        workflow.add_conditional_edges("tech_stack_review", check_tech_stack_review)

        # Fan-out: pick_milestone -> all proposal nodes
        for node_name in proposal_node_names:
            workflow.add_edge("pick_milestone", node_name)

        # Fan-in: all proposal nodes -> manager_decision
        for node_name in proposal_node_names:
            workflow.add_edge(node_name, "manager_decision")

        workflow.add_edge("manager_decision", "consensus_review")
        workflow.add_conditional_edges("consensus_review", check_consensus_review)

        self.app = workflow.compile()

    def process_request(self, user_query: str):
        title = pyfiglet.figlet_format("dev-council", font="slant")
        console.print(Text(title, style="bold magenta"))

        console.rule("[bold blue]New Request[/bold blue]")
        console.print(f"[bold]Requests:[/bold] {user_query}")

        initial_state = {"input": user_query}

        self.app.invoke(initial_state)

        console.rule("[bold green]Process Completed Successfully[/bold green]")


def get_manager():
    return ManagerAgent()
