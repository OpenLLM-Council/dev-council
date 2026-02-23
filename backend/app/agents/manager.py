import os
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Literal, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import pyfiglet

from app.tools.mermaid import generate_flow_diagram
from app.agents.project_lead import get_project_lead_agent
from app.agents.milestone import get_milestone_agent
from app.agents.flow_diagram import get_flow_diagram_agent
from app.tools.save_file import save_file, markdown_to_pdf
from app.agents.tech_stack_agent import get_tech_stack_agent
from app.agents.tech_stack_agent import get_tech_stack_agent
from app.agents.consensus_agent import get_consensus_agent, get_manager_decision_agent
from app.agents.coder_agent import get_coder_agent
from app.agents.reviewer_agent import get_reviewer_agent
from app.agents.instructions_agent import get_instructions_agent
from app.tools.llm_resources import get_available_llms, get_coder_llms
from app.tools.file_writer import write_code_files

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
    all_milestones: list[str]
    current_milestone_index: int
    llm_proposals: Annotated[dict, merge_dicts]
    proceed_to_next: bool

    project_path: str
    chosen_approach: str
    chosen_llm_model: str
    memory: InMemorySaver
    generated_code: str
    review_feedback: str
    code_approved: bool
    code_attempt: int
    run_instructions: str
    user_run_feedback: str


def call_project_lead(state: ManagerState):
    """Generates or revises the project plan."""
    project_lead = get_project_lead_agent(memory=state.get("memory"))
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

    project_path = state.get("project_path", "outputs")
    save_file("project_plan.md", plan_text, base_path=project_path)
    markdown_to_pdf("project_plan.md", "project_plan.pdf", base_path=project_path)
    console.print("[bold green]✓ Project plan saved[/bold green]")

    return {"project_plan": plan_text, "revision_needed": False}


def human_review(state: ManagerState):
    """Asks the user for review and sets the next path."""
    console.rule("[bold magenta]SRS Review[/bold magenta]")
    project_path = state.get("project_path", "outputs")
    console.print(
        Panel(
            f"[bold]Markdown:[/bold] {project_path}/project_plan.md\n[bold]PDF:[/bold]      {project_path}/project_plan.pdf",
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
        milestone_agent = get_milestone_agent(memory=state.get("memory"))
        project_plan = state["project_plan"]

        response = milestone_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Create a milestone table based on this plan:\n\n{project_plan}"
                    )
                ],
            },
            {"configurable": {"thread_id": "1"}},
        )
        milestones = extract_text(response)

        project_path = state.get("project_path", "outputs")
        save_file("milestone.md", milestones, base_path=project_path)
        markdown_to_pdf("milestone.md", "milestone.pdf", base_path=project_path)

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

        project_path = state.get("project_path", "outputs")
        generate_flow_diagram(flow_diagram_code, project_path)

    console.print("[bold green]✓ Flow diagram generated[/bold green]")
    return {"flow_diagram_code": flow_diagram_code}


def call_tech_stack(state: ManagerState):
    """Generates or revises a tech stack based on the SRS document."""
    tech_stack_agent = get_tech_stack_agent(memory=state.get("memory"))
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
                {
                    "messages": [
                        HumanMessage(content=f"Project Plan:\n\n{project_plan}")
                    ]
                },
                {"configurable": {"thread_id": "1"}},
            )

    tech_stack = extract_text(response)

    project_path = state.get("project_path", "outputs")
    save_file("tech_stack.md", tech_stack, base_path=project_path)
    markdown_to_pdf("tech_stack.md", "tech_stack.pdf", base_path=project_path)
    console.print("[bold green]✓ Tech stack saved[/bold green]")

    return {"tech_stack": tech_stack, "revision_needed": False}


def tech_stack_review(state: ManagerState):
    """Asks the user for review of the tech stack."""
    console.rule("[bold magenta]Tech Stack Review[/bold magenta]")
    project_path = state.get("project_path", "outputs")
    console.print(
        Panel(
            f"[bold]Markdown:[/bold] {project_path}/tech_stack.md\n[bold]PDF:[/bold]      {project_path}/tech_stack.pdf",
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
    """Parses all milestones and prepares the first one for proposals and coding."""
    console.rule("[bold cyan]Preparing Milestones[/bold cyan]")
    milestones_text = state.get("milestones", "")

    lines = milestones_text.strip().split("\n")
    parsed_milestones = []

    for line in lines:
        if (
            line.startswith("|")
            and "---" not in line
            and "Milestone" not in line
            and "Description" not in line
        ):
            cols = [col.strip() for col in line.split("|") if col.strip()]
            if len(cols) >= 2:
                parsed_milestones.append(cols[1])

    if not parsed_milestones:
        parsed_milestones = ["First milestone from the project plan"]

    index = state.get("current_milestone_index", 0)

    if index < len(parsed_milestones):
        current_milestone = parsed_milestones[index]
    else:
        current_milestone = parsed_milestones[-1]

    console.print(f"[bold]Total Milestones:[/bold] {len(parsed_milestones)}")
    console.print(
        f"[bold]Selected Milestone (Index {index}):[/bold] {current_milestone}"
    )

    return {
        "all_milestones": parsed_milestones,
        "current_milestone": current_milestone,
        "current_milestone_index": index,
        "milestone_folder": f"milestone_{index + 1}",
        "llm_proposals": {},
        "chosen_approach": "",
    }


def make_proposal_node(llm_name: str, model_name: str):
    """Factory: returns a node function that generates a proposal for the given LLM."""

    def propose(state: ManagerState):
        console.print(
            f"[bold blue]  ➤ {llm_name} ({model_name}) proposing...[/bold blue]"
        )
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

        folder = state.get("milestone_folder", "milestone_1")
        safe_name = llm_name.lower().replace(" ", "_")
        file_path = f"{folder}/proposal_{safe_name}.md"
        project_path = state.get("project_path", "outputs")
        save_file(
            file_path,
            f"# Proposal from {llm_name}\n\n{proposal}",
            base_path=project_path,
        )
        console.print(f"[dim]  Saved to {project_path}/{file_path}[/dim]")

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
        input_text = f"## Milestone\n{milestone}\n\n" f"## Proposals\n{proposals_text}"

    manager_agent = get_manager_decision_agent()

    with console.status("[bold green]Manager evaluating proposals...", spinner="dots"):
        response = manager_agent.invoke({"input": input_text})

    chosen = response.content if hasattr(response, "content") else str(response)

    # Wrap with heading to satisfy markdown-pdf hierarchy requirement
    folder = state.get("milestone_folder", "milestone_1")
    md_content = f"# Consensus Decision\n\n{chosen}"
    project_path = state.get("project_path", "outputs")
    save_file(f"{folder}/consensus_decision.md", md_content, base_path=project_path)
    try:
        markdown_to_pdf(
            f"{folder}/consensus_decision.md",
            f"{folder}/consensus_decision.pdf",
            base_path=project_path,
        )
    except Exception as e:
        console.print(f"[bold yellow]⚠ PDF generation skipped: {e}[/bold yellow]")
    console.print("[bold green]✓ Manager decision saved[/bold green]")

    return {"chosen_approach": chosen, "revision_needed": False}


def consensus_review(state: ManagerState):
    """HITL review for the manager's consensus decision."""
    console.rule("[bold magenta]Consensus Review[/bold magenta]")
    folder = state.get("milestone_folder", "milestone_1")
    project_path = state.get("project_path", "outputs")
    console.print(
        Panel(
            f"[bold]Markdown:[/bold] {project_path}/{folder}/consensus_decision.md\n[bold]PDF:[/bold]      {project_path}/{folder}/consensus_decision.pdf",
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
) -> Literal["manager_decision", "code_milestone"]:
    if state.get("revision_needed"):
        return "manager_decision"
    return "code_milestone"


def _extract_chosen_llm_model(chosen_approach: str) -> str:
    """Picks a coder-specialized model if available, otherwise falls back to
    whichever LLM the manager named in the decision text."""
    coder_llms = get_coder_llms()
    if coder_llms:
        if chosen_approach:
            for llm_info in coder_llms:
                if llm_info["name"].upper() in chosen_approach.upper():
                    return llm_info["model"]
        return coder_llms[0]["model"]

    available = get_available_llms()
    if chosen_approach:
        for llm_info in available:
            if llm_info["name"].upper() in chosen_approach.upper():
                return llm_info["model"]

    return available[0]["model"] if available else ""


def code_milestone(state: ManagerState):
    """The assigned LLM writes code for the current milestone."""
    console.rule("[bold cyan]Coding Milestone[/bold cyan]")

    milestone = state.get("current_milestone", "")
    tech_stack = state.get("tech_stack", "")
    chosen_approach = state.get("chosen_approach", "")
    project_plan = state.get("project_plan", "")
    review_feedback = state.get("review_feedback", "")
    revision_needed = state.get("revision_needed", False)

    # Resolve the chosen LLM model
    chosen_model = state.get("chosen_llm_model") or _extract_chosen_llm_model(
        chosen_approach
    )

    console.print(f"[bold blue]  ➤ Coding with model: {chosen_model}[/bold blue]")

    if revision_needed and review_feedback:
        input_text = (
            f"## Milestone\n{milestone}\n\n"
            f"## Chosen Approach (from Consensus)\n{chosen_approach}\n\n"
            f"## Approved Tech Stack\n{tech_stack}\n\n"
            f"## Reviewer Feedback\n{review_feedback}\n\n"
            f"Fix all issues raised in the reviewer feedback."
        )
        agent = get_coder_agent(chosen_model, revision=True)
    else:
        input_text = (
            f"## Milestone\n{milestone}\n\n"
            f"## Chosen Approach (from Consensus)\n{chosen_approach}\n\n"
            f"## Approved Tech Stack\n{tech_stack}\n\n"
            f"## Project SRS (context)\n{project_plan}"
        )
        agent = get_coder_agent(chosen_model, revision=False)

    with console.status("[bold green]Writing code...", spinner="dots"):
        response = agent.invoke({"input": input_text})

    generated_code = response.content if hasattr(response, "content") else str(response)
    console.print("[bold green]  ✓ Code generation complete[/bold green]")

    attempt = state.get("code_attempt", 0) + 1
    return {
        "generated_code": generated_code,
        "chosen_llm_model": chosen_model,
        "revision_needed": False,
        "code_attempt": attempt,
    }


def write_code_to_disk(state: ManagerState):
    """Writes the approved code files to the project directory."""
    console.rule("[bold cyan]Writing Code Files[/bold cyan]")

    generated_code = state.get("generated_code", "")
    project_path = state.get("project_path", "outputs")
    folder = state.get("milestone_folder", "milestone_1")

    code_dir = os.path.join(project_path, "code")
    written = write_code_files(generated_code, code_dir)

    console.print(
        f"[bold green]  ✓ Written {len(written)} file(s) to {code_dir}[/bold green]"
    )
    for f in written:
        console.print(f"[dim]    {f}[/dim]")

    return {}


def generate_run_instructions(state: ManagerState):
    """Generates a how-to-run README for the milestone."""
    console.rule("[bold cyan]Generating Run Instructions[/bold cyan]")

    milestone = state.get("current_milestone", "")
    tech_stack = state.get("tech_stack", "")
    generated_code = state.get("generated_code", "")
    project_path = state.get("project_path", "outputs")
    folder = state.get("milestone_folder", "milestone_1")

    input_text = (
        f"## Milestone\n{milestone}\n\n"
        f"## Tech Stack\n{tech_stack}\n\n"
        f"## Generated Code\n{generated_code}"
    )

    agent = get_instructions_agent()
    with console.status("[bold green]Writing run instructions...", spinner="dots"):
        response = agent.invoke({"input": input_text})

    instructions = response.content if hasattr(response, "content") else str(response)

    save_file(f"{folder}/run_instructions.md", instructions, base_path=project_path)
    console.print(
        f"[bold green]  ✓ Run instructions saved to {project_path}/{folder}/run_instructions.md[/bold green]"
    )

    return {"run_instructions": instructions}


def user_run_review(state: ManagerState):
    """HITL: User runs the project and gives feedback."""
    console.rule("[bold magenta]User Run Review[/bold magenta]")

    project_path = state.get("project_path", "outputs")
    folder = state.get("milestone_folder", "milestone_1")
    instructions_path = f"{project_path}/{folder}/run_instructions.md"
    code_path = f"{project_path}/code/"

    console.print(
        Panel(
            f"[bold]Run Instructions:[/bold] {instructions_path}\n"
            f"[bold]Code Directory:[/bold]   {code_path}\n\n"
            f"Please follow the instructions to run the project, then come back here.",
            title="[bold blue]Your Project is Ready![/bold blue]",
            border_style="green",
        )
    )

    while True:
        decision = (
            console.input(
                "[bold yellow]Did the project run successfully? (yes/no): [/bold yellow]"
            )
            .strip()
            .lower()
        )
        if decision == "yes":
            console.print("[bold green]Great! Milestone complete ✓[/bold green]")

            current_milestone = state.get("current_milestone", "")
            milestones_text = state.get("milestones", "")
            if current_milestone in milestones_text:
                updated_milestone_desc = (
                    current_milestone.replace(" [ ] ", " [x] ")
                    if " [ ] " in current_milestone
                    else f"✅ {current_milestone}"
                )
                milestones_text = milestones_text.replace(
                    current_milestone, updated_milestone_desc
                )

                save_file("milestone.md", milestones_text, base_path=project_path)
                try:
                    markdown_to_pdf(
                        "milestone.md", "milestone.pdf", base_path=project_path
                    )
                except Exception as e:
                    pass
            else:
                updated_milestone_desc = current_milestone

            current_index = state.get("current_milestone_index", 0)
            all_milestones = state.get("all_milestones", [])

            proceed = True
            if current_index + 1 < len(all_milestones):
                ans = (
                    console.input(
                        "[bold cyan]Proceed to next milestone? (yes/no): [/bold cyan]"
                    )
                    .strip()
                    .lower()
                )
                proceed = ans == "yes"

            return {
                "revision_needed": False,
                "user_run_feedback": "",
                "milestones": milestones_text,
                "current_milestone_index": (
                    current_index + 1 if proceed else current_index
                ),
                "proceed_to_next": proceed,
            }
        elif decision == "no":
            feedback = console.input(
                "[bold cyan]Describe what went wrong: [/bold cyan]"
            ).strip()
            if not feedback:
                console.print(
                    "[bold red]No feedback provided. Please try again.[/bold red]"
                )
                continue
            console.print("[bold yellow]Sending feedback to coder...[/bold yellow]")
            return {
                "revision_needed": True,
                "user_run_feedback": feedback,
                "review_feedback": feedback,
                "code_attempt": 0,
            }
        else:
            console.print(
                "[bold red]Invalid input. Please enter 'yes' or 'no'.[/bold red]"
            )


def check_user_run_review(
    state: ManagerState,
) -> Literal["code_milestone", "pick_milestone", "__end__"]:
    if state.get("revision_needed"):
        return "code_milestone"

    if not state.get("proceed_to_next", True):
        console.print(
            "[bold green]Closing project generation as requested.[/bold green]"
        )
        return END

    milestones = state.get("all_milestones", [])
    current_index = state.get("current_milestone_index", 0)

    if current_index < len(milestones):
        console.print(
            f"[bold blue]Moving to Milestone {current_index + 1} / {len(milestones)}[/bold blue]"
        )
        return "pick_milestone"

    console.print("[bold green]All milestones completed![/bold green]")
    return END


class ManagerAgent:
    def __init__(self):
        workflow = StateGraph(ManagerState)

        workflow.add_node("call_project_lead", call_project_lead)
        workflow.add_node("human_review", human_review)
        workflow.add_node("call_milestone", call_milestone)
        workflow.add_node("call_flow_diagram", call_flow_diagram)
        workflow.add_node("call_tech_stack", call_tech_stack)
        workflow.add_node("tech_stack_review", tech_stack_review)

        workflow.add_node("pick_milestone", pick_milestone)
        workflow.add_node("manager_decision", manager_decision)
        workflow.add_node("consensus_review", consensus_review)

        available_llms = get_available_llms()
        proposal_node_names = []
        for llm_info in available_llms:
            node_name = f"propose_{llm_info['name']}"
            proposal_node_names.append(node_name)
            workflow.add_node(
                node_name, make_proposal_node(llm_info["name"], llm_info["model"])
            )

        workflow.add_edge(START, "call_project_lead")
        workflow.add_edge("call_project_lead", "human_review")
        workflow.add_conditional_edges("human_review", check_review)

        workflow.add_edge("call_milestone", "call_flow_diagram")
        workflow.add_edge("call_flow_diagram", "call_tech_stack")
        workflow.add_edge("call_tech_stack", "tech_stack_review")
        workflow.add_conditional_edges("tech_stack_review", check_tech_stack_review)

        for node_name in proposal_node_names:
            workflow.add_edge("pick_milestone", node_name)

        for node_name in proposal_node_names:
            workflow.add_edge(node_name, "manager_decision")

        workflow.add_edge("manager_decision", "consensus_review")
        workflow.add_conditional_edges("consensus_review", check_consensus_review)

        # --- Coding pipeline: generate → write → (loop or continue) ---
        workflow.add_node("code_milestone", code_milestone)
        workflow.add_node("write_code_to_disk", write_code_to_disk)
        workflow.add_node("generate_run_instructions", generate_run_instructions)
        workflow.add_node("user_run_review", user_run_review)

        workflow.add_edge("code_milestone", "write_code_to_disk")
        workflow.add_edge("write_code_to_disk", "generate_run_instructions")
        workflow.add_edge("generate_run_instructions", "user_run_review")
        workflow.add_conditional_edges("user_run_review", check_user_run_review)

        self.app = workflow.compile()

    def process_request(self, user_query: str, project_path: str):
        title = pyfiglet.figlet_format("dev-council", font="slant")
        console.print(Text(title, style="bold magenta"))

        console.rule("[bold blue]New Request[/bold blue]")
        console.print(f"[bold]Requests:[/bold] {user_query}")
        console.print(f"[bold]Project Path:[/bold] {project_path}")

        initial_state = {"input": user_query, "project_path": project_path}

        self.app.invoke(initial_state)

        console.rule("[bold green]Process Completed Successfully[/bold green]")


def get_manager():
    return ManagerAgent()
