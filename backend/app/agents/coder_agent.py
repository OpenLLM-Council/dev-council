"""
Production-grade Coder Agent for dev-council.

Uses LangGraph's ToolNode + tools_condition for reliable tool execution.
This eliminates JSON parsing issues (backslash escaping on Windows paths, etc.)
by letting LangGraph handle tool-call parsing and ToolMessage formatting natively.

Includes:
- Read deduplication cache (prevents context window overflow from repeated reads)
- History trimming (keeps context manageable for local LLMs)
- GuardedToolNode (intercepts duplicate read calls before execution)
"""
import os
import copy
from typing import TypedDict, Annotated, List
import operator

from rich.console import Console

console = Console()

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import settings

# Import all production tools
from app.tools.file_reader import (
    get_project_tree,
    search_in_project,
    search_in_file,
    get_file_length,
    read_whole_file,
    read_file_chunk,
    get_lines,
    read_file,
    read_directory,
)
from app.tools.file_editor import (
    delete_lines,
    insert_after_line,
    replace_in_file,
    multi_replace_in_file,
    write_file,
)
from app.tools.memory_tools import (
    update_memory,
    get_memory,
)
from app.tools.shell import run_shell, get_platform_info, start_background_process, stop_background_process


# ---------------------------------------------------------------------------
# Global state for tool initialization
# ---------------------------------------------------------------------------

_code_dir: str = ""
_memory_dir: str = ""


def set_directories(code_path: str, memory_path: str):
    """Called by manager.py to set the directories for tools that need absolute paths."""
    global _code_dir, _memory_dir
    _code_dir = code_path
    _memory_dir = memory_path


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

CODER_TOOLS = [
    get_project_tree,
    search_in_project,
    search_in_file,
    get_file_length,
    read_whole_file,
    read_file_chunk,
    get_lines,
    read_file,
    read_directory,
    delete_lines,
    insert_after_line,
    replace_in_file,
    multi_replace_in_file,
    write_file,
    update_memory,
    get_memory,
    run_shell,
    get_platform_info,
    start_background_process,
    stop_background_process,
]

_WRITE_TOOLS = {
    "write_file", "insert_after_line", "replace_in_file",
    "delete_lines", "multi_replace_in_file",
}

_READ_TOOLS = {
    "get_project_tree", "search_in_file", "get_file_length",
    "search_in_project",
    "read_whole_file", "read_file_chunk", "get_lines",
    "read_file", "read_directory",
}


# ---------------------------------------------------------------------------
# Read Deduplication Cache
# ---------------------------------------------------------------------------

class _ReadCache:
    """Tracks read-tool calls; duplicates get a terse notice instead of re-executing."""
    def __init__(self):
        self._seen: dict[str, str] = {}

    def _key(self, tool_name: str, args: dict) -> str:
        sorted_args = sorted((k, str(v)) for k, v in args.items())
        return f"{tool_name}::{sorted_args}"

    def check(self, tool_name: str, args: dict) -> str | None:
        if tool_name not in _READ_TOOLS:
            return None
        return self._seen.get(self._key(tool_name, args))

    def store(self, tool_name: str, args: dict, result: str):
        if tool_name in _READ_TOOLS:
            self._seen[self._key(tool_name, args)] = result

    def invalidate_for_path(self, filepath: str):
        """Remove cached reads that involve the given filepath."""
        # Normalize for comparison
        norm = filepath.replace("\\", "/").rstrip("/")
        to_remove = [
            k for k in self._seen
            if norm in k.replace("\\", "/")
        ]
        # Also invalidate get_project_tree (directory structure changed)
        to_remove += [k for k in self._seen if k.startswith("get_project_tree::")]
        for k in set(to_remove):
            self._seen.pop(k, None)

    def invalidate_tree(self):
        """Remove cached get_project_tree entries (after shell commands that may create files)."""
        to_remove = [k for k in self._seen if k.startswith("get_project_tree::")]
        for k in to_remove:
            self._seen.pop(k, None)

    def invalidate_all(self):
        """Clear all cached reads after commands that may mutate multiple files."""
        self._seen.clear()

    def reset(self):
        self._seen.clear()


_read_cache = _ReadCache()


# ---------------------------------------------------------------------------
# History trimming — keeps context manageable for local LLMs
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES = 40
TOOL_RESULT_TRUNCATE_CHARS = 500
TOOL_RESULT_RECENT_WINDOW = 10
MAX_AGENT_STEPS = 80


def _trim_history(messages: list) -> list:
    """Trim old messages and truncate old tool results to save context."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        recent = messages
    else:
        first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        recent = messages[-MAX_HISTORY_MESSAGES:]
        if first_human and first_human not in recent:
            recent = [first_human] + recent

    cutoff = len(recent) - TOOL_RESULT_RECENT_WINDOW
    result = []
    for i, msg in enumerate(recent):
        if i < cutoff and isinstance(msg, ToolMessage):
            content = str(msg.content)
            if len(content) > TOOL_RESULT_TRUNCATE_CHARS:
                msg = ToolMessage(
                    content=content[:TOOL_RESULT_TRUNCATE_CHARS] + " ... [truncated]",
                    tool_call_id=msg.tool_call_id,
                )
        result.append(msg)
    return result


# ---------------------------------------------------------------------------
# GuardedToolNode — intercepts duplicate read calls
# ---------------------------------------------------------------------------

class GuardedToolNode(ToolNode):
    """
    Wraps ToolNode to intercept and short-circuit duplicate read-tool calls.
    When the model tries to re-read something it already has, it gets back a
    firm "ALREADY READ" message, preventing context waste.
    """

    def invoke(self, state, config=None, **kwargs):
        messages = state.get("messages", [])
        last = messages[-1] if messages else None

        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            return super().invoke(state, config, **kwargs)

        patched_tool_messages = []
        non_duplicate_calls = []

        for call in last.tool_calls:
            name = call["name"]
            args = call["args"]
            cached = _read_cache.check(name, args)

            if cached is not None:
                console.print(f"[dim yellow]    ↳ {name}: duplicate blocked[/dim yellow]")
                notice = (
                    f"DUPLICATE CALL BLOCKED: '{name}' with these exact arguments "
                    f"was already called. The result is in your context above. "
                    f"Do NOT call this again — use the existing result."
                )
                patched_tool_messages.append(
                    ToolMessage(content=notice, tool_call_id=call["id"])
                )
            else:
                non_duplicate_calls.append(call)

        if not non_duplicate_calls:
            return {"messages": patched_tool_messages}

        try:
            if len(non_duplicate_calls) == len(last.tool_calls):
                result = super().invoke(state, config, **kwargs)
            else:
                patched_last = copy.copy(last)
                patched_last.tool_calls = non_duplicate_calls
                patched_state = dict(state)
                patched_state["messages"] = messages[:-1] + [patched_last]
                result = super().invoke(patched_state, config, **kwargs)
        except Exception as e:
            console.print(f"[bold red]    ✗ Tool execution error: {e}[/bold red]")
            error_messages = [
                ToolMessage(content=f"ERROR: {e}", tool_call_id=c["id"])
                for c in non_duplicate_calls
            ]
            return {"messages": patched_tool_messages + error_messages}

        new_messages = result.get("messages", [])
        for call, msg in zip(non_duplicate_calls, new_messages):
            content_str = str(msg.content)
            _read_cache.store(call["name"], call["args"], content_str)

            # Invalidate stale cache entries after writes
            if call["name"] in _WRITE_TOOLS:
                filepath = call["args"].get("filepath", call["args"].get("file_path", ""))
                if filepath:
                    _read_cache.invalidate_for_path(filepath)
            elif call["name"] == "run_shell":
                _read_cache.invalidate_all()
            elif call["name"] == "start_background_process":
                _read_cache.invalidate_tree()

            preview = content_str[:100].replace("\n", " ")
            if call["name"] in ("write_file", "insert_after_line", "replace_in_file", "delete_lines", "multi_replace_in_file"):
                filepath = call["args"].get("filepath", call["args"].get("file_path", "?"))
                console.print(f"[bold cyan]    ✓ {call['name']}[/bold cyan]: {filepath}")
            elif call["name"] == "run_shell":
                cmd = call["args"].get("command", "?")
                cwd = call["args"].get("cwd", "")
                console.print(f"[bold magenta]    $ {cmd}[/bold magenta]{f' (in {cwd})' if cwd else ''}")
                if preview:
                    console.print(f"[dim]      → {preview}[/dim]")
            elif call["name"] in ("start_background_process", "stop_background_process"):
                pname = call["args"].get("process_name", "?")
                console.print(f"[bold magenta]    ⚙ {call['name']}[/bold magenta]: {pname}")
                if preview:
                    console.print(f"[dim]      → {preview}[/dim]")
            else:
                console.print(f"[dim]    ↳ {call['name']}: {preview}[/dim]")

        all_messages = patched_tool_messages + new_messages
        return {"messages": all_messages}


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """You are a senior software developer implementing a project milestone.

You have access to tools for reading, writing, editing files, AND running shell commands.

══════════════════════════════════════════════════════
CRITICAL — USE SHELL COMMANDS FOR SETUP
══════════════════════════════════════════════════════

FIRST call `get_platform_info` to detect the OS and shell. Then use the correct
syntax for that platform when running shell commands with `run_shell`.
On Windows, `run_shell` executes through `cmd.exe`, not bash.

Before writing ANY code, use `run_shell` to scaffold projects and install dependencies.
Use `run_shell` only for setup, dependency installation, builds, tests, and scaffolding.

IMPORTANT: ALWAYS use non-interactive flags so commands never wait for input!
Examples of correct non-interactive commands:
- Next.js:    run_shell("npx create-next-app@latest my-app --yes --ts --tailwind --app --src-dir --import-alias '@/*' --eslint", cwd="...")
- React/Vite: run_shell("npm create vite@latest my-app -- --template react-ts", cwd="...")
- Express:    run_shell("npm init -y && npm install express", cwd="...")
- Python:     run_shell("pip install flask", cwd="...") or run_shell("uv init", cwd="...")
- Angular:    run_shell("npx @angular/cli new my-app --defaults --skip-git", cwd="...")
- SvelteKit:  run_shell("npm create svelte@latest my-app -- --template skeleton --types ts", cwd="...")
- Any CLI:    ALWAYS pass --yes, --default, --no-input, -y, or equivalent flags

NEVER run a scaffolding command without its non-interactive flags — it will hang forever.
NEVER use bash-only syntax on Windows such as `mkdir -p`, `pwd`, or brace expansion like `src/{a,b}`.
NEVER use `echo`, `>`, `>>`, `Add-Content`, `Set-Content`, `Out-File`, or `type nul`
to create/edit project files. Use `write_file`, `insert_after_line`, `replace_in_file`,
and `delete_lines` for all file content changes.

Do NOT manually create package.json, vite.config.ts, tsconfig.json, etc. — let the CLI generate them.
After scaffolding, use get_project_tree to see what was created, then write/edit only your custom code.

For long-running processes (dev servers, watchers):
- start_background_process("node server.js", "my-server", cwd="...") — starts in background
- stop_background_process("my-server") — sends Ctrl+C / terminates it
Do NOT use run_shell for servers — it will hang until timeout.

══════════════════════════════════════════════════════
PARADIGM 1 — MODIFY / DELETE existing code
══════════════════════════════════════════════════════

1. get_project_tree — see the directory layout
2. search_in_project(project_path, pattern) — find the right file when you do not know it yet
3. search_in_file(pattern, context_lines=3) — inspect the code to change
4. search_in_file(pattern, context_lines=0) — find all call sites if needed
5. EDIT:
   - Removing?  → delete_lines (pass ALL ranges in ONE call)
   - Renaming?  → multi_replace_in_file with all occurrences
   - Logic fix? → get_lines on the exact lines, then replace_in_file
5. DONE — write a brief summary of changes

══════════════════════════════════════════════════════
PARADIGM 2 — ADD new code (new file or new feature)
══════════════════════════════════════════════════════

1. run_shell — scaffold the project / install dependencies FIRST
2. get_project_tree — see what the scaffolding created
3. search_in_project(project_path, pattern) — find likely target files when needed
4. read_whole_file — read context (for files ≤ 300 lines)
   For files > 300 lines: get_file_length, then read_file_chunk in 70-line windows
5. Write the code:
   - New file → write_file(filepath, content)
   - Add to existing → insert_after_line(filepath, after_line, new_code)
6. Install any extra deps: run_shell("npm install axios react-router-dom", cwd="...")
7. Wire it up — add imports/calls using replace_in_file or insert_after_line
8. DONE — write a brief summary

══════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════
- Use forward slashes in file paths (e.g. C:/Users/user/project/src/app.py)
- filepath must point to a FILE, never a directory
- If a tool returns "DUPLICATE CALL BLOCKED", use the result already in your context
- After a successful edit, move on — do NOT re-read the file to verify
- Always use scalable folder architecture
- Use ONLY the approved tech stack from the milestone specification
- Write complete, production-ready code
- Create/update each file individually
- Save progress to memory using update_memory after major steps
"""

CODER_REVISION_PROMPT = """You are a senior software developer fixing code based on reviewer feedback.

You have access to tools for reading, writing, editing files, AND running shell commands.

## WORKFLOW

1. get_project_tree — see the current structure
2. If missing dependencies are flagged, use run_shell to install them first
3. For EACH issue in the feedback:
   a. search_in_project if you need to locate the file first, then search_in_file
   b. get_lines to get exact text for replacement
   c. Use replace_in_file, delete_lines, or insert_after_line to fix it
4. Save progress with update_memory
5. Write a brief summary of all changes

## RULES

- run_shell is only for setup/build/test commands, never for writing project files
- On Windows, run_shell executes through cmd.exe, so avoid bash-only syntax like mkdir -p, pwd, and brace expansion
- Never use echo/redirection/Add-Content/Set-Content/type nul to write code or config files
- Use forward slashes in file paths (e.g. C:/Users/user/project/src/app.py)
- filepath must point to a FILE, never a directory
- Address ALL reviewer feedback — do not skip anything
- Never rewrite entire files; use surgical, targeted edits only
- Copy exact text from get_lines output when building old_text for replace_in_file
- After a successful edit, move on — do NOT re-read the file to verify
- Use run_shell for any dependency installation or build commands
"""


# ---------------------------------------------------------------------------
# Agent Graph State
# ---------------------------------------------------------------------------

class _CoderState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

def get_coder_agent(model_name: str, revision: bool = False):
    """
    Build a LangGraph-based tool-calling coder agent.

    Uses ToolNode + tools_condition so LangGraph handles tool-call parsing,
    ToolMessage formatting, and the coder→tools→coder loop natively.
    This eliminates backslash/JSON parsing issues with Ollama models.

    Returns a CoderAgentExecutor with .invoke({"input": ...}) -> {"output": ...}
    """
    llm = ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_URL,
        num_predict=8192,
        timeout=300,
    )
    llm_with_tools = llm.bind_tools(CODER_TOOLS)

    system_prompt = CODER_REVISION_PROMPT if revision else CODER_SYSTEM_PROMPT

    _step = {"n": 0}

    # -- coder node: invokes LLM with trimmed history --
    def coder_node(state: _CoderState):
        _step["n"] += 1
        step = _step["n"]
        if step > MAX_AGENT_STEPS:
            console.print(
                f"[bold yellow]  Step limit reached ({MAX_AGENT_STEPS}); stopping likely tool loop[/bold yellow]"
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I stopped because I exceeded the safe tool-call limit, which usually "
                            "means I got stuck in a shell or file-edit loop. Please review the recent "
                            "tool outputs and continue with a smaller corrective prompt."
                        )
                    )
                ]
            }
        messages = state.get("messages", [])
        trimmed = _trim_history(messages)
        n_msgs = len(trimmed)
        console.print(f"[dim]  Step {step}: LLM thinking... ({n_msgs} messages in context)[/dim]")
        try:
            response = llm_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + trimmed
            )
        except Exception as e:
            error_msg = str(e)[:200]
            console.print(f"[bold red]  Step {step}: LLM error — {error_msg}[/bold red]")
            # Return a message asking the LLM to try again or wrap up
            response = AIMessage(content=f"I encountered an error calling the model: {error_msg}. Let me summarize what was done so far.")
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            for c in tool_calls:
                args_summary = ", ".join(f"{k}={repr(v)[:60]}" for k, v in c["args"].items())
                console.print(f"[bold blue]  Step {step}:[/bold blue] {c['name']}({args_summary})")
        else:
            snippet = (response.content or "")[:120].replace("\n", " ")
            console.print(f"[bold green]  Step {step}: LLM responded[/bold green] [dim]({snippet}...)[/dim]")
        return {"messages": [response]}

    # -- build the graph --
    builder = StateGraph(_CoderState)
    builder.add_node("coder", coder_node)
    builder.add_node("tools", GuardedToolNode(CODER_TOOLS))
    builder.set_entry_point("coder")

    builder.add_conditional_edges(
        "coder",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "coder")

    graph = builder.compile()

    # Reset dedup cache for each new agent build
    _read_cache.reset()

    return CoderAgentExecutor(graph)


class CoderAgentExecutor:
    """
    Thin wrapper around the compiled LangGraph so the manager can call
    agent.invoke({"input": text}) -> {"output": text} without changes.
    """

    def __init__(self, graph):
        self._graph = graph

    def invoke(self, input_dict: dict) -> dict:
        user_message = input_dict.get("input", "")
        initial_state = {"messages": [HumanMessage(content=user_message)]}

        try:
            final_state = self._graph.invoke(initial_state)
        except KeyboardInterrupt:
            console.print("[bold yellow]  ⚠ Interrupted by user[/bold yellow]")
            return {"output": "Agent interrupted by user."}
        except Exception as e:
            console.print(f"[bold red]  ✗ Agent error: {e}[/bold red]")
            return {"output": f"Agent encountered an error: {e}"}

        # Extract the last AI message as the output
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return {"output": msg.content}

        return {"output": "Agent completed without generating a final response."}
