#!/usr/bin/env python3
"""dev-council: a minimal BTP coding CLI powered by Ollama."""
from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import threading
import uuid
from datetime import datetime
from pathlib import Path

import checkpoint as ckpt
from agent import (
    AgentState,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
    run,
)
from compaction import estimate_tokens, get_context_limit, manual_compact
from config import (
    CONFIG_DIR,
    DAILY_DIR,
    DEFAULTS,
    MR_SESSION_DIR,
    SESSION_HIST_FILE,
    calc_cost,
    current_provider,
    load_config,
    save_config,
)
from context import build_system_prompt
from memory import load_index, search_memory
from mcp import (
    add_server_to_user_config,
    list_config_files,
    load_mcp_configs,
    remove_server_from_user_config,
)
from mcp.client import get_mcp_manager
from mcp.tools import refresh_server, reload_mcp
from providers import (
    PROVIDERS,
    bare_model,
    detect_provider,
    get_api_key,
    get_base_url,
    list_ollama_models,
    stream,
)
from skill.loader import find_skill, load_skills, substitute_arguments
from task import (
    clear_all_tasks,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    update_task,
)
from tools import ask_input_interactive


VERSION = "0.1.2"

C = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def clr(text: str, *keys: str) -> str:
    return "".join(C[key] for key in keys) + str(text) + C["reset"]


def info(message: str) -> None:
    print(clr(message, "cyan"))


def ok(message: str) -> None:
    print(clr(message, "green"))


def warn(message: str) -> None:
    print(clr(f"Warning: {message}", "yellow"))


def err(message: str) -> None:
    print(clr(f"Error: {message}", "red"), file=sys.stderr)


_scheduled_queries: list[str] = []
_scheduled_lock = threading.Lock()
_active_state: AgentState | None = None
_active_config: dict | None = None


def _ensure_utf8_stdio() -> None:
    """Wrap Windows stdio only during CLI execution, not at import time."""
    if sys.platform != "win32":
        return
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        if str(getattr(stream, "encoding", "") or "").lower() == "utf-8":
            continue
        setattr(sys, name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))


def _enqueue_system_query(query: str) -> None:
    with _scheduled_lock:
        _scheduled_queries.append(query)


def _drain_scheduled_queries() -> list[str]:
    with _scheduled_lock:
        pending = list(_scheduled_queries)
        _scheduled_queries.clear()
    return pending


def _btp_dir() -> Path:
    path = Path.cwd() / "btp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _council_dir() -> Path:
    path = _btp_dir() / "council"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_record(state: AgentState, session_id: str) -> dict:
    return {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "turn_count": state.turn_count,
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
        "messages": state.messages,
    }


def _save_history_snapshot(record: dict, latest_name: str = "session_latest.json") -> Path:
    MR_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = MR_SESSION_DIR / latest_name
    latest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return latest_path


def _save_daily_snapshot(record: dict) -> Path:
    day_dir = DAILY_DIR / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    suffix = record["session_id"][:8]
    filename = f"session_{datetime.now().strftime('%H%M%S')}_{suffix}.json"
    path = day_dir / filename
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _update_master_history(record: dict, config: dict) -> None:
    SESSION_HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = {"sessions": []}
    if SESSION_HIST_FILE.exists():
        try:
            history = json.loads(SESSION_HIST_FILE.read_text(encoding="utf-8"))
        except Exception:
            history = {"sessions": []}
    sessions = history.get("sessions", [])
    sessions.append(record)
    keep = int(config.get("session_history_limit", DEFAULTS["session_history_limit"]))
    history["sessions"] = sessions[-keep:]
    history["total_turns"] = sum(item.get("turn_count", 0) for item in history["sessions"])
    SESSION_HIST_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def save_session(state: AgentState, config: dict, session_id: str) -> None:
    record = _session_record(state, session_id)
    latest = _save_history_snapshot(record)
    daily = _save_daily_snapshot(record)
    _update_master_history(record, config)
    ok(f"Session saved -> {latest}")
    ok(f"              -> {daily}")


def _autosave_session() -> None:
    if _active_state is None or _active_config is None:
        return
    session_id = _active_config.get("_session_id", "")
    if not session_id:
        return
    try:
        save_session(_active_state, _active_config, session_id)
    except Exception:
        pass


def load_session_file(path: Path) -> AgentState | None:
    if not path.exists():
        err(f"Session file not found: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        err(f"Failed to load session: {exc}")
        return None

    state = AgentState()
    state.messages = payload.get("messages", [])
    state.turn_count = int(payload.get("turn_count", 0))
    state.total_input_tokens = int(payload.get("total_input_tokens", 0))
    state.total_output_tokens = int(payload.get("total_output_tokens", 0))
    return state


def _permission_prompt(description: str, config: dict) -> bool:
    if config.get("permission_mode") == "accept-all":
        return True
    raw = ask_input_interactive(clr(f"Allow this action? {description} [y/N] ", "yellow"), config)
    return raw.strip().lower() in {"y", "yes"}


def _print_tool_result(result: str, config: dict) -> None:
    if not result:
        return
    if "--- a/" in result and "+++ b/" in result:
        print(result)
        return

    lines = result.strip().splitlines()
    if not lines:
        return

    is_error = result.strip().startswith("Error")
    limit = 10 if not config.get("verbose") else 50
    preview_lines = lines[:limit]
    preview = "\n".join(preview_lines)

    if len(lines) > limit or len(preview) > 2000:
        preview = preview[:2000]
        if not preview.endswith("\n"):
            preview += "\n"
        preview += clr(f"  [...truncated {len(lines) - len(preview_lines)} lines...]", "dim")

    color = "red" if is_error else "dim"
    print(clr("  result:", color))
    print(textwrap.indent(preview, "    "))


def _context_usage(state: AgentState, config: dict) -> tuple[int, int, int]:
    used = estimate_tokens(state.messages)
    limit = get_context_limit(config.get("model", DEFAULTS["model"]))
    percent = min(999, int((used / limit) * 100)) if limit else 0
    return used, limit, percent


def _context_footer(state: AgentState, config: dict) -> str:
    used, limit, percent = _context_usage(state, config)
    return f"[Context: {percent}% used | ~{used:,} / {limit:,} tokens]"


def _print_context_footer(state: AgentState, config: dict) -> None:
    print(clr(_context_footer(state, config), "dim"))


def _auto_compaction_notice(percent: int) -> None:
    print(clr(f"⚠️ Context compaction triggered automatically (usage: {percent}%)", "yellow"))


def _print_banner() -> None:
    banner = r"""
     ____  _______     __       ____ ___  _   _ _   _  ____ ___ _
    |  _ \| ____\ \   / /      / ___/ _ \| | | | \ | |/ ___|_ _| |
    | | | |  _|  \ \ / /_____ | |  | | | | | | |  \| | |    | || |
    | |_| | |___  \ V /|_____| | |__| |_| | |_| | |\  | |___ | || |___
    |____/|_____|  \_/          \____\___/ \___/|_| \_|\____|___|_____|
"""
    print(clr(banner.rstrip(), "cyan", "bold"))
    print(clr(f"dev-council {VERSION}", "cyan", "bold"))
    print(clr("BTP stages: SRS -> Tech Stack -> Code -> QA -> Deployment", "dim"))
    print(clr("Use /model to choose Single LLM or Consensus mode. Press Ctrl+C to exit.", "dim"))


def _exit_on_interrupt() -> None:
    print()
    ok("Exiting dev-council.")


def _run_agent_query(
    query: str,
    state: AgentState,
    config: dict,
    model_override: str = "",
    quiet: bool = False,
    use_skills: bool = False,
) -> str:
    effective_config = dict(config)
    if model_override:
        effective_config["model"] = model_override
    effective_config["_run_query_callback"] = _enqueue_system_query
    effective_config["_auto_compact_notice"] = _auto_compaction_notice
    if use_skills:
        query, _ = _apply_skill_context(query, announce=not quiet, force_coding=True)
    system_prompt = build_system_prompt(effective_config)
    response_parts: list[str] = []

    for event in run(query, state, effective_config, system_prompt):
        if isinstance(event, TextChunk):
            response_parts.append(event.text)
            if not quiet:
                print(event.text, end="", flush=True)
        elif isinstance(event, ThinkingChunk):
            if effective_config.get("verbose") and not quiet:
                print(clr(event.text, "dim"), end="", flush=True)
        elif isinstance(event, ToolStart):
            if not quiet:
                print()
                target = (
                    event.inputs.get("file_path")
                    or event.inputs.get("path")
                    or event.inputs.get("command")
                    or event.inputs.get("target")
                    or event.inputs.get("query")
                )
                if target:
                    info(f"[tool] {event.name}: {target}")
                else:
                    info(f"[tool] {event.name}")

                if effective_config.get("verbose"):
                    params_str = json.dumps(event.inputs, ensure_ascii=False)
                    print(clr(f"  parameters: {params_str}", "dim"))
        elif isinstance(event, ToolEnd):
            if not quiet:
                _print_tool_result(event.result, effective_config)
        elif isinstance(event, PermissionRequest):
            event.granted = _permission_prompt(event.description, effective_config)
        elif isinstance(event, TurnDone):
            if effective_config.get("verbose") and not quiet:
                print()
                info(f"[tokens] input={event.input_tokens} output={event.output_tokens}")

    if not quiet:
        print()
    return "".join(response_parts)


def _run_text_prompt(
    prompt: str,
    config: dict,
    model: str = "",
    system: str = "",
    use_skills: bool = False,
    announce_skills: bool = False,
) -> str:
    prompt_model = model or config["model"]
    prompt_system = system or "You are dev-council. Respond with clean Markdown only."
    if use_skills:
        prompt, _ = _apply_skill_context(prompt, announce=announce_skills, force_coding=True)
    text_parts: list[str] = []
    llm_config = dict(config)
    llm_config["no_tools"] = True
    for event in stream(
        model=prompt_model,
        system=prompt_system,
        messages=[{"role": "user", "content": prompt}],
        tool_schemas=[],
        config=llm_config,
    ):
        if hasattr(event, "text"):
            text_parts.append(event.text)
    return "".join(text_parts).strip()


def _run_generation_prompt(prompt: str, config: dict, system: str = "") -> str:
    if config.get("llm_mode") != "consensus":
        return _run_text_prompt(prompt, config, system=system)

    selected_models = list(config.get("consensus_models") or [])
    if not selected_models:
        warn("Consensus mode has no models selected; falling back to active single model.")
        return _run_text_prompt(prompt, config, system=system)

    proposals: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for index, model_name in enumerate(selected_models, 1):
        info(f"Consensus prompt {index}/{len(selected_models)}: {model_name}")
        try:
            proposal = _run_text_prompt(prompt, config, model=model_name, system=system)
            proposals.append((model_name, proposal))
        except Exception as exc:
            failures.append((model_name, str(exc)))
            warn(f"Consensus model failed: {model_name} -> {exc}")

    if not proposals:
        details = "; ".join(f"{model_name}: {error}" for model_name, error in failures)
        raise RuntimeError(f"All consensus models failed for this prompt. {details}")

    if len(proposals) == 1:
        if failures:
            warn("Continuing with the only successful consensus model response.")
        return proposals[0][1]

    synthesis_prompt = ["Synthesize these model responses into one final output."]
    synthesis_prompt.append(f"Original prompt:\n{prompt}\n")
    for model_name, proposal in proposals:
        synthesis_prompt.append(f"[{model_name}]\n{proposal}\n")
    synthesis_prompt.append("Return only the final synthesized answer.")
    # Use the judge model for synthesis if configured, otherwise first successful proposer
    judge = config.get("judge_model") or proposals[0][0]
    info(f"Synthesizing consensus with {judge}")
    return _run_text_prompt(
        "\n".join(synthesis_prompt),
        config,
        model=judge,
        system="You are a consensus editor. Produce the final answer only.",
    )


def _project_snapshot(limit: int = 200) -> str:
    files: list[str] = []
    for path in sorted(Path.cwd().rglob("*")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file():
            files.append(str(path.relative_to(Path.cwd())))
        if len(files) >= limit:
            break
    return "\n".join(files)


def _stage_context(extra: str = "") -> str:
    sections = []
    btp = _btp_dir()
    for name in ["srs.md", "milestones.md", "tech_stack.md", "qa_report.md", "deployment_plan.md"]:
        path = btp / name
        if path.exists():
            sections.append(f"[{name}]\n{path.read_text(encoding='utf-8')[:8000]}")
    if extra:
        sections.append(f"[user]\n{extra}")
    return "\n\n".join(sections).strip()


def _tokenize(text: str) -> set[str]:
    cleaned = []
    current = []
    for char in text.lower():
        if char.isalnum() or char in {"-", "_"}:
            current.append(char)
        else:
            if current:
                cleaned.append("".join(current))
                current = []
    if current:
        cleaned.append("".join(current))
    return {token for token in cleaned if len(token) >= 3}


def _select_relevant_skills(query: str, force_coding: bool = True) -> list:
    query_terms = _tokenize(query)
    scored: list[tuple[int, object]] = []
    coding_skill_names = {
        "implementation-core",
        "frontend-builder",
        "backend-builder",
        "fullstack-builder",
        "testing-guard",
    }

    frontend_terms = {"frontend", "react", "next", "html", "css", "ui", "ux", "landing", "tailwind"}
    backend_terms = {"backend", "python", "api", "server", "database", "auth", "fastapi", "django", "flask", "sql"}
    fullstack_terms = {"saas", "fullstack", "full-stack", "dashboard", "platform", "product", "portal", "crm"}
    testing_terms = {"test", "tests", "testing", "qa", "pytest", "unit", "integration", "e2e", "bug", "fix"}

    for skill in load_skills():
        if force_coding and skill.name not in coding_skill_names:
            continue
        corpus = " ".join(
            [
                skill.name,
                skill.description,
                skill.when_to_use,
                " ".join(skill.triggers),
            ]
        )
        skill_terms = _tokenize(corpus)
        score = len(query_terms & skill_terms)

        if skill.name == "implementation-core" and force_coding:
            score += 3
        if query_terms & frontend_terms and skill.name == "frontend-builder":
            score += 4
        if query_terms & backend_terms and skill.name == "backend-builder":
            score += 4
        if query_terms & fullstack_terms and skill.name == "fullstack-builder":
            score += 4
        if query_terms & testing_terms and skill.name == "testing-guard":
            score += 4

        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda item: (-item[0], item[1].name))
    selected = []
    seen = set()
    for _, skill in scored:
        if skill.name in seen:
            continue
        selected.append(skill)
        seen.add(skill.name)
        if len(selected) >= 3:
            break

    if force_coding and "implementation-core" not in seen:
        for skill in load_skills():
            if skill.name == "implementation-core":
                selected.insert(0, skill)
                break

    return selected[:3]


def _apply_skill_context(query: str, announce: bool = True, force_coding: bool = True) -> tuple[str, list[str]]:
    skills = _select_relevant_skills(query, force_coding=force_coding)
    if not skills:
        return query, []

    names = [skill.name for skill in skills]
    if announce:
        info(f"[skills] Using: {', '.join(names)}")

    rendered = []
    for skill in skills:
        rendered_prompt = substitute_arguments(skill.prompt, query, skill.arguments)
        rendered.append(f"[Auto-applied skill: {skill.name}]\n{rendered_prompt}")

    enriched_query = (
        f"{query}\n\n"
        "Apply the following skill guidance while working:\n\n"
        + "\n\n".join(rendered)
    )
    return enriched_query, names


def _project_is_effectively_empty() -> bool:
    ignored = {"btp", "__pycache__", "node_modules", ".git"}
    for child in Path.cwd().iterdir():
        if child.name.startswith("."):
            continue
        if child.name in ignored:
            continue
        return False
    return True


def _looks_like_large_product_request(query: str) -> bool:
    lowered = query.lower()
    words = re.findall(r"\b[\w-]+\b", lowered)
    build_terms = (
        "build",
        "create",
        "develop",
        "make a full",
        "implement a system",
    )
    product_terms = (
        "saas",
        "full stack",
        "fullstack",
        "full-stack",
        "full system",
        "dashboard",
        "platform",
        "portal",
        "marketplace",
        "crm",
        "erp",
        "admin panel",
        "web app",
        "application",
        "product",
        "system",
        "feature set",
    )
    has_build_term = any(term in lowered for term in build_terms)
    has_product_term = any(term in lowered for term in product_terms)
    return (has_build_term and has_product_term) or (len(words) > 30 and has_product_term)


def _should_apply_skill_context(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return False

    informational_starts = (
        "read ",
        "show ",
        "tell ",
        "explain ",
        "summarize ",
        "what ",
        "why ",
        "how ",
        "list ",
        "find ",
        "search ",
        "inspect ",
        "review ",
    )
    mutation_terms = (
        "add",
        "build",
        "change",
        "code",
        "create",
        "debug",
        "develop",
        "edit",
        "fix",
        "implement",
        "refactor",
        "remove",
        "repair",
        "test",
        "update",
        "write",
    )

    if lowered.startswith(informational_starts) and not any(term in lowered for term in mutation_terms):
        return False
    return any(term in lowered for term in mutation_terms)


_STAGE_SPECS = {
    "srs": {
        "file": "srs.md",
        "title": "Software Requirements Specification",
        "prompt": (
            "Create a Software Requirements Specification for this request.\n\n"
            "{context}\n\n"
            "Include: overview, goals, actors, functional requirements, non-functional "
            "requirements, assumptions, risks, success criteria, and acceptance criteria."
        ),
    },
    "milestones": {
        "file": "milestones.md",
        "title": "Milestone Plan",
        "prompt": (
            "Create a phased milestone plan for this project context.\n\n"
            "{context}\n\n"
            "Define deliverables, dependencies, sequencing, and done criteria for each milestone."
        ),
    },
    "techstack": {
        "file": "tech_stack.md",
        "title": "Technology Stack Options",
        "prompt": (
            "Generate exactly 2 or 3 distinct technology stack options for this project context.\n\n"
            "{context}\n\n"
            "Use this exact structure for each option:\n"
            "Option N - <name>\n"
            "Frontend: <frontend>\n"
            "Backend: <backend>\n"
            "DB: <database>\n"
            "Deploy: <deployment target>\n"
            "Why: <one-line rationale>\n"
        ),
    },
    "qa": {
        "file": "qa_report.md",
        "title": "QA Strategy and Report",
        "prompt": (
            "Create a QA strategy and assessment for this project context.\n\n"
            "{context}\n\n"
            "Include unit tests, integration tests, end-to-end checks, performance validation, "
            "security review, gaps, and recommended fixes."
        ),
    },
    "deploy": {
        "file": "deployment_plan.md",
        "title": "Deployment and CI/CD Plan",
        "prompt": (
            "Create a deployment and CI/CD plan for this project context.\n\n"
            "{context}\n\n"
            "Include build pipeline, environments, secrets, release flow, rollback, monitoring, "
            "and production-readiness gates."
        ),
    },
}


def _write_stage_file(stage_name: str, content: str) -> Path:
    spec = _STAGE_SPECS[stage_name]
    path = _btp_dir() / spec["file"]
    if _active_config and _active_config.get("_session_id"):
        ckpt.track_file_edit(_active_config["_session_id"], str(path))
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _run_stage(stage_name: str, user_text: str, config: dict) -> Path:
    spec = _STAGE_SPECS[stage_name]
    context = _stage_context(user_text)
    prompt = spec["prompt"].format(context=context or user_text)
    system = f"You are dev-council working on BTP stage output: {spec['title']}."
    result = _run_generation_prompt(prompt, config, system=system)
    path = _write_stage_file(stage_name, result)
    ok(f"Wrote {path}")
    return path


def _parse_tech_stack_options(text: str) -> list[dict[str, str]]:
    chunks = re.split(r"(?im)^\s*Option\s+\d+\s*(?:[-—:])\s*", text)
    options: list[dict[str, str]] = []
    for chunk in chunks[1:]:
        lines = [line.strip() for line in chunk.strip().splitlines() if line.strip()]
        if not lines:
            continue
        option = {"name": lines[0].strip("* ")}
        for line in lines[1:]:
            key, sep, value = line.partition(":")
            if not sep:
                continue
            normalized = key.strip().lower()
            if normalized in {"frontend", "backend", "db", "database", "deploy", "deployment", "why"}:
                if normalized == "database":
                    normalized = "db"
                if normalized == "deployment":
                    normalized = "deploy"
                option[normalized] = value.strip()
        required = {"name", "frontend", "backend", "db", "deploy", "why"}
        if required <= set(option):
            options.append(option)
    return options[:3]


def _format_tech_stack_option(index: int, option: dict[str, str]) -> str:
    return "\n".join(
        [
            f"Option {index} - {option['name']}",
            f"Frontend: {option['frontend']}",
            f"Backend: {option['backend']}",
            f"DB: {option['db']}",
            f"Deploy: {option['deploy']}",
            f"Why: {option['why']}",
        ]
    )


def _run_tech_stack_selection(user_text: str, config: dict) -> Path:
    spec = _STAGE_SPECS["techstack"]
    context = _stage_context(user_text)
    prompt = spec["prompt"].format(context=context or user_text)
    result = _run_generation_prompt(
        prompt,
        config,
        system="You are dev-council preparing concise stack options for a user decision.",
    )
    options = _parse_tech_stack_options(result)
    if not (2 <= len(options) <= 3):
        warn("The model did not return 2-3 parseable stack options; using safe default options for selection.")
        options = [
            {
                "name": "FastAPI + React",
                "frontend": "React + Vite + Tailwind",
                "backend": "Python + FastAPI",
                "db": "PostgreSQL",
                "deploy": "Docker + Railway",
                "why": "Fast to build, easy to test, and suitable for most full-stack apps.",
            },
            {
                "name": "Next.js Full Stack",
                "frontend": "Next.js + Tailwind",
                "backend": "Next.js API routes/server actions",
                "db": "PostgreSQL",
                "deploy": "Vercel + managed Postgres",
                "why": "Best fit when frontend velocity and simple deployment matter most.",
            },
            {
                "name": "MERN Stack",
                "frontend": "React + Tailwind",
                "backend": "Node.js + Express",
                "db": "MongoDB",
                "deploy": "Docker + Railway",
                "why": "Flexible JavaScript stack for rapid product prototyping.",
            },
        ]

    print()
    info("Choose a technology stack before coding:")
    for index, option in enumerate(options, 1):
        print(_format_tech_stack_option(index, option))
        print()

    while True:
        raw = ask_input_interactive("Select tech stack option number: ", config).strip()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                selected = _format_tech_stack_option(index + 1, options[index])
                path = _write_stage_file("techstack", selected)
                ok(f"Selected {options[index]['name']} -> {path}")
                return path
        err(f"Choose a number between 1 and {len(options)}.")


def _select_endpoint(config: dict, purpose: str) -> str:
    options = [("local", "Use the local Ollama server")]
    if config.get("ollama_cloud_base_url"):
        options.append(("cloud", "Use the configured Ollama cloud endpoint"))

    if len(options) == 1:
        return options[0][0]

    print()
    info(f"Choose an Ollama endpoint for {purpose}:")
    for idx, (name, description) in enumerate(options, 1):
        print(f"  [{idx}] {name} - {description}")

    while True:
        raw = ask_input_interactive("Select endpoint number: ", config).strip()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index][0]
        err("Invalid endpoint selection.")


def _fetch_models_for_endpoint(endpoint: str, config: dict) -> list[str]:
    if endpoint == "local":
        models = _fetch_local_models_from_ollama_list()
        if models:
            return models
    base_url = get_base_url(endpoint, config)
    api_key = get_api_key(endpoint, config)
    models = list_ollama_models(base_url, api_key=api_key)
    if not models:
        raise RuntimeError(f"No models found at {base_url}/api/tags")
    return models


def _fetch_local_models_from_ollama_list() -> list[str]:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    models: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if not parts or parts[0].lower() == "name":
            continue
        models.append(parts[0])
    return models


def _choose_single_model(config: dict, endpoint_hint: str = "") -> str:
    endpoint = endpoint_hint or _select_endpoint(config, "single-model coding")
    models = _fetch_models_for_endpoint(endpoint, config)

    print()
    info(f"Available models on {endpoint}:")
    for idx, model_name in enumerate(models, 1):
        print(f"  [{idx:2d}] {model_name}")

    while True:
        raw = ask_input_interactive("Select model number: ", config).strip()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(models):
                selected = f"{endpoint}/{models[index]}"
                config["model"] = selected
                config["active_model"] = selected
                config["llm_mode"] = "single"
                config["active_ollama_endpoint"] = endpoint
                save_config(config)
                ok(f"Model set to {selected}")
                return selected
        err("Invalid model selection.")


def _choose_multiple_models(config: dict) -> list[str]:
    endpoint = _select_endpoint(config, "council consensus")
    models = _fetch_models_for_endpoint(endpoint, config)

    print()
    info(f"Available models on {endpoint}:")
    for idx, model_name in enumerate(models, 1):
        print(f"  [{idx:2d}] {model_name}")

    while True:
        raw_count = ask_input_interactive("How many models should vote? ", config).strip()
        if raw_count.isdigit() and 1 <= int(raw_count) <= len(models):
            count = int(raw_count)
            break
        err(f"Choose a number between 1 and {len(models)}.")

    while True:
        raw = ask_input_interactive(
            f"Select {count} model numbers (comma separated): ",
            config,
        ).strip()
        try:
            indexes = [int(part.strip()) - 1 for part in raw.split(",") if part.strip()]
        except ValueError:
            indexes = []
        unique_indexes = []
        for index in indexes:
            if index not in unique_indexes:
                unique_indexes.append(index)
        if len(unique_indexes) != count or any(index < 0 or index >= len(models) for index in unique_indexes):
            err("Invalid council selection.")
            continue
        selected = [f"{endpoint}/{models[index]}" for index in unique_indexes]
        config["llm_mode"] = "consensus"
        config["consensus_models"] = selected
        config["model"] = selected[0]
        config["active_ollama_endpoint"] = endpoint
        save_config(config)
        ok(f"Consensus models set: {', '.join(selected)}")
        return selected


def _select_model_mode(config: dict) -> str:
    print()
    info("Do you want to use a Single LLM or Consensus mode?")
    print("  [1] Single LLM")
    print("  [2] Consensus")
    while True:
        raw = ask_input_interactive("Select mode number: ", config).strip().lower()
        if raw in {"1", "single", "single llm"}:
            return "single"
        if raw in {"2", "consensus"}:
            return "consensus"
        err("Choose 1 for Single LLM or 2 for Consensus.")


def _run_model_selection_flow(config: dict) -> None:
    mode = _select_model_mode(config)
    if mode == "single":
        _choose_single_model(config)
        return
    selected = _choose_multiple_models(config)

    # Ask if the user wants a separate judge/synthesis model
    print()
    info("Use a separate judge model for synthesis & implementation?")
    print("  The judge model synthesizes proposals and executes the code stage.")
    print("  If skipped, the first consensus model is used as the judge.")
    raw = ask_input_interactive("Pick a judge model? [y/N]: ", config).strip().lower()
    if raw in {"y", "yes"}:
        endpoint = config.get("active_ollama_endpoint", "local")
        models = _fetch_models_for_endpoint(endpoint, config)
        print()
        info(f"Available models on {endpoint}:")
        for idx, model_name in enumerate(models, 1):
            print(f"  [{idx:2d}] {model_name}")
        while True:
            raw_idx = ask_input_interactive("Select judge model number: ", config).strip()
            if raw_idx.isdigit():
                index = int(raw_idx) - 1
                if 0 <= index < len(models):
                    judge = f"{endpoint}/{models[index]}"
                    config["judge_model"] = judge
                    save_config(config)
                    ok(f"Judge model set to {judge}")
                    break
            err("Invalid model selection.")
    else:
        config.pop("judge_model", None)


def _run_council(task_text: str, state: AgentState, config: dict) -> None:
    if not task_text.strip():
        task_text = ask_input_interactive("Council task: ", config).strip()
    if not task_text:
        return

    selected_models = _choose_multiple_models(config)
    council_root = _council_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
    council_root.mkdir(parents=True, exist_ok=True)

    snapshot = _project_snapshot()
    proposals: list[tuple[str, str]] = []

    for index, model_name in enumerate(selected_models, 1):
        info(f"Collecting proposal {index}/{len(selected_models)} from {model_name}")
        proposal_prompt = textwrap.dedent(
            f"""
            You are one model in a coding council for the following task:

            {task_text}

            Project file snapshot:
            {snapshot}

            Existing BTP context:
            {_stage_context()}

            Produce a concise implementation proposal with these sections:
            1. Summary
            2. Files to change
            3. Approach
            4. Risks
            5. Test plan
            """
        ).strip()
        proposal = _run_text_prompt(
            proposal_prompt,
            config,
            model=model_name,
            system="You are a senior software engineer proposing one candidate solution.",
            use_skills=True,
        )
        proposal_path = council_root / f"proposal_{index}_{bare_model(model_name).replace(':', '_')}.md"
        proposal_path.write_text(proposal + "\n", encoding="utf-8")
        proposals.append((model_name, proposal))

    synthesis_model = config.get("judge_model") or selected_models[0]
    synthesis_prompt = ["Synthesize these model proposals into one consensus brief."]
    synthesis_prompt.append(f"Original task:\n{task_text}\n")
    for model_name, proposal in proposals:
        synthesis_prompt.append(f"[{model_name}]\n{proposal}\n")
    synthesis_prompt.append(
        "Return a single consensus brief with sections: Summary, Agreed Changes, "
        "Tradeoffs, Risks, and Tests."
    )
    consensus = _run_text_prompt(
        "\n".join(synthesis_prompt),
        config,
        model=synthesis_model,
        system="You are a consensus editor for a coding council.",
        use_skills=True,
    )
    consensus_path = council_root / "consensus.md"
    consensus_path.write_text(consensus + "\n", encoding="utf-8")
    ok(f"Council consensus saved to {consensus_path}")

    implementation_prompt = textwrap.dedent(
        f"""
        Implement the user's request using this council consensus.

        User request:
        {task_text}

        Council consensus:
        {consensus}

        Work inside the current repository only.
        Keep the runtime aligned with dev-council and the BTP workflow.
        """
    ).strip()
    info(f"Implementing with synthesis model {synthesis_model}")
    _run_agent_query(implementation_prompt, state, config, model_override=synthesis_model)


def _run_consensus_agent_query(task_text: str, state: AgentState, config: dict) -> None:
    selected_models = list(config.get("consensus_models") or [])
    if len(selected_models) <= 1:
        model = selected_models[0] if selected_models else ""
        _run_agent_query(task_text, state, config, model_override=model, use_skills=True)
        return

    proposals: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for index, model_name in enumerate(selected_models, 1):
        info(f"Collecting implementation proposal {index}/{len(selected_models)} from {model_name}")
        try:
            proposal = _run_text_prompt(
                task_text,
                config,
                model=model_name,
                system="You are one model in a coding consensus. Propose the implementation approach.",
                use_skills=True,
            )
            proposals.append((model_name, proposal))
        except Exception as exc:
            failures.append((model_name, str(exc)))
            warn(f"Consensus model failed: {model_name} -> {exc}")

    if not proposals:
        details = "; ".join(f"{model_name}: {error}" for model_name, error in failures)
        raise RuntimeError(f"All consensus models failed for implementation planning. {details}")

    synthesis_prompt = ["Synthesize these implementation proposals into one executable brief."]
    synthesis_prompt.append(f"Original task:\n{task_text}\n")
    for model_name, proposal in proposals:
        synthesis_prompt.append(f"[{model_name}]\n{proposal}\n")
    synthesis_prompt.append("Return concise implementation instructions with agreed changes, risks, and tests.")
    # Use judge model for synthesis and implementation, or fall back to first successful proposer
    judge = config.get("judge_model") or proposals[0][0]
    info(f"Synthesizing implementation consensus with {judge}")
    consensus = _run_text_prompt(
        "\n".join(synthesis_prompt),
        config,
        model=judge,
        system="You are a consensus editor for implementation planning.",
        use_skills=True,
    )
    info(f"Starting implementation with {judge}")
    _run_agent_query(
        f"{task_text}\n\nConsensus implementation brief:\n{consensus}",
        state,
        config,
        model_override=judge,
        use_skills=True,
    )


# ── Pipeline checkpoint system ───────────────────────────────────────────────

_PIPELINE_STATE_FILE = "btp/.pipeline_state.json"

# Ordered list of pipeline stages — "code" is the implementation step
_PIPELINE_STAGES = ["srs", "techstack", "code", "qa", "deploy"]


def _pipeline_state_path() -> Path:
    return Path.cwd() / _PIPELINE_STATE_FILE


def _save_pipeline_state(
    query: str,
    completed_stages: list[str],
    config: dict,
) -> None:
    """Persist current pipeline progress so it can be resumed after a crash."""
    data = {
        "query": query,
        "completed_stages": completed_stages,
        "llm_mode": config.get("llm_mode", "single"),
        "model": config.get("model", ""),
        "consensus_models": config.get("consensus_models", []),
        "judge_model": config.get("judge_model", ""),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _pipeline_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_pipeline_state() -> dict | None:
    """Load a previously saved pipeline state, or None if no checkpoint exists."""
    path = _pipeline_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_pipeline_state() -> None:
    path = _pipeline_state_path()
    if path.exists():
        path.unlink()


def _run_full_btp_cycle(
    query: str,
    state: AgentState,
    config: dict,
    resume_from: list[str] | None = None,
) -> None:
    """Run the full BTP pipeline with per-stage checkpointing.

    If *resume_from* is given it is the list of already-completed stage names
    and those stages will be skipped.
    """
    completed: list[str] = list(resume_from) if resume_from else []

    if completed:
        remaining = [s for s in _PIPELINE_STAGES if s not in completed]
        info(f"Resuming BTP pipeline from stage: {remaining[0] if remaining else 'done'}")
        info(f"  Already completed: {', '.join(completed)}")
    else:
        info("Running full BTP cycle: SRS -> Tech Stack -> Code -> QA -> Deployment")

    # ── Stage: SRS ──────────────────────────────────────────────────────
    if "srs" not in completed:
        try:
            _run_stage("srs", query, config)
            completed.append("srs")
            _save_pipeline_state(query, completed, config)
        except Exception as exc:
            _save_pipeline_state(query, completed, config)
            err(f"SRS stage failed: {exc}")
            info(f"Pipeline checkpoint saved. Run '/pipeline resume' to retry from this stage.")
            return

    # ── Stage: Tech Stack ───────────────────────────────────────────────
    if "techstack" not in completed:
        try:
            _run_tech_stack_selection(_stage_context(query), config)
            completed.append("techstack")
            _save_pipeline_state(query, completed, config)
        except Exception as exc:
            _save_pipeline_state(query, completed, config)
            err(f"Tech Stack stage failed: {exc}")
            info(f"Pipeline checkpoint saved. Run '/pipeline resume' to retry from this stage.")
            return

    # ── Stage: Code (implementation) ────────────────────────────────────
    if "code" not in completed:
        implementation_prompt = textwrap.dedent(
            f"""
            BUILD the requested product NOW by creating actual files in this repository.

            User request:
            {query}

            BTP planning context:
            {_stage_context(query)}

            CRITICAL INSTRUCTIONS — you MUST follow these:
            1. Use the Write tool to create each source file. Do NOT describe code — write it to files.
            2. Use the Bash tool to run shell commands (e.g. npm init, pip install, mkdir).
               The tool name is "Bash" (capital B), NOT "bash".
            3. Use the Edit tool to modify existing files.
            4. Do NOT call the Skill tool during implementation — skills are already applied.
            5. Do NOT just plan or describe — you must produce working files.
            6. Use non-interactive commands only. Pass `--yes` or `-y` whenever supported.
            7. After creating files, verify by reading them back or running tests.

            Available tools for this task: Write, Bash, Edit, Read, Glob, Grep.
            Start by creating the project structure, then write each file.
            """
        ).strip()
        try:
            if config.get("llm_mode") == "consensus":
                _run_consensus_agent_query(implementation_prompt, state, config)
            else:
                _run_agent_query(implementation_prompt, state, config, use_skills=True)
            completed.append("code")
            _save_pipeline_state(query, completed, config)
        except Exception as exc:
            _save_pipeline_state(query, completed, config)
            err(f"Code stage failed: {exc}")
            info(f"Pipeline checkpoint saved. Run '/pipeline resume' to retry from this stage.")
            return

    # ── Stage: QA ───────────────────────────────────────────────────────
    if "qa" not in completed:
        try:
            _run_stage("qa", _stage_context(query), config)
            completed.append("qa")
            _save_pipeline_state(query, completed, config)
        except Exception as exc:
            _save_pipeline_state(query, completed, config)
            err(f"QA stage failed: {exc}")
            info(f"Pipeline checkpoint saved. Run '/pipeline resume' to retry from this stage.")
            return

    # ── Stage: Deploy ───────────────────────────────────────────────────
    if "deploy" not in completed:
        try:
            _run_stage("deploy", _stage_context(query), config)
            completed.append("deploy")
            _save_pipeline_state(query, completed, config)
        except Exception as exc:
            _save_pipeline_state(query, completed, config)
            err(f"Deploy stage failed: {exc}")
            info(f"Pipeline checkpoint saved. Run '/pipeline resume' to retry from this stage.")
            return

    # All stages done — clean up the checkpoint file
    _clear_pipeline_state()
    ok("All BTP pipeline stages completed successfully.")


def cmd_help(_args: str, _state: AgentState, _config: dict) -> bool:
    help_text = """
/model
  Switch between Single LLM and Consensus mode. Single mode stores active_model;
  Consensus mode stores consensus_models and uses them for pipeline generation.

/compact
  Manually trigger context compaction and replace the active conversation context.

/skills
  List available agent skills loaded from disk and built-ins.

MCP Tools
  Connected MCP servers expose callable tools named mcp__<server>__<tool>.
  Use /mcp to list server status and /mcp reload to refresh configured servers.

Memory
  Agent memory stores stable facts and can be searched with /memory.

Context
  Each response prints current context usage as a footer.

Pipeline
  Big product requests run SRS -> Tech Stack selection -> Code -> QA -> Deployment.
  Simple requests bypass the pipeline and run directly.
  Each stage is checkpointed — if a stage fails, run '/pipeline resume' to continue.
    /pipeline <request>    Start a new pipeline (auto-detects existing checkpoint)
    /pipeline resume       Resume from the last saved checkpoint
    /pipeline status       Show checkpoint state
    /pipeline reset        Discard saved checkpoint
"""
    print(help_text.strip())
    return True


def cmd_clear(_args: str, state: AgentState, _config: dict) -> bool:
    state.messages.clear()
    state.turn_count = 0
    state.total_input_tokens = 0
    state.total_output_tokens = 0
    ok("Conversation cleared.")
    return True


def cmd_model(args: str, _state: AgentState, config: dict) -> bool:
    raw = args.strip()
    if not raw:
        _run_model_selection_flow(config)
        return True
    if raw in {"local", "cloud"}:
        _choose_single_model(config, endpoint_hint=raw)
        return True
    if "/" not in raw:
        raw = f"{config.get('active_ollama_endpoint', 'local')}/{raw}"
    config["model"] = raw
    config["active_model"] = raw
    config["llm_mode"] = "single"
    config["active_ollama_endpoint"] = detect_provider(raw)
    save_config(config)
    ok(f"Model set to {raw}")
    return True


def cmd_config(args: str, _state: AgentState, config: dict) -> bool:
    raw = args.strip()
    if not raw:
        print(json.dumps({k: v for k, v in config.items() if not k.startswith("_")}, indent=2))
        return True
    if "=" not in raw:
        info(f"{raw} = {config.get(raw)}")
        return True
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        parsed: object = lowered == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            parsed = value
    config[key] = parsed
    save_config(config)
    ok(f"Updated {key}")
    return True


def cmd_configure(args: str, _state: AgentState, config: dict) -> bool:
    raw = args.strip()
    llm_params = {
        "model": "Active LLM model identifier",
        "ollama_local_base_url": "Local Ollama API URL",
        "ollama_cloud_base_url": "Cloud Ollama API URL",
        "ollama_cloud_api_key": "API key for cloud endpoint",
        "max_tokens": "Maximum context tokens",
        "thinking_budget": "Token budget for thinking/reasoning",
    }

    if raw:
        if "=" not in raw:
            if raw in llm_params:
                info(f"{raw} = {config.get(raw)} ({llm_params[raw]})")
            else:
                err(f"Unknown LLM parameter: {raw}")
            return True
        key, value = raw.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in llm_params:
            warn(f"'{key}' is not a standard LLM parameter, but updating anyway.")
        
        try:
            if value.isdigit():
                config[key] = int(value)
            elif value.lower() in {"true", "false"}:
                config[key] = value.lower() == "true"
            else:
                config[key] = value
            save_config(config)
            ok(f"Configured {key} = {config[key]}")
        except Exception as exc:
            err(f"Failed to update {key}: {exc}")
        return True

    info("Interactive LLM Configuration (press Enter to skip a value):")
    for key, desc in llm_params.items():
        current = config.get(key, "")
        val = ask_input_interactive(f"{key} [{current}] ({desc}): ", config).strip()
        if val:
            try:
                if val.isdigit():
                    config[key] = int(val)
                elif val.lower() in {"true", "false"}:
                    config[key] = val.lower() == "true"
                else:
                    config[key] = val
            except Exception as exc:
                err(f"Invalid value for {key}: {exc}")
    
    save_config(config)
    ok("Configuration updated.")
    return True


def cmd_save(_args: str, state: AgentState, config: dict) -> bool:
    session_id = config.get("_session_id", str(uuid.uuid4())[:8])
    save_session(state, config, session_id)
    return True


def cmd_load(args: str, state: AgentState, _config: dict) -> bool:
    raw = args.strip()
    if not raw:
        err("Usage: /load <path-to-session.json>")
        return True
    new_state = load_session_file(Path(raw))
    if new_state is None:
        return True
    state.messages = new_state.messages
    state.turn_count = new_state.turn_count
    state.total_input_tokens = new_state.total_input_tokens
    state.total_output_tokens = new_state.total_output_tokens
    ok(f"Loaded session from {raw}")
    return True


def cmd_resume(_args: str, state: AgentState, _config: dict) -> bool:
    latest = MR_SESSION_DIR / "session_latest.json"
    new_state = load_session_file(latest)
    if new_state is None:
        return True
    state.messages = new_state.messages
    state.turn_count = new_state.turn_count
    state.total_input_tokens = new_state.total_input_tokens
    state.total_output_tokens = new_state.total_output_tokens
    ok(f"Resumed session from {latest}")
    return True


def cmd_history(_args: str, state: AgentState, _config: dict) -> bool:
    if not state.messages:
        info("No conversation history yet.")
        return True
    for idx, message in enumerate(state.messages, 1):
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        snippet = content[:300] + ("..." if len(content) > 300 else "")
        print(f"[{idx:03d}] {message.get('role', '?')}: {snippet}")
    return True


def cmd_context(_args: str, state: AgentState, config: dict) -> bool:
    used, limit, percent = _context_usage(state, config)
    info(f"Context estimate: {percent}% used | ~{used:,} / {limit:,} tokens")
    return True


def cmd_cost(_args: str, state: AgentState, config: dict) -> bool:
    cost = calc_cost(config["model"], state.total_input_tokens, state.total_output_tokens)
    info(f"Input tokens:  {state.total_input_tokens}")
    info(f"Output tokens: {state.total_output_tokens}")
    info(f"Estimated cost: ${cost:.4f}")
    return True


def cmd_verbose(_args: str, _state: AgentState, config: dict) -> bool:
    config["verbose"] = not bool(config.get("verbose"))
    save_config(config)
    ok(f"verbose = {config['verbose']}")
    return True


def cmd_thinking(_args: str, _state: AgentState, config: dict) -> bool:
    config["thinking"] = not bool(config.get("thinking"))
    save_config(config)
    ok(f"thinking = {config['thinking']}")
    return True


def cmd_permissions(args: str, _state: AgentState, config: dict) -> bool:
    raw = args.strip()
    if not raw:
        info(f"permission_mode = {config.get('permission_mode')}")
        return True
    if raw not in {"auto", "manual", "accept-all", "plan"}:
        err("Use one of: auto, manual, accept-all, plan")
        return True
    config["permission_mode"] = raw
    save_config(config)
    ok(f"permission_mode = {raw}")
    return True


def cmd_cwd(args: str, _state: AgentState, _config: dict) -> bool:
    raw = args.strip()
    if not raw:
        info(str(Path.cwd()))
        return True
    target = Path(raw).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        err(f"Directory not found: {target}")
        return True
    os.chdir(target)
    ok(f"cwd -> {target}")
    return True


def cmd_skills(_args: str, _state: AgentState, _config: dict) -> bool:
    skills = load_skills()
    if not skills:
        info("No skills available.")
        return True
    for skill in skills:
        triggers = ", ".join(skill.triggers)
        print(f"- {skill.name} [{triggers}]")
        if skill.description:
            print(f"  {skill.description}")
        if skill.when_to_use:
            print(f"  when: {skill.when_to_use}")
    return True


def cmd_memory(args: str, _state: AgentState, _config: dict) -> bool:
    query = args.strip()
    entries = search_memory(query) if query else load_index("all")
    if not entries:
        info("No memories found.")
        return True
    for entry in entries:
        print(f"- {entry.name} [{entry.scope}/{entry.type}]")
        if entry.description:
            print(f"  {entry.description}")
    return True


def cmd_mcp(args: str, _state: AgentState, _config: dict) -> bool:
    parts = shlex.split(args) if args.strip() else []
    if not parts:
        configs = load_mcp_configs()
        files = list_config_files()
        manager = get_mcp_manager()
        print("MCP config files:")
        for file_path in files:
            print(f"- {file_path}")
        if not configs:
            info("No MCP servers configured.")
            return True
        print("Servers:")
        live = {client.config.name: client for client in manager.list_servers()}
        for name, cfg in configs.items():
            client = live.get(name)
            status = client.state.value if client else "disconnected"
            print(f"- {name}: {status} ({cfg.transport.value})")
            if client and client.state.value == "connected":
                for tool in client._tools:
                    print(f"  - {tool.qualified_name}: {tool.description}")
        return True

    subcmd = parts[0]
    if subcmd == "reload":
        if len(parts) == 1:
            reload_mcp()
            ok("Reloaded all MCP servers.")
        else:
            error = refresh_server(parts[1])
            if error:
                err(error)
            else:
                ok(f"Reloaded MCP server {parts[1]}")
        return True

    if subcmd == "add" and len(parts) >= 3:
        name = parts[1]
        command = parts[2]
        add_server_to_user_config(name, {"type": "stdio", "command": command, "args": parts[3:]})
        ok(f"Added MCP server {name}")
        return True

    if subcmd == "remove" and len(parts) == 2:
        removed = remove_server_from_user_config(parts[1])
        if removed:
            ok(f"Removed MCP server {parts[1]}")
        else:
            err(f"MCP server not found: {parts[1]}")
        return True

    err("Usage: /mcp [reload [name] | add <name> <command> [args...] | remove <name>]")
    return True


def cmd_tasks(args: str, _state: AgentState, _config: dict) -> bool:
    parts = shlex.split(args) if args.strip() else []
    if not parts:
        tasks = list_tasks()
        if not tasks:
            info("No tasks.")
            return True
        for task in tasks:
            print(f"- #{task.id} [{task.status.value}] {task.subject}")
        return True

    subcmd = parts[0]
    rest = " ".join(parts[1:]).strip()
    if subcmd == "create":
        if not rest:
            err("Usage: /tasks create <subject>")
            return True
        task = create_task(rest, rest)
        ok(f"Created task #{task.id}")
        return True
    if subcmd in {"start", "done", "cancel"}:
        if not rest:
            err(f"Usage: /tasks {subcmd} <id>")
            return True
        status = {"start": "in_progress", "done": "completed", "cancel": "cancelled"}[subcmd]
        task, _ = update_task(rest, status=status)
        if task is None:
            err(f"Task not found: {rest}")
        else:
            ok(f"Updated task #{rest}")
        return True
    if subcmd == "delete":
        if delete_task(rest):
            ok(f"Deleted task #{rest}")
        else:
            err(f"Task not found: {rest}")
        return True
    if subcmd == "get":
        task = get_task(rest)
        if task is None:
            err(f"Task not found: {rest}")
            return True
        print(json.dumps(task.to_dict(), indent=2))
        return True
    if subcmd == "clear":
        clear_all_tasks()
        ok("Cleared all tasks.")
        return True

    err("Usage: /tasks [create|get|start|done|cancel|delete|clear]")
    return True


def cmd_council(args: str, state: AgentState, config: dict) -> bool:
    _run_council(args.strip(), state, config)
    _record_snapshot(state, config, f"/council {args.strip()}".strip())
    return True


def cmd_srs(args: str, _state: AgentState, config: dict) -> bool:
    user_text = args.strip() or ask_input_interactive("SRS request: ", config).strip()
    if user_text:
        _run_stage("srs", user_text, config)
        if _active_state is not None:
            _record_snapshot(_active_state, config, f"/srs {user_text}")
    return True


def cmd_milestones(args: str, _state: AgentState, config: dict) -> bool:
    user_text = args.strip() or _stage_context()
    if not user_text:
        user_text = ask_input_interactive("Milestone context: ", config).strip()
    if user_text:
        _run_stage("milestones", user_text, config)
        if _active_state is not None:
            _record_snapshot(_active_state, config, f"/milestones {user_text[:80]}")
    return True


def cmd_techstack(args: str, _state: AgentState, config: dict) -> bool:
    user_text = args.strip() or _stage_context()
    if not user_text:
        user_text = ask_input_interactive("Tech stack context: ", config).strip()
    if user_text:
        _run_tech_stack_selection(user_text, config)
        if _active_state is not None:
            _record_snapshot(_active_state, config, f"/techstack {user_text[:80]}")
    return True


def cmd_qa(args: str, _state: AgentState, config: dict) -> bool:
    user_text = args.strip() or _stage_context()
    if not user_text:
        user_text = ask_input_interactive("QA context: ", config).strip()
    if user_text:
        _run_stage("qa", user_text, config)
        if _active_state is not None:
            _record_snapshot(_active_state, config, f"/qa {user_text[:80]}")
    return True


def cmd_deploy(args: str, _state: AgentState, config: dict) -> bool:
    user_text = args.strip() or _stage_context()
    if not user_text:
        user_text = ask_input_interactive("Deployment context: ", config).strip()
    if user_text:
        _run_stage("deploy", user_text, config)
        if _active_state is not None:
            _record_snapshot(_active_state, config, f"/deploy {user_text[:80]}")
    return True


def cmd_pipeline(args: str, state: AgentState, config: dict) -> bool:
    raw = args.strip()
    parts = raw.split(None, 1)
    subcmd = parts[0].lower() if parts else ""

    # /pipeline reset — discard any saved checkpoint
    if subcmd == "reset":
        _clear_pipeline_state()
        ok("Pipeline checkpoint cleared.")
        return True

    # /pipeline status — show checkpoint state
    if subcmd == "status":
        saved = _load_pipeline_state()
        if saved is None:
            info("No pipeline checkpoint found.")
        else:
            completed = saved.get("completed_stages", [])
            remaining = [s for s in _PIPELINE_STAGES if s not in completed]
            info(f"Pipeline checkpoint found (saved {saved.get('saved_at', '?')})")
            info(f"  Query: {saved['query'][:100]}")
            info(f"  Completed: {', '.join(completed) or 'none'}")
            info(f"  Remaining: {', '.join(remaining) or 'all done'}")
            if saved.get("judge_model"):
                info(f"  Judge model: {saved['judge_model']}")
        return True

    # /pipeline resume — resume from checkpoint
    if subcmd == "resume":
        saved = _load_pipeline_state()
        if saved is None:
            err("No pipeline checkpoint to resume.")
            return True
        completed = saved.get("completed_stages", [])
        remaining = [s for s in _PIPELINE_STAGES if s not in completed]
        if not remaining:
            ok("Pipeline already completed. Use '/pipeline reset' to start fresh.")
            _clear_pipeline_state()
            return True
        query = saved["query"]
        # Restore model settings from checkpoint
        if saved.get("llm_mode"):
            config["llm_mode"] = saved["llm_mode"]
        if saved.get("model"):
            config["model"] = saved["model"]
        if saved.get("consensus_models"):
            config["consensus_models"] = saved["consensus_models"]
        if saved.get("judge_model"):
            config["judge_model"] = saved["judge_model"]
        info(f"Resuming pipeline for: {query[:100]}")
        _run_full_btp_cycle(query, state, config, resume_from=completed)
        _record_snapshot(state, config, f"/pipeline resume {query[:60]}")
        return True

    # Regular /pipeline <request> — check for existing checkpoint first
    request = raw or ask_input_interactive("Pipeline request: ", config).strip()
    if not request:
        return True

    saved = _load_pipeline_state()
    if saved is not None:
        completed = saved.get("completed_stages", [])
        remaining = [s for s in _PIPELINE_STAGES if s not in completed]
        if remaining:
            print()
            warn(f"An unfinished pipeline was found (saved {saved.get('saved_at', '?')})")
            info(f"  Query: {saved['query'][:100]}")
            info(f"  Completed: {', '.join(completed)}")
            info(f"  Next stage: {remaining[0]}")
            choice = ask_input_interactive(
                "Resume this pipeline? [y]es / [n]ew / [c]ancel: ", config
            ).strip().lower()
            if choice in {"y", "yes", "resume"}:
                # Restore model settings
                if saved.get("llm_mode"):
                    config["llm_mode"] = saved["llm_mode"]
                if saved.get("model"):
                    config["model"] = saved["model"]
                if saved.get("consensus_models"):
                    config["consensus_models"] = saved["consensus_models"]
                if saved.get("judge_model"):
                    config["judge_model"] = saved["judge_model"]
                _run_full_btp_cycle(saved["query"], state, config, resume_from=completed)
                _record_snapshot(state, config, f"/pipeline resume {saved['query'][:60]}")
                return True
            elif choice in {"n", "new"}:
                _clear_pipeline_state()
            else:
                info("Pipeline cancelled.")
                return True

    _run_model_selection_flow(config)
    _run_full_btp_cycle(request, state, config)
    _record_snapshot(state, config, f"/pipeline {request[:80]}")
    return True


def cmd_checkpoint(args: str, state: AgentState, config: dict) -> bool:
    session_id = config.get("_session_id", "")
    raw = args.strip()
    if not session_id:
        err("No active session id.")
        return True
    if raw == "clear":
        ckpt.delete_session_checkpoints(session_id)
        ok("Cleared checkpoints for this session.")
        return True
    if not raw:
        snapshots = ckpt.list_snapshots(session_id)
        if not snapshots:
            info("No checkpoints yet.")
            return True
        for snapshot in snapshots:
            print(
                f"- #{snapshot['id']} turns={snapshot['turn_count']} "
                f"files={snapshot['file_count']} prompt={snapshot['user_prompt_preview']}"
            )
        return True
    try:
        snapshot_id = int(raw)
    except ValueError:
        err("Usage: /checkpoint [clear|<id>]")
        return True
    results = ckpt.rewind_files(session_id, snapshot_id)
    if not results:
        err(f"Checkpoint not found: {snapshot_id}")
        return True
    for line in results:
        print(line)
    ckpt.make_snapshot(session_id, state, config, f"(rewind to {snapshot_id})")
    ok(f"Restored checkpoint #{snapshot_id}")
    return True


cmd_rewind = cmd_checkpoint


def cmd_plan(args: str, _state: AgentState, config: dict) -> bool:
    raw = args.strip()
    if raw == "done":
        config["permission_mode"] = config.get("_pre_plan_permission", "auto")
        config.pop("_pre_plan_permission", None)
        config.pop("_plan_file", None)
        save_config(config)
        ok("Plan mode disabled.")
        return True
    if raw == "status":
        if config.get("permission_mode") == "plan":
            info(f"Plan mode active: {config.get('_plan_file', '')}")
        else:
            info("Plan mode is off.")
        return True
    if not raw:
        plan_file = config.get("_plan_file")
        if not plan_file:
            info("No active plan file.")
            return True
        print(Path(plan_file).read_text(encoding="utf-8"))
        return True

    plans_dir = CONFIG_DIR / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_file = plans_dir / f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    plan_file.write_text(f"# Plan\n\n{raw}\n", encoding="utf-8")
    config["_pre_plan_permission"] = config.get("permission_mode", "auto")
    config["permission_mode"] = "plan"
    config["_plan_file"] = str(plan_file)
    save_config(config)
    ok(f"Plan mode enabled: {plan_file}")
    return True


def cmd_compact(args: str, state: AgentState, config: dict) -> bool:
    success, message = manual_compact(state, config, focus=args.strip())
    if success:
        ok(message)
    else:
        info(message)
    return True


def cmd_status(_args: str, state: AgentState, config: dict) -> bool:
    print(f"Model: {config['model']}")
    print(f"LLM mode: {config.get('llm_mode', 'single')}")
    if config.get("llm_mode") == "consensus":
        print(f"Consensus models: {', '.join(config.get('consensus_models') or [])}")
    print(f"Endpoint: {current_provider(config)}")
    print(f"CWD: {Path.cwd()}")
    print(f"Messages: {len(state.messages)}")
    print(f"Tokens in/out: {state.total_input_tokens}/{state.total_output_tokens}")
    print(f"Plan mode: {config.get('permission_mode') == 'plan'}")
    return True


def cmd_doctor(_args: str, _state: AgentState, config: dict) -> bool:
    for endpoint in PROVIDERS:
        base_url = get_base_url(endpoint, config)
        if not base_url:
            info(f"{endpoint}: not configured")
            continue
        models = list_ollama_models(base_url, api_key=get_api_key(endpoint, config))
        if models:
            ok(f"{endpoint}: reachable ({len(models)} models)")
        else:
            err(f"{endpoint}: unreachable or no models at {base_url}/api/tags")
    info(f"Skills: {len(load_skills())}")
    info(f"MCP configs: {len(load_mcp_configs())}")
    return True


def cmd_exit(_args: str, _state: AgentState, _config: dict) -> bool:
    return False


COMMANDS = {
    "help": cmd_help,
    "clear": cmd_clear,
    "model": cmd_model,
    "config": cmd_config,
    "configure": cmd_configure,
    "save": cmd_save,
    "load": cmd_load,
    "resume": cmd_resume,
    "history": cmd_history,
    "context": cmd_context,
    "cost": cmd_cost,
    "verbose": cmd_verbose,
    "thinking": cmd_thinking,
    "permissions": cmd_permissions,
    "cwd": cmd_cwd,
    "skills": cmd_skills,
    "memory": cmd_memory,
    "mcp": cmd_mcp,
    "tasks": cmd_tasks,
    "task": cmd_tasks,
    "council": cmd_council,
    "srs": cmd_srs,
    "milestones": cmd_milestones,
    "techstack": cmd_techstack,
    "qa": cmd_qa,
    "deploy": cmd_deploy,
    "pipeline": cmd_pipeline,
    "checkpoint": cmd_checkpoint,
    "rewind": cmd_rewind,
    "plan": cmd_plan,
    "compact": cmd_compact,
    "status": cmd_status,
    "doctor": cmd_doctor,
    "exit": cmd_exit,
    "quit": cmd_exit,
}


def _invoke_skill_from_slash(line: str, state: AgentState, config: dict) -> bool | None:
    skill = find_skill(line)
    if skill is None:
        return None
    _, _, remainder = line.strip().partition(" ")
    rendered = substitute_arguments(skill.prompt, remainder.strip(), skill.arguments)
    _run_agent_query(f"[Skill: {skill.name}]\n\n{rendered}", state, config)
    return True


def handle_slash(line: str, state: AgentState, config: dict) -> bool:
    raw = line.strip()[1:]
    name, _, args = raw.partition(" ")
    command = COMMANDS.get(name)
    if command is not None:
        return bool(command(args, state, config))
    skill_result = _invoke_skill_from_slash(line, state, config)
    if skill_result is not None:
        return bool(skill_result)
    err(f"Unknown command: /{name}")
    return True


def _make_prompt_prefix(config: dict) -> str:
    model = bare_model(config["model"])
    cwd_name = Path.cwd().name
    return clr(f"[{cwd_name}:{model}] > ", "yellow", "bold")


def _record_snapshot(state: AgentState, config: dict, user_prompt: str) -> None:
    session_id = config.get("_session_id", "")
    if not session_id:
        return
    tracked = ckpt.get_tracked_edits()
    ckpt.make_snapshot(session_id, state, config, user_prompt, tracked_edits=tracked)
    ckpt.reset_tracked()


def _process_input(user_input: str, state: AgentState, config: dict) -> bool:
    if not user_input.strip():
        return True
    if user_input.startswith("/"):
        keep_running = handle_slash(user_input, state, config)
        if keep_running:
            _print_context_footer(state, config)
        return keep_running

    if _looks_like_large_product_request(user_input):
        _run_model_selection_flow(config)
        _run_full_btp_cycle(user_input, state, config)
        _record_snapshot(state, config, f"[full-cycle] {user_input}")
        _print_context_footer(state, config)
        return True

    _run_agent_query(user_input, state, config, use_skills=_should_apply_skill_context(user_input))
    _record_snapshot(state, config, user_input)
    _print_context_footer(state, config)
    return True


def _run_scheduled_turns(state: AgentState, config: dict) -> None:
    for query in _drain_scheduled_queries():
        info("[scheduled] running queued system event")
        _run_agent_query(query, state, config, use_skills=_should_apply_skill_context(query))
        _record_snapshot(state, config, query)
        _print_context_footer(state, config)


def main() -> int:
    _ensure_utf8_stdio()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true")
    parser.add_argument("-m", "--model", dest="model")
    parser.add_argument("--accept-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(VERSION)
        return 0

    config = load_config()
    if args.model:
        config["model"] = args.model
        config["active_ollama_endpoint"] = detect_provider(args.model)
    if args.accept_all:
        config["permission_mode"] = "accept-all"
    if args.verbose:
        config["verbose"] = True

    state = AgentState()
    session_id = str(uuid.uuid4())[:8]
    config["_session_id"] = session_id
    ckpt.set_session(session_id)
    ckpt.make_snapshot(session_id, state, config, "(initial state)")

    global _active_state, _active_config
    _active_state = state
    _active_config = config

    prompt_text = " ".join(args.prompt).strip()
    if args.print_mode and prompt_text:
        try:
            _run_agent_query(prompt_text, state, config, use_skills=_should_apply_skill_context(prompt_text))
            _record_snapshot(state, config, prompt_text)
            _print_context_footer(state, config)
        except KeyboardInterrupt:
            _exit_on_interrupt()
        return 0

    _print_banner()

    if prompt_text:
        try:
            _process_input(prompt_text, state, config)
        except KeyboardInterrupt:
            _exit_on_interrupt()
            return 0

    while True:
        try:
            _run_scheduled_turns(state, config)
            user_input = ask_input_interactive(_make_prompt_prefix(config), config)
            keep_running = _process_input(user_input, state, config)
        except (KeyboardInterrupt, EOFError):
            _exit_on_interrupt()
            break
        if not keep_running:
            break

    return 0


atexit.register(_autosave_session)


if __name__ == "__main__":
    raise SystemExit(main())
