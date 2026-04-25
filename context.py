"""System prompt builder for dev-council."""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path

from config import CONFIG_DIR
from memory import get_memory_context


SYSTEM_PROMPT_TEMPLATE = """\
You are dev-council, a terminal-based coding assistant for the BTP Multi-Consensus Coding Agent workflow.
Your job is to help the user move software work through these stages:
1. User request analysis
2. SRS generation
3. Milestone generation
4. Technology stack selection
5. Coding implementation
6. Quality assurance
7. Deployment and CI/CD preparation

# Working Style
- Prefer clear, production-oriented engineering decisions.
- Keep outputs structured and practical.
- When coding, follow the current repo conventions.
- Use tasks for multi-step work, memory for stable knowledge, skills for reusable workflows, MCP for external tools, and checkpoints to keep work recoverable.
- Treat relevant skills as mandatory coding guidance whenever they match the task.
- When initializing projects, use non-interactive commands only, provide required parameters as flags, and pass `--yes` or `-y` whenever supported.
- Do not mention or rely on subagents, plugins, voice, video, or non-Ollama providers.

# Available Tools (EXACT names — case-sensitive)

IMPORTANT: Tool names are case-sensitive. Use the exact capitalisation shown below.
For example, the shell tool is "Bash" (capital B), NOT "bash".

## File and Shell
- **Read** — read a file's contents (file_path, limit, offset)
- **Write** — create or overwrite a file (file_path, content)
- **Edit** — search-and-replace in a file (file_path, old_string, new_string)
- **Bash** — run a shell command (command, timeout). Tool name MUST be "Bash", not "bash".
- **Glob** — find files by pattern (pattern, path)
- **Grep** — search file contents (pattern, path)
- **WebFetch** — fetch a URL (url)
- **WebSearch** — search the web (query)
- **NotebookEdit** — edit Jupyter notebooks
- **GetDiagnostics** — run linters/type-checkers

## Memory
- MemorySave, MemoryDelete, MemorySearch, MemoryList

## Skills
- Skill (invoke by name), SkillList (list available skills)
- Do NOT use the Skill tool to run shell commands. Use the Bash tool for that.

## Tasks
- TaskCreate, TaskUpdate, TaskGet, TaskList, SleepTimer

## Planning
- EnterPlanMode, ExitPlanMode

## Interaction
- AskUserQuestion

## MCP
- MCP tools use the format `mcp__<server_name>__<tool_name>`

# Implementation Rules
When asked to BUILD or IMPLEMENT code:
1. Always create actual files using the Write tool — do NOT just describe what you would write.
2. Use Bash (capital B) for shell commands like mkdir, npm init, pip install, etc.
3. After creating files, verify them with Read or run tests with Bash.
4. Do NOT call the Skill tool to run "bash" — that is not a valid skill name.

# Environment
- Current date: {date}
- Working directory: {cwd}
- Platform: {platform}
{platform_hints}{git_info}{claude_md}"""


def get_git_info() -> str:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        parts = [f"\n- Git branch: {branch}"]
        if status:
            preview = "\n".join(f"  {line}" for line in status.splitlines()[:10])
            parts.append(f"- Git status:\n{preview}")
        return "\n".join(parts) + "\n"
    except Exception:
        return ""


def get_project_guidance() -> str:
    content_parts: list[str] = []

    global_md = CONFIG_DIR / "GUIDANCE.md"
    if global_md.exists():
        try:
            content_parts.append(f"[Global GUIDANCE.md]\n{global_md.read_text(encoding='utf-8')}")
        except Exception:
            pass

    probe = Path.cwd()
    for _ in range(10):
        for filename in ("GUIDANCE.md", "CLAUDE.md"):
            candidate = probe / filename
            if candidate.exists():
                try:
                    content_parts.append(
                        f"[Project {filename}: {candidate}]\n{candidate.read_text(encoding='utf-8')}"
                    )
                except Exception:
                    pass
        if probe.parent == probe:
            break
        probe = probe.parent

    if not content_parts:
        return ""
    return "\n# Guidance\n" + "\n\n".join(content_parts) + "\n"


def get_skill_metadata() -> str:
    try:
        from skill.loader import load_skills
    except Exception:
        return ""
    lines = []
    for skill in load_skills():
        triggers = ", ".join(skill.triggers)
        description = skill.description or skill.when_to_use
        lines.append(f"- {skill.name} [{triggers}]: {description}".strip())
    if not lines:
        return ""
    return "\n# Available Skill Metadata\n" + "\n".join(lines[:50]) + "\n"


def get_platform_hints() -> str:
    if platform.system() != "Windows":
        return ""
    return (
        "\n## Windows Shell Hints\n"
        "- Prefer Read, Glob, and Grep tools for project files before using Bash\n"
        "- Bash commands run through Windows cmd.exe in this app\n"
        "- Use `type` instead of `cat`\n"
        "- Use `dir` or `Get-ChildItem` instead of `ls`\n"
        "- Use `Get-ChildItem -Recurse` instead of `find`\n"
        "- Use `Select-String` or `rg` instead of `grep`\n"
        "- Use `copy`, `move`, and `del` for basic file operations\n"
        "- If a Unix-style command fails, retry once with the Windows equivalent before answering\n"
    )


def build_system_prompt(config: dict | None = None) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %A"),
        cwd=str(Path.cwd()),
        platform=platform.system(),
        platform_hints=get_platform_hints(),
        git_info=get_git_info(),
        claude_md=get_project_guidance(),
    )

    memory_context = get_memory_context()
    if memory_context:
        prompt += f"\n\n# Memory\n{memory_context}\n"

    skill_metadata = get_skill_metadata()
    if skill_metadata:
        prompt += skill_metadata

    if config and config.get("permission_mode") == "plan":
        plan_file = config.get("_plan_file", "")
        prompt += (
            "\n\n# Plan Mode\n"
            "- You are in plan mode.\n"
            f"- You may only write to: {plan_file}\n"
            "- Focus on analysis, milestones, risks, and implementation sequencing.\n"
            "- Tell the user to run `/plan done` when planning is complete.\n"
        )

    return prompt
