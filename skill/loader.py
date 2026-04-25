"""Skill loading: parse markdown files with YAML frontmatter into SkillDef objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SkillDef:
    name: str
    description: str
    triggers: list[str]          # ["/commit", "commit changes"]
    tools: list[str]             # ["Bash", "Read"]  (allowed-tools)
    prompt: str                  # full prompt body after frontmatter
    file_path: str
    # Enhanced fields
    when_to_use: str = ""        # when Claude should auto-invoke this skill
    argument_hint: str = ""      # e.g. "[branch] [description]"
    arguments: list[str] = field(default_factory=list)  # named arg names
    model: str = ""              # model override
    user_invocable: bool = True  # appears in /skills list
    context: str = "inline"      # only inline is supported in dev-council
    source: str = "user"         # "user", "project", "builtin"


# ── Directory paths ────────────────────────────────────────────────────────

def _get_skill_paths() -> list[Path]:
    return [
        Path.cwd() / ".agents" / "skills",
        Path.cwd() / ".codex" / "skills",
        Path.cwd() / ".dev-council" / "skills",   # project-level (priority)
        Path.home() / ".agents" / "skills",
        Path.home() / ".codex" / "skills",
        Path.home() / ".dev-council" / "skills",   # user-level
    ]


def _iter_skill_files(skill_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not skill_dir.is_dir():
        return files
    files.extend(sorted(skill_dir.glob("*.md")))
    files.extend(sorted(skill_dir.glob("*/SKILL.md")))
    return files


# ── List field parser ──────────────────────────────────────────────────────

def _parse_list_field(value: str) -> list[str]:
    """Parse YAML-like list: ``[a, b, c]`` or ``"a, b, c"``."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]


# ── Single-file parser ─────────────────────────────────────────────────────

def _parse_skill_file(path: Path, source: str = "user") -> Optional[SkillDef]:
    """Parse a markdown file with ``---`` frontmatter into a SkillDef.

    Frontmatter fields:
        name, description, triggers, tools / allowed-tools,
        when_to_use, argument-hint, arguments, model,
        user-invocable, context
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter_raw = parts[1].strip()
    prompt = parts[2].strip()

    fields: dict[str, str] = {}
    lines = frontmatter_raw.splitlines()
    idx = 0
    while idx < len(lines):
        raw_line = lines[idx]
        line = raw_line.strip()
        idx += 1
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if val == "|":
            block: list[str] = []
            while idx < len(lines):
                candidate = lines[idx]
                if candidate and not candidate.startswith((" ", "\t")):
                    break
                block.append(candidate.strip())
                idx += 1
            fields[key] = " ".join(part for part in block if part).strip()
        else:
            fields[key] = val.strip().strip('"').strip("'")

    name = fields.get("name", "")
    if not name:
        return None

    # allowed-tools wins over tools if present
    tools_raw = fields.get("allowed-tools", fields.get("tools", ""))
    tools = _parse_list_field(tools_raw) if tools_raw else []

    triggers_raw = fields.get("triggers", "")
    triggers = _parse_list_field(triggers_raw) if triggers_raw else [f"/{name}"]

    arguments_raw = fields.get("arguments", "")
    arguments = _parse_list_field(arguments_raw) if arguments_raw else []

    user_invocable_raw = fields.get("user-invocable", "true")
    user_invocable = user_invocable_raw.lower() not in ("false", "0", "no")

    context = "inline"

    return SkillDef(
        name=name,
        description=fields.get("description", ""),
        triggers=triggers,
        tools=tools,
        prompt=prompt,
        file_path=str(path),
        when_to_use=fields.get("when_to_use", ""),
        argument_hint=fields.get("argument-hint", ""),
        arguments=arguments,
        model=fields.get("model", ""),
        user_invocable=user_invocable,
        context=context,
        source=source,
    )


# ── Registry of built-in skills (registered by builtin.py) ────────────────

_BUILTIN_SKILLS: list[SkillDef] = []


def register_builtin_skill(skill: SkillDef) -> None:
    _BUILTIN_SKILLS.append(skill)


# ── Load all skills ────────────────────────────────────────────────────────

def load_skills(include_builtins: bool = True) -> list[SkillDef]:
    """Return skills from disk + builtins, deduplicated (project > user > builtin)."""
    seen: dict[str, SkillDef] = {}

    # Builtins go in first (lowest priority)
    if include_builtins:
        for sk in _BUILTIN_SKILLS:
            seen[sk.name] = sk

    # Later paths override earlier ones. User-level first, project-level last.
    skill_paths = list(reversed(_get_skill_paths()))
    for skill_dir in skill_paths:
        src = "project" if str(skill_dir).startswith(str(Path.cwd())) else "user"
        for md_file in _iter_skill_files(skill_dir):
            skill = _parse_skill_file(md_file, source=src)
            if skill:
                seen[skill.name] = skill

    return list(seen.values())


def find_skill(query: str) -> Optional[SkillDef]:
    """Find a skill whose trigger matches the first word (or whole string) of query."""
    query = query.strip()
    if not query:
        return None

    first_word = query.split()[0]
    for skill in load_skills():
        for trigger in skill.triggers:
            if first_word == trigger:
                return skill
            if trigger.startswith(first_word + " "):
                return skill
    return None


# ── Argument substitution ─────────────────────────────────────────────────

def substitute_arguments(prompt: str, args: str, arg_names: list[str]) -> str:
    """Replace $ARGUMENTS (whole args string) and $ARG_NAME placeholders.

    Named args are positional: first word → first name, etc.
    """
    # Always substitute $ARGUMENTS
    result = prompt.replace("$ARGUMENTS", args)

    # Named args: split by whitespace
    arg_values = args.split()
    for i, arg_name in enumerate(arg_names):
        placeholder = f"${arg_name.upper()}"
        value = arg_values[i] if i < len(arg_values) else ""
        result = result.replace(placeholder, value)

    return result
