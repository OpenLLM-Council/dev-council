"""Skill execution for dev-council."""
from __future__ import annotations

from typing import Generator

from .loader import SkillDef, substitute_arguments


def execute_skill(
    skill: SkillDef,
    args: str,
    state,
    config: dict,
    system_prompt: str,
) -> Generator:
    rendered = substitute_arguments(skill.prompt, args, skill.arguments)
    message = f"[Skill: {skill.name}]\n\n{rendered}"
    import agent as _agent

    yield from _agent.run(message, state, config, system_prompt)
