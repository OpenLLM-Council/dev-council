"""Built-in SDLC skills for dev-council."""
from __future__ import annotations

from .loader import SkillDef, register_builtin_skill


def _register_builtins() -> None:
    register_builtin_skill(
        SkillDef(
            name="srs",
            description="Create a Software Requirements Specification from a user request",
            triggers=["/srs"],
            tools=[],
            prompt=(
                "Create a complete Software Requirements Specification for this request:\n\n"
                "$ARGUMENTS\n\n"
                "Include: scope, goals, functional requirements, non-functional requirements, "
                "constraints, assumptions, risks, success criteria, and open questions."
            ),
            file_path="<builtin>",
            when_to_use="Use for SDLC stage 2 when requirements need to be formalized.",
            argument_hint="<request>",
            arguments=["request"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="milestones",
            description="Turn an SRS or project request into milestones",
            triggers=["/milestones"],
            tools=[],
            prompt=(
                "Create a milestone roadmap for this project context:\n\n"
                "$ARGUMENTS\n\n"
                "Define milestone names, deliverables, dependencies, sequencing, and acceptance checks."
            ),
            file_path="<builtin>",
            when_to_use="Use for SDLC stage 3 after the SRS is available.",
            argument_hint="<srs-or-request>",
            arguments=["context"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="techstack",
            description="Recommend an implementation stack aligned with the project",
            triggers=["/techstack"],
            tools=[],
            prompt=(
                "Recommend a technology stack for the following project context:\n\n"
                "$ARGUMENTS\n\n"
                "Evaluate frontend, backend, database, infrastructure, testing, and CI/CD choices. "
                "Explain tradeoffs and end with a recommended stack."
            ),
            file_path="<builtin>",
            when_to_use="Use for SDLC stage 4 to lock the implementation stack.",
            argument_hint="<srs-and-milestones>",
            arguments=["context"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="qa",
            description="Produce a QA strategy or report for the current implementation",
            triggers=["/qa"],
            tools=[],
            prompt=(
                "Create a QA assessment for this project context:\n\n"
                "$ARGUMENTS\n\n"
                "Cover unit tests, integration tests, end-to-end validation, performance risks, "
                "security risks, gaps, and recommended fixes."
            ),
            file_path="<builtin>",
            when_to_use="Use for SDLC stage 6 after or during implementation review.",
            argument_hint="<implementation-context>",
            arguments=["context"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="deploy",
            description="Produce a deployment and CI/CD plan",
            triggers=["/deploy"],
            tools=[],
            prompt=(
                "Create a deployment and CI/CD plan for this project context:\n\n"
                "$ARGUMENTS\n\n"
                "Include environments, build/test pipeline, secrets strategy, release flow, rollback, "
                "monitoring, and production-readiness checks."
            ),
            file_path="<builtin>",
            when_to_use="Use for SDLC stage 7 to prepare production delivery.",
            argument_hint="<project-context>",
            arguments=["context"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )
    register_builtin_skill(
        SkillDef(
            name="pipeline",
            description="Summarize the full SDLC workflow for a request",
            triggers=["/pipeline"],
            tools=[],
            prompt=(
                "For the following request, outline the full SDLC workflow across SRS, milestones, "
                "tech stack, coding, QA, and deployment:\n\n$ARGUMENTS"
            ),
            file_path="<builtin>",
            when_to_use="Use to frame an end-to-end SDLC execution plan.",
            argument_hint="<request>",
            arguments=["request"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="implementation-core",
            description="Core coding discipline for implementing features safely and cleanly",
            triggers=["/implementation-core"],
            tools=[],
            prompt=(
                "Apply strong coding fundamentals while implementing this request:\n\n"
                "$ARGUMENTS\n\n"
                "Prefer small, coherent changes, preserve existing conventions, update affected tests, "
                "and avoid interactive project scaffolding. When bootstrapping a project, use explicit "
                "flags and non-interactive commands such as `--yes` or `-y`."
            ),
            file_path="<builtin>",
            when_to_use="Use whenever you are writing or modifying code.",
            argument_hint="<coding-task>",
            arguments=["task"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="frontend-builder",
            description="Frontend coding guidance for UI, layout, responsiveness, and component structure",
            triggers=["/frontend-builder"],
            tools=[],
            prompt=(
                "Apply frontend engineering guidance to this task:\n\n"
                "$ARGUMENTS\n\n"
                "Focus on clear component boundaries, responsive behavior, accessible semantics, "
                "stable state flow, and polished UX. Avoid generic UI scaffolds and preserve the "
                "existing design language when one already exists."
            ),
            file_path="<builtin>",
            when_to_use="Use for frontend, UI, landing page, React, Next.js, HTML, CSS, or design-heavy work.",
            argument_hint="<frontend-task>",
            arguments=["task"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="backend-builder",
            description="Backend coding guidance for APIs, services, data models, and reliability",
            triggers=["/backend-builder"],
            tools=[],
            prompt=(
                "Apply backend engineering guidance to this task:\n\n"
                "$ARGUMENTS\n\n"
                "Design clear interfaces, validate inputs, keep business logic separated from transport, "
                "handle errors intentionally, and think through persistence, security, and maintainability."
            ),
            file_path="<builtin>",
            when_to_use="Use for backend, API, service, database, auth, queue, or server-side work.",
            argument_hint="<backend-task>",
            arguments=["task"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="fullstack-builder",
            description="Full-stack product guidance spanning frontend, backend, data, and deployment",
            triggers=["/fullstack-builder"],
            tools=[],
            prompt=(
                "Apply full-stack product engineering guidance to this task:\n\n"
                "$ARGUMENTS\n\n"
                "Keep the frontend, backend, data model, auth flow, and deployment path aligned. "
                "Favor simple end-to-end slices that can be tested quickly."
            ),
            file_path="<builtin>",
            when_to_use="Use for SaaS, full-stack apps, dashboards, platforms, and larger product builds.",
            argument_hint="<fullstack-task>",
            arguments=["task"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )

    register_builtin_skill(
        SkillDef(
            name="testing-guard",
            description="Testing and verification guidance for new or changed behavior",
            triggers=["/testing-guard"],
            tools=[],
            prompt=(
                "Apply testing and verification guidance to this task:\n\n"
                "$ARGUMENTS\n\n"
                "Identify the most important unit, integration, and end-to-end checks. "
                "Prefer targeted tests that validate the changed behavior and call out any gaps clearly."
            ),
            file_path="<builtin>",
            when_to_use="Use when implementing behavior changes, fixing bugs, or adding features that need validation.",
            argument_hint="<testing-task>",
            arguments=["task"],
            user_invocable=True,
            context="inline",
            source="builtin",
        )
    )


_register_builtins()
