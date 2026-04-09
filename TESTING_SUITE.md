# Testing Suite

This document covers the main runtime behaviors of `dev-council`.

## 1. Basic Startup

Run:

```bash
python dev_council.py --version
python dev_council.py
```

Expected:

- version prints
- REPL starts
- banner mentions `dev-council`
- banner mentions BTP stages

## 2. Single Model Selection

Run:

```text
/model
```

Expected:

- endpoint selection appears if both local and cloud are configured
- models are loaded from `base_url/api/tags`
- choosing one saves `local/<model>` or `cloud/<model>`

## 3. Council Selection

Run:

```text
/council build a simple todo app
```

Expected:

- endpoint selection appears
- model count is requested
- model list is fetched from `base_url/api/tags`
- selected models generate proposal files under `btp/council/<timestamp>/`
- `consensus.md` is created
- implementation starts with the synthesis model

## 4. Automatic Skill Usage

Run one frontend-style request:

```text
Create a React landing page with hero, pricing, testimonials, and responsive layout.
```

Expected:

- terminal shows a line like:

```text
[skills] Using: implementation-core, frontend-builder
```

Run one backend-style request:

```text
Create a FastAPI backend with JWT auth and PostgreSQL models.
```

Expected:

- terminal shows a line like:

```text
[skills] Using: implementation-core, backend-builder
```

Run one SaaS/full-stack request:

```text
Build a SaaS dashboard with auth, billing placeholders, admin views, and API endpoints.
```

Expected:

- terminal shows `fullstack-builder` in the skill list

## 5. Intent-Based Cycle Routing

Open an effectively empty folder and run:

```text
Create sha1.py that hashes a string from stdin.
```

Expected:

- no full-cycle prompt
- direct coding starts

Open an effectively empty folder and run:

```text
Build a full stack SaaS CRM with dashboard, auth, API, and deployment setup.
```

Expected:

- CLI asks whether to run the full BTP cycle first
- choosing `Y` runs:
  - SRS
  - Milestones
  - Tech Stack
  - Code
  - QA
  - Deployment
- files are created under `btp/`

## 6. Stage Commands

Run:

```text
/srs Build a task management SaaS
/milestones
/techstack
/qa
/deploy
```

Expected files:

- `btp/srs.md`
- `btp/milestones.md`
- `btp/tech_stack.md`
- `btp/qa_report.md`
- `btp/deployment_plan.md`

## 7. Checkpoints

Run a request that edits files, then:

```text
/checkpoint
```

Expected:

- snapshot list appears

Then restore one:

```text
/checkpoint 1
```

Expected:

- file restore output appears

## 8. MCP

Add a server:

```text
/mcp add demo uvx mcp-server-git
/mcp
/mcp reload
```

Expected:

- server is added
- config file is shown
- status is visible

## 9. Skills Listing

Run:

```text
/skills
```

Expected:

- built-in BTP and coding skills are listed
- descriptions and `when:` guidance are shown

## 10. Non-Interactive Bootstrap Expectation

Ask for a scaffolded app in an empty folder.

Expected:

- the agent uses non-interactive bootstrap commands only
- required parameters are passed by flags
- `--yes` or `-y` is used when supported by the generator

## 11. Automated Checks

Run:

```bash
python -m py_compile dev_council.py config.py context.py providers.py tools.py
python -m pytest tests/test_minimal_runtime.py -q
```

Expected:

- py_compile succeeds
- tests pass
