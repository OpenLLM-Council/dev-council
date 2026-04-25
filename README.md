# dev-council

`dev-council` is a terminal coding agent for a BTP-style workflow:

```text
SRS -> Tech Stack Selection -> Code -> QA -> Deployment
```

It is built around Ollama-compatible models, automatic skill loading, MCP tools, memory, checkpoints, and context compaction.

## What It Does

- Routes simple requests directly to the coding agent.
- Routes larger product requirements through the full BTP pipeline.
- Prompts for Single LLM or Consensus mode before big pipeline runs.
- Uses live local Ollama model discovery through `ollama list`.
- Pauses at the Tech Stack stage and waits for the user to pick one of 2-3 options.
- Shows context usage after responses.
- Supports manual `/compact` and automatic compaction at 80% context usage.
- Loads skills from built-ins plus project/user skill folders.
- Connects MCP servers over stdio, HTTP, or SSE and exposes tools to the agent.

## Intent Routing

Normal text input is classified before execution:

- Simple requests, bug fixes, questions, and single-file changes bypass the pipeline.
- Big requirements trigger the full pipeline. The heuristic looks for product-building language such as `build`, `create`, `develop`, `make a full`, `implement a system`, and longer product descriptions.

Example simple request:

```text
Fix the typo in README.md
```

Example pipeline request:

```text
Build a full stack SaaS dashboard with auth, admin views, APIs, and deployment setup.
```

## Model Modes

Use `/model` to switch modes at any time.

Single LLM mode:

```text
/model
1
<model number>
```

Consensus mode:

```text
/model
2
<number of models>
<comma-separated model numbers>
```

Config keys:

- `llm_mode`: `single` or `consensus`
- `active_model`: the selected single model
- `consensus_models`: selected consensus voters
- `model`: the active execution model, kept compatible with existing provider code

## Commands

The final public help surface is intentionally small:

- `/model` — Switch LLM or consensus models.
- `/compact` — Manually trigger context compaction.
- `/skills` — List available agent skills.
- `/mcp` — List connected MCP servers and discovered tools.
- `/memory` — Search or list agent memory.
- `/context` — Show current context usage.
- `/pipeline` — Run the full SRS → Tech Stack → Code → QA → Deployment pipeline.

Additional internal/developer commands still exist for compatibility, but `/help` only documents implemented user-facing features.

## Pipeline

For big requirements, the pipeline runs:

1. SRS generation into `btp/srs.md`
2. Tech stack option generation
3. User selection of exactly one stack
4. Code implementation
5. QA report into `btp/qa_report.md`
6. Deployment plan into `btp/deployment_plan.md`

The Tech Stack stage displays 2-3 options with:

- name
- frontend
- backend
- database
- deployment target
- one-line rationale

The agent does not continue to Code until a stack option is selected.

## Skills

Skills are discovered from:

- built-in BTP/coding skills
- `.agents/skills/*/SKILL.md`
- `.codex/skills/*/SKILL.md`
- `.dev-council/skills/*.md`
- user-level equivalents under the home directory

Skill metadata is injected into the system prompt. Full skill content is loaded only when a skill is selected or invoked.

List skills:

```text
/skills
```

## MCP

MCP servers are configured through either a project `.mcp.json` file or the user-level config at:

```text
~/.dev-council/mcp.json
```

Config loading order:

1. user config: `~/.dev-council/mcp.json`
2. nearest project config: `.mcp.json`

If both define the same server name, the project `.mcp.json` wins.

Connected tools are registered as:

```text
mcp__<server>__<tool>
```

Useful commands:

```text
/mcp
/mcp reload
/mcp add demo <command> [args...]
```

### Project `.mcp.json` Example

Create `.mcp.json` in the repo root:

```json
{
  "mcpServers": {
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "."]
    }
  }
}
```

Then reload MCP inside the CLI:

```text
/mcp reload
/mcp
```

Expected output:

- the `git` server is listed
- connected tools appear under the server
- tool names look like `mcp__git__<tool>`

### User Config Example

To make a server available across projects, put it in `~/.dev-council/mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:/Users/Prasanna/OneDrive/Desktop"
      ]
    }
  }
}
```

You can also add a stdio server from the CLI:

```text
/mcp add demo uvx mcp-server-git --repository .
/mcp reload
```

This writes to the user-level MCP config.

### Remote MCP Example

HTTP/SSE-style servers can be configured with a URL and optional headers:

```json
{
  "mcpServers": {
    "remote": {
      "type": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

## Context Compaction

Manual compaction:

```text
/compact
```

Automatic compaction triggers before the next LLM call when context usage exceeds 80%. The CLI prints a notice like:

```text
⚠️ Context compaction triggered automatically (usage: 84%)
```

Responses include a footer:

```text
[Context: 67% used | ~14,200 / 21,000 tokens]
```

## Install

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python dev_council.py
```

The CLI opens with a large `dev-council` banner. Press `Ctrl+C` at any prompt or during a running turn to exit cleanly.

Or install the CLI entrypoint:

```bash
pip install .
dev-council
```

## Requirements

- Python `>=3.11, <3.13` per `pyproject.toml`
- Ollama installed for local model discovery
- At least one local Ollama model available through `ollama list`

## Project Layout

```text
dev_council.py      main CLI and pipeline orchestration
agent.py            core agent loop
providers.py        Ollama-compatible model transport
compaction.py       manual and automatic context compaction
context.py          system prompt builder
tools.py            built-in file, shell, web, memory, task, and plan tools
mcp/                MCP client and tool registration
skill/              skill loading and built-in BTP skills
memory/             persistent memory system
task/               task storage and helpers
checkpoint/         file checkpoints and rewind support
tests/              runtime and integration-style tests
```
