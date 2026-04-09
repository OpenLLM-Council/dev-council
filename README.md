# dev-council

`dev-council` is a minimal BTP coding CLI built around the Multi-Consensus Coding Agent workflow.

It keeps only the useful core systems:

- Ollama local and Ollama cloud model support
- BTP stage flow
- `/council` multi-model consensus coding
- single-model coding via `/model`
- automatic skill matching and application during coding
- MCP support
- skills
- tasks
- memory
- checkpoints
- plan mode

Everything else from the old broader assistant surface has been removed.

## BTP Flow

The runtime is organized around these stages:

1. User request analysis
2. SRS generation
3. Milestone generation
4. Technology stack selection
5. Consensus-based code generation
6. Quality assurance
7. Deployment and CI/CD preparation

## Commands

- `/model`
  Choose one Ollama model for normal coding.
- `/council`
  Ask how many models to use, fetch available models from `base_url/api/tags`, let you pick them, generate proposals, build consensus, then implement with the consensus model.
- direct user requests
  The runtime detects simple coding asks versus larger product asks. In an effectively empty folder, big SaaS/full-stack requests prompt for `full BTP cycle` or `direct code`.
- `/srs`
  Generate `btp/srs.md`.
- `/milestones`
  Generate `btp/milestones.md`.
- `/techstack`
  Generate `btp/tech_stack.md`.
- `/qa`
  Generate `btp/qa_report.md`.
- `/deploy`
  Generate `btp/deployment_plan.md`.
- `/pipeline`
  Generate the planning artifacts across the BTP stages.
- `/memory`, `/skills`, `/mcp`, `/tasks`, `/checkpoint`, `/plan`, `/status`, `/doctor`

## Model Support

Only two model sources are supported:

- `local/<model>`
- `cloud/<model>`

Both are expected to be Ollama-compatible and expose `POST /api/chat` and `GET /api/tags`.

## Skills

Relevant coding skills are auto-matched from the available skill metadata and applied compulsorily during implementation.

The terminal shows active skill usage like:

```text
[skills] Using: implementation-core, backend-builder
```

## Config

Config is stored in:

- `~/.dev-council/config.json`

Important keys:

- `ollama_local_base_url`
- `ollama_cloud_base_url`
- `ollama_cloud_api_key`
- `model`
- `active_ollama_endpoint`
- `permission_mode`

## Install

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python dev_council.py
```

Or install the CLI entrypoint:

```bash
pip install .
dev-council
```

## Project Layout

```text
dev_council.py      main CLI
agent.py            core agent loop
providers.py        Ollama local/cloud transport
tools.py            built-in file, shell, web, memory, task, plan tools
memory/             persistent memory system
mcp/                MCP client and tool registration
skill/              skill loading and built-in BTP skills
task/               task storage and helpers
checkpoint/         file checkpoints and rewind support
```

## Notes

- Old subagent support has been removed.
- Old plugin support has been removed.
- Old docs, demos, video, and voice modules have been removed.
- Non-Ollama providers have been removed.
