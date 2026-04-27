# Testing Suite

Use this guide to verify `dev-council` from both automated tests and hands-on CLI scenarios.

## 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Confirm Python and Ollama:

```bash
python --version
ollama list
```

Expected:

- Python is compatible with the project.
- `ollama list` prints at least one model.

## 2. Automated Tests

Run the full test suite:

```bash
python -m pytest -q
```

Expected:

```text
17 passed
```

Run syntax checks:

```powershell
$env:PYTHONPYCACHEPREFIX=(Join-Path (Get-Location) '.test-tmp\pycache')
python -m py_compile agent.py compaction.py config.py context.py dev_council.py providers.py tool_registry.py tools.py mcp\client.py mcp\config.py mcp\tools.py mcp\types.py skill\loader.py skill\tools.py skill\builtin.py
```

Expected:

- command exits with code `0`
- no syntax errors

## 3. Startup and Help

Run:

```bash
python dev_council.py --version
python dev_council.py
```

Inside the CLI:

```text
/help
```

Expected:

- version prints
- REPL starts
- a large `dev-council` ASCII banner appears
- startup text says `Press Ctrl+C to exit`
- `/help` only documents:
  - `/model`
  - `/compact`
  - `/skills`
  - `MCP Tools`
  - `Memory`
  - `Context`
  - `Pipeline`

Press `Ctrl+C` at the prompt.

Expected:

- the CLI prints `Exiting dev-council.`
- the process exits cleanly

## 4. Single LLM Selection

Inside the CLI:

```text
/model
```

Choose:

```text
1
```

Then pick a listed model number.

Expected:

- the CLI runs live local model discovery through `ollama list`
- available models are shown as a numbered list
- selection is saved as `active_model`
- `model` is updated to the selected `local/<model>`
- `/status` shows `LLM mode: single`

## 5. Consensus Selection

Inside the CLI:

```text
/model
```

Choose:

```text
2
```

Then enter a vote count and exactly that many comma-separated model numbers.

Example:

```text
3
1,2,3
```

Expected:

- selected models are saved as `consensus_models`
- `llm_mode` becomes `consensus`
- `/status` prints the selected consensus models

## 6. Simple Request Bypass

Inside the CLI, run a small request:

```text
What commands are available?
```

Or:

```text
Fix a typo in README.md
```

Expected:

- no SRS/Tech Stack/QA/Deployment pipeline starts
- the agent answers or handles the request directly
- a context footer appears after the response

## 7. Big Requirement Pipeline

Inside the CLI, run:

```text
Build a full stack SaaS CRM with auth, admin dashboard, customer records, API endpoints, and deployment setup.
```

Expected:

- model selection appears before the pipeline starts
- SRS is generated
- Tech Stack options are displayed as a numbered list
- the CLI waits for your stack selection
- after selection, Code, QA, and Deployment stages continue
- generated files appear under `SDLC/`

Check files:

```powershell
Get-ChildItem SDLC
```

Expected key files:

- `srs.md`
- `tech_stack.md`
- `qa_report.md`
- `deployment_plan.md`

## 8. Manual Pipeline Command

Inside the CLI:

```text
/pipeline Build a small issue tracking web app with login, projects, tickets, comments, and deployment.
```

Expected:

- the same model selection flow runs
- Tech Stack pauses for selection
- pipeline artifacts are written to `SDLC/`
- final status says the SDLC pipeline is complete

## 9. Tech Stack Selection

Trigger a pipeline or run:

```text
/techstack Build a task management SaaS with teams, projects, and notifications.
```

Expected option format:

```text
Option 1 - <name>
Frontend: ...
Backend: ...
DB: ...
Deploy: ...
Why: ...
```

Expected behavior:

- exactly 2-3 stack options are displayed
- the CLI asks for an option number
- `SDLC/tech_stack.md` contains only the selected option

## 10. Context Commands

Inside the CLI:

```text
/context
/compact
```

Expected:

- `/context` prints token usage and percentage
- `/compact` summarizes the active conversation when enough messages exist
- every command response is followed by a context footer

Auto-compaction is harder to trigger manually. It should print:

```text
⚠️ Context compaction triggered automatically (usage: NN%)
```

when usage exceeds 80% before an LLM call.

## 11. Skills

Inside the CLI:

```text
/skills
```

Expected:

- built-in SDLC/coding skills are listed
- project skills from `.agents/skills/*/SKILL.md` are listed, including:
  - `code-reviewer`
  - `context-engineering`
  - `mcp-builder`

You can also verify by script:

```bash
python -c "from skill.loader import load_skills; print([s.name for s in load_skills()])"
```

## 12. MCP

### Configure MCP

Project-level config goes in `.mcp.json` at the repo root. User-level config goes in:

```text
~/.dev-council/mcp.json
```

Project `.mcp.json` overrides user config when both define the same server name.

Example project `.mcp.json`:

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

Example user-level filesystem server:

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

You can also add a stdio server from inside the CLI:

```text
/mcp add demo uvx mcp-server-git --repository .
/mcp reload
```

### Verify MCP

List configured MCP servers:

```text
/mcp
```

Reload configured servers:

```text
/mcp reload
```

Expected:

- configured servers are listed
- connected servers show discovered tools under their status
- tool names use `mcp__<server>__<tool>`

The automated suite includes a stdio MCP fixture that proves discovery and tool calling without needing an external server.

### MCP Troubleshooting

If a server is not listed:

```text
/mcp reload
/mcp
```

If the server is listed but disconnected:

- confirm the command exists, for example `uvx --version` or `npx --version`
- confirm the package can run outside dev-council
- check that `.mcp.json` is valid JSON
- use an absolute path if the MCP server needs a filesystem root

## 13. Memory

Inside the CLI:

```text
/memory
/memory project
```

Expected:

- existing memories are listed or searched
- empty memory stores report no memories without failing

## 14. Checkpoints

Run a request that edits files, then:

```text
/checkpoint
```

Expected:

- snapshots are listed for the active session

Restore one:

```text
/checkpoint 1
```

Expected:

- file restore output appears
- a new checkpoint is created after restore

## 15. Documentation Checks

After changes, verify:

```bash
python -m pytest -q
python -c "import dev_council; from agent import AgentState; dev_council.cmd_help('', AgentState(), {'model':'local/test'})"
```

Expected:

- tests pass
- help output matches the final public surface

## Troubleshooting

- If `ollama list` is empty, pull or create a local model before testing `/model`.
- If pytest hits Windows cache permission errors, this project disables pytest cache through `pyproject.toml`; rerun from the repo root.
- If MCP servers do not appear, check `.mcp.json` and run `/mcp reload`.
- If a big request does not trigger the pipeline, include product-building language such as `build a full stack app`, `create a platform`, or `implement a system`.
