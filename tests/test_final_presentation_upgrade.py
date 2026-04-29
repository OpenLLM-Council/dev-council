import json
import sys
import uuid
from pathlib import Path

import compaction
import dev_council
import providers
import tools
from agent import AgentState
from mcp.client import HttpTransport, MCPClient
from mcp.types import INIT_PARAMS, MCPServerConfig, MCPTransport
from skill.loader import load_skills


def test_intent_detection_simple_and_big():
    assert not dev_council._looks_like_large_product_request("Create sha1.py")
    assert dev_council._looks_like_large_product_request(
        "Build a full stack SaaS dashboard with auth and APIs"
    )
    assert dev_council._looks_like_large_product_request(
        "We need an application for teams to manage projects, users, billing, reporting, "
        "notifications, onboarding, roles, permissions, audit logs, deployment, analytics, "
        "settings, dashboards, integrations, exports, imports, workflows, admin operations, "
        "team invitations, subscription plans, and operational reports."
    )


def test_simple_request_bypasses_pipeline(monkeypatch):
    called = {"direct": False, "pipeline": False, "model": False}

    monkeypatch.setattr(dev_council, "_run_agent_query", lambda *a, **k: called.__setitem__("direct", True))
    monkeypatch.setattr(dev_council, "_run_full_SDLC_cycle", lambda *a, **k: called.__setitem__("pipeline", True))
    monkeypatch.setattr(dev_council, "_run_model_selection_flow", lambda *a, **k: called.__setitem__("model", True))
    monkeypatch.setattr(dev_council, "_record_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(dev_council, "_print_context_footer", lambda *a, **k: None)

    assert dev_council._process_input("Fix typo in README", AgentState(), {"model": "local/test"})
    assert called == {"direct": True, "pipeline": False, "model": False}


def test_big_request_runs_model_selection_and_pipeline(monkeypatch):
    called = {"direct": False, "pipeline": False, "model": False}

    monkeypatch.setattr(dev_council, "_run_agent_query", lambda *a, **k: called.__setitem__("direct", True))
    monkeypatch.setattr(dev_council, "_run_full_SDLC_cycle", lambda *a, **k: called.__setitem__("pipeline", True))
    monkeypatch.setattr(dev_council, "_run_model_selection_flow", lambda *a, **k: called.__setitem__("model", True))
    monkeypatch.setattr(dev_council, "_record_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(dev_council, "_print_context_footer", lambda *a, **k: None)

    assert dev_council._process_input("Build a full stack SaaS dashboard", AgentState(), {"model": "local/test"})
    assert called == {"direct": False, "pipeline": True, "model": True}


def test_read_only_request_does_not_apply_coding_skills(monkeypatch):
    captured = {}

    def fake_run_agent_query(*args, **kwargs):
        captured["use_skills"] = kwargs.get("use_skills")

    monkeypatch.setattr(dev_council, "_run_agent_query", fake_run_agent_query)
    monkeypatch.setattr(dev_council, "_record_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(dev_council, "_print_context_footer", lambda *a, **k: None)

    dev_council._process_input("read the README.md and tell about this project", AgentState(), {"model": "local/test"})
    assert captured["use_skills"] is False


def test_model_single_mode_selection(monkeypatch):
    config = {"model": "local/old", "active_ollama_endpoint": "local"}
    answers = iter(["1", "2"])

    monkeypatch.setattr(dev_council, "ask_input_interactive", lambda *a, **k: next(answers))
    monkeypatch.setattr(dev_council, "_fetch_models_for_endpoint", lambda *a, **k: ["a:latest", "b:latest"])
    monkeypatch.setattr(dev_council, "save_config", lambda *a, **k: None)

    dev_council._run_model_selection_flow(config)
    assert config["llm_mode"] == "single"
    assert config["active_model"] == "local/b:latest"
    assert config["model"] == "local/b:latest"


def test_model_consensus_selection(monkeypatch):
    config = {"model": "local/old", "active_ollama_endpoint": "local"}
    # "2" = consensus mode, "3" = 3 models to vote, "1,3,2" = model selection, "n" = decline judge model
    answers = iter(["2", "3", "1,3,2", "n"])

    monkeypatch.setattr(dev_council, "ask_input_interactive", lambda *a, **k: next(answers))
    monkeypatch.setattr(dev_council, "_fetch_models_for_endpoint", lambda *a, **k: ["a", "b", "c"])
    monkeypatch.setattr(dev_council, "save_config", lambda *a, **k: None)

    dev_council._run_model_selection_flow(config)
    assert config["llm_mode"] == "consensus"
    assert config["consensus_models"] == ["local/a", "local/c", "local/b"]
    assert config["model"] == "local/a"
    assert "judge_model" not in config  # declined judge


def test_generation_prompt_consensus_continues_after_one_model_failure(monkeypatch):
    calls = []
    warnings = []

    def fake_run_text_prompt(prompt, config, model="", system="", **kwargs):
        calls.append(model)
        if model == "local/bad":
            raise RuntimeError("HTTP 500 from upstream")
        return f"reply from {model}"

    monkeypatch.setattr(dev_council, "_run_text_prompt", fake_run_text_prompt)
    monkeypatch.setattr(dev_council, "warn", warnings.append)

    result = dev_council._run_generation_prompt(
        "Write an SRS",
        {
            "llm_mode": "consensus",
            "consensus_models": ["local/bad", "local/good"],
            "model": "local/good",
        },
    )

    assert result == "reply from local/good"
    assert calls == ["local/bad", "local/good"]
    assert any("Consensus model failed: local/bad" in message for message in warnings)


def _repo_tmp_dir(name: str) -> Path:
    path = Path.cwd() / ".test-tmp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_read_tool_works_without_pathlib_newline_support():
    path = _repo_tmp_dir("read-tool") / "sample.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")

    result = tools._read(str(path))

    assert "alpha" in result
    assert "beta" in result
    assert "unexpected keyword argument 'newline'" not in result


def test_windows_retry_translates_simple_unix_read_commands():
    assert tools._windows_retry_command("ls README.md") == "dir README.md"
    assert tools._windows_retry_command("cat README.md") == "type README.md"
    assert tools._windows_retry_command("pwd") == "cd"
    assert tools._looks_like_windows_command_not_found(
        "'ls' is not recognized as an internal or external command"
    )


def test_tech_stack_selection_writes_chosen_option(monkeypatch):
    monkeypatch.chdir(_repo_tmp_dir("techstack"))
    output = """
Option 1 - MERN Stack
Frontend: React + Tailwind
Backend: Node.js + Express
DB: MongoDB
Deploy: Docker + Railway
Why: Fast for JavaScript teams

Option 2 - FastAPI + React
Frontend: React + Vite
Backend: Python + FastAPI
DB: PostgreSQL
Deploy: Docker + Fly.io
Why: Strong typed backend with simple deployment
"""
    monkeypatch.setattr(dev_council, "_run_generation_prompt", lambda *a, **k: output)
    monkeypatch.setattr(dev_council, "ask_input_interactive", lambda *a, **k: "2")

    path = dev_council._run_tech_stack_selection("Build a product", {"model": "local/test"})
    content = path.read_text(encoding="utf-8")
    assert "Option 2 - FastAPI + React" in content
    assert "Option 1 - MERN Stack" not in content


def test_manual_compact_replaces_messages(monkeypatch):
    state = AgentState()
    state.messages = [
        {"role": "user", "content": "one " * 100},
        {"role": "assistant", "content": "two " * 100},
        {"role": "user", "content": "three " * 100},
        {"role": "assistant", "content": "four " * 100},
    ]

    def fake_stream(*args, **kwargs):
        yield providers.TextChunk("summary")
        yield providers.AssistantTurn("summary", [], 1, 1)

    monkeypatch.setattr(providers, "stream", fake_stream)
    success, message = compaction.manual_compact(state, {"model": "local/test"})

    assert success
    assert "Compacted" in message
    assert state.messages[0]["content"].startswith("[Previous conversation summary]")


def test_auto_compaction_threshold_and_notice(monkeypatch):
    state = AgentState()
    state.messages = [{"role": "user", "content": "x" * 400}]
    notices = []

    monkeypatch.setattr(compaction, "get_context_limit", lambda model: 100)
    monkeypatch.setattr(compaction, "compact_messages", lambda messages, config, focus="": [{"role": "user", "content": "summary"}])

    assert compaction.maybe_compact(state, {"model": "local/test", "_auto_compact_notice": notices.append})
    assert notices == [113]
    assert state.messages == [{"role": "user", "content": "summary"}]


def test_context_footer_format():
    state = AgentState()
    state.messages = [{"role": "user", "content": "abcd" * 10}]
    footer = dev_council._context_footer(state, {"model": "local/test"})
    assert footer.startswith("[Context:")
    assert "tokens]" in footer


def test_tool_protocol_can_be_flattened_for_provider_fallback():
    messages = [
        {"role": "user", "content": "read README"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "Read"}]},
        {"role": "tool", "name": "Read", "content": "README contents"},
    ]
    flattened = providers.messages_to_ollama_plain(messages)

    assert flattened[1]["role"] == "assistant"
    assert "Tool calls requested: Read" in flattened[1]["content"]
    assert flattened[2]["role"] == "user"
    assert "README contents" in flattened[2]["content"]


def test_help_only_documents_final_surface(capsys):
    dev_council.cmd_help("", AgentState(), {})
    output = capsys.readouterr().out
    assert "/model" in output
    assert "/compact" in output
    assert "/skills" in output
    assert "/permission" in output
    assert "MCP Tools" in output
    assert "Memory" in output
    assert "Context" in output
    assert "Pipeline" in output
    assert "/doctor" not in output
    assert "/council" not in output


def test_permission_command_accepts_approve_all(monkeypatch):
    saved = []
    monkeypatch.setattr(dev_council, "save_config", lambda cfg: saved.append(dict(cfg)))
    config = {"permission_mode": "manual"}

    assert dev_council.cmd_permissions("approve all", AgentState(), config)

    assert config["permission_mode"] == "accept-all"
    assert saved[-1]["permission_mode"] == "accept-all"


def test_permission_command_interactive_manual(monkeypatch):
    saved = []
    monkeypatch.setattr(dev_council, "save_config", lambda cfg: saved.append(dict(cfg)))
    monkeypatch.setattr(dev_council, "ask_input_interactive", lambda *args, **kwargs: "2")
    config = {"permission_mode": "accept-all"}

    assert dev_council.handle_slash("/permission", AgentState(), config)

    assert config["permission_mode"] == "manual"
    assert saved[-1]["permission_mode"] == "manual"


def test_banner_prints_big_cli_title(capsys):
    dev_council._print_banner()
    output = capsys.readouterr().out
    assert "____" in output
    assert "dev-council" in output
    assert "Ctrl+C to exit" in output


def test_keyboard_interrupt_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dev_council.py"])
    monkeypatch.setattr(dev_council, "ask_input_interactive", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(dev_council.ckpt, "make_snapshot", lambda *a, **k: None)
    assert dev_council.main() == 0
    output = capsys.readouterr().out
    assert "Exiting dev-council." in output


def test_agents_skills_are_loaded():
    names = {skill.name for skill in load_skills()}
    assert {"code-reviewer", "context-engineering", "mcp-builder"} <= names


def test_mcp_stdio_discovers_and_calls_tool():
    server = _repo_tmp_dir("mcp") / "mcp_echo_server.py"
    server.write_text(
        """
import json
import sys

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "echo", "version": "1"}}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        tool = {"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [tool]}}), flush=True)
    elif method == "tools/call":
        text = msg.get("params", {}).get("arguments", {}).get("text", "")
        content = [{"type": "text", "text": "echo:" + text}]
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": content}}), flush=True)
""",
        encoding="utf-8",
    )
    client = MCPClient(MCPServerConfig(name="test", command=sys.executable, args=[str(server)], timeout=5))
    try:
        client.connect()
        tools = client.list_tools()
        assert tools[0].qualified_name == "mcp__test__echo"
        assert client.call_tool("echo", {"text": "hello"}) == "echo:hello"
    finally:
        client.disconnect()


def test_mcp_http_transport_parses_event_stream_response():
    transport = HttpTransport(
        MCPServerConfig(name="remote", transport=MCPTransport.HTTP, url="https://example.com/mcp", timeout=5)
    )

    class FakeResponse:
        headers = {"content-type": "text/event-stream"}
        text = (
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"remote","version":"1.0.0"}}}\n\n'
        )

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self):
            self.seen_headers = None

        def post(self, url, json, timeout):
            return FakeResponse()

    fake_client = FakeClient()
    transport._client = fake_client
    result = transport.request("initialize", INIT_PARAMS, timeout=5)

    assert result["protocolVersion"] == "2024-11-05"
