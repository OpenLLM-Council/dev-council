"""Tests for tool name normalization, judge model, and providers robustness."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Tool name normalization tests ─────────────────────────────────────────────

class TestToolNameNormalization:
    """Verify case-insensitive tool lookup in tool_registry."""

    def test_exact_name_lookup(self):
        from tool_registry import get_tool
        import tools  # ensure registration
        assert get_tool("Bash") is not None
        assert get_tool("Read") is not None
        assert get_tool("Write") is not None

    def test_lowercase_name_lookup(self):
        from tool_registry import get_tool
        import tools
        assert get_tool("bash") is not None
        assert get_tool("read") is not None
        assert get_tool("write") is not None

    def test_uppercase_name_lookup(self):
        from tool_registry import get_tool
        import tools
        assert get_tool("BASH") is not None
        assert get_tool("WRITE") is not None

    def test_resolved_name_matches_canonical(self):
        """Case-insensitive lookup should return the tool with the canonical name."""
        from tool_registry import get_tool
        import tools
        tool = get_tool("bash")
        assert tool is not None
        assert tool.name == "Bash"

    def test_nonexistent_tool_returns_none(self):
        from tool_registry import get_tool
        import tools
        assert get_tool("nonexistent_tool_xyz") is None

    def test_execute_tool_lowercase_works(self):
        """execute_tool in tool_registry should work with lowercase names."""
        from tool_registry import execute_tool
        import tools
        # Calling 'bash' (lowercase) with a safe command should work
        result = execute_tool("bash", {"command": "echo hello"}, {})
        # Should NOT return "Error: tool 'bash' not found."
        assert "not found" not in result.lower()

    def test_skill_case_insensitive(self):
        from tool_registry import get_tool
        import tools
        assert get_tool("skill") is not None
        assert get_tool("Skill") is not None
        tool = get_tool("skill")
        assert tool.name == "Skill"


# ── Judge model config tests ─────────────────────────────────────────────────

_TEST_DIR = Path(__file__).resolve().parent.parent / ".test-tmp" / "judge-model"


@pytest.fixture
def _use_judge_test_dir(monkeypatch):
    if _TEST_DIR.exists():
        shutil.rmtree(_TEST_DIR, ignore_errors=True)
    _TEST_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(_TEST_DIR)
    yield
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


class TestJudgeModel:
    """Verify judge_model is saved/restored in pipeline checkpoints."""

    def test_checkpoint_saves_judge_model(self, _use_judge_test_dir):
        from dev_council import _save_pipeline_state, _load_pipeline_state
        config = {
            "model": "local/model-a",
            "llm_mode": "consensus",
            "consensus_models": ["local/model-a", "local/model-b"],
            "judge_model": "local/gemini-3-flash-preview:cloud",
        }
        _save_pipeline_state("test query", ["srs"], config)
        state = _load_pipeline_state()
        assert state["judge_model"] == "local/gemini-3-flash-preview:cloud"

    def test_checkpoint_without_judge_model(self, _use_judge_test_dir):
        from dev_council import _save_pipeline_state, _load_pipeline_state
        config = {"model": "local/model-a", "llm_mode": "single"}
        _save_pipeline_state("test query", ["srs"], config)
        state = _load_pipeline_state()
        assert state["judge_model"] == ""

    def test_judge_model_fallback_to_empty(self, _use_judge_test_dir):
        """When no judge_model in config, saved value should be empty string."""
        from dev_council import _save_pipeline_state, _load_pipeline_state
        config = {"model": "m", "consensus_models": ["a", "b"]}
        _save_pipeline_state("q", ["srs", "techstack"], config)
        state = _load_pipeline_state()
        assert state["judge_model"] == ""


# ── Providers robustness tests ────────────────────────────────────────────────

class TestProvidersEdgeCases:
    """Verify providers module handles edge cases."""

    def test_messages_to_ollama_plain_empty(self):
        from providers import messages_to_ollama_plain
        assert messages_to_ollama_plain([]) == []

    def test_messages_to_ollama_plain_user_only(self):
        from providers import messages_to_ollama_plain
        result = messages_to_ollama_plain([{"role": "user", "content": "hello"}])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_messages_to_ollama_plain_strips_tools(self):
        from providers import messages_to_ollama_plain
        messages = [
            {"role": "user", "content": "build something"},
            {
                "role": "assistant",
                "content": "I'll create files.",
                "tool_calls": [{"name": "Write", "input": {"file_path": "a.py", "content": "x"}}],
            },
            {"role": "tool", "name": "Write", "content": "Created a.py"},
        ]
        result = messages_to_ollama_plain(messages)
        assert len(result) == 3
        # Assistant message should have tool annotation as text
        assert "Tool calls requested: Write" in result[1]["content"]
        # Tool result should be converted to user message
        assert result[2]["role"] == "user"
        assert "Tool result from Write" in result[2]["content"]

    def test_http_error_details_with_body(self):
        """_http_error_details extracts body from HTTPError."""
        from providers import _http_error_details
        import io
        import urllib.error

        # Create a mock HTTPError with a readable body
        body = io.BytesIO(b"model not found")
        err = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=body,
        )
        details = _http_error_details(err)
        assert "400" in details
        assert "model not found" in details

    def test_detect_provider_defaults_to_local(self):
        from providers import detect_provider
        assert detect_provider("some-model") == "local"
        assert detect_provider("local/model") == "local"
        assert detect_provider("cloud/model") == "cloud"

    def test_bare_model_strips_provider(self):
        from providers import bare_model
        assert bare_model("local/gemini-3-flash") == "gemini-3-flash"
        assert bare_model("gemini-3-flash") == "gemini-3-flash"


# ── System prompt tests ──────────────────────────────────────────────────────

class TestSystemPrompt:
    """Verify system prompt contains critical tool guidance."""

    def test_prompt_mentions_bash_capitalization(self):
        from context import build_system_prompt
        prompt = build_system_prompt()
        assert "Bash" in prompt
        assert 'NOT "bash"' in prompt or "not \"bash\"" in prompt.lower()

    def test_prompt_has_implementation_rules(self):
        from context import build_system_prompt
        prompt = build_system_prompt()
        assert "Implementation Rules" in prompt

    def test_prompt_warns_against_skill_for_bash(self):
        from context import build_system_prompt
        prompt = build_system_prompt()
        assert "Do NOT use the Skill tool to run shell commands" in prompt

    def test_prompt_lists_write_tool(self):
        from context import build_system_prompt
        prompt = build_system_prompt()
        assert "**Write**" in prompt
