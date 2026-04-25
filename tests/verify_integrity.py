"""Final integrity verification script."""
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import providers
import tool_registry
import context
import tools
import dev_council
import agent
import config
import compaction

print("1. ALL IMPORTS OK")

# 2. Tool name normalization
from tool_registry import get_tool

t = get_tool("bash")
assert t is not None and t.name == "Bash", f"FAIL: bash resolved to {t}"
t2 = get_tool("write")
assert t2 is not None and t2.name == "Write", f"FAIL: write resolved to {t2}"
t3 = get_tool("EDIT")
assert t3 is not None and t3.name == "Edit", f"FAIL: EDIT resolved to {t3}"
t4 = get_tool("nonexistent")
assert t4 is None, "FAIL: nonexistent should be None"
print("2. TOOL NAME NORMALIZATION OK")

# 3. System prompt content
from context import build_system_prompt

prompt = build_system_prompt()
assert "Implementation Rules" in prompt, "FAIL: missing Implementation Rules"
assert "**Bash**" in prompt, "FAIL: missing bold Bash"
assert "**Write**" in prompt, "FAIL: missing bold Write"
assert "Do NOT use the Skill tool to run shell commands" in prompt, "FAIL: missing skill warning"
print("3. SYSTEM PROMPT CONTENT OK")

# 4. Pipeline checkpoint with judge_model
old_cwd = os.getcwd()
with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    cfg = {
        "model": "m",
        "consensus_models": ["a", "b"],
        "judge_model": "local/gemini-3:cloud",
        "llm_mode": "consensus",
    }
    dev_council._save_pipeline_state("test", ["srs", "techstack"], cfg)
    state = dev_council._load_pipeline_state()
    assert state["judge_model"] == "local/gemini-3:cloud", "FAIL: judge_model not saved"
    assert state["completed_stages"] == ["srs", "techstack"], "FAIL: stages not saved"
    dev_council._clear_pipeline_state()
    assert dev_council._load_pipeline_state() is None, "FAIL: clear didn't work"
    os.chdir(old_cwd)
print("4. PIPELINE CHECKPOINT WITH JUDGE MODEL OK")

# 5. Pipeline stages
assert dev_council._PIPELINE_STAGES == ["srs", "techstack", "code", "qa", "deploy"]
print("5. PIPELINE STAGES OK")

# 6. All essential functions
for fn in [
    "_run_full_btp_cycle", "cmd_pipeline", "_run_consensus_agent_query",
    "_run_generation_prompt", "_run_model_selection_flow", "_save_pipeline_state",
    "_load_pipeline_state", "_clear_pipeline_state", "_run_agent_query",
    "_run_text_prompt", "_run_council", "_choose_multiple_models",
]:
    assert hasattr(dev_council, fn), f"FAIL: missing {fn}"
print("6. ALL FUNCTIONS EXIST OK")

# 7. Provider error handling
assert hasattr(providers, "_http_error_details")
assert hasattr(providers, "messages_to_ollama_plain")
assert hasattr(providers, "_make_request")
assert providers._RETRYABLE_STATUS_CODES == {408, 429, 502, 503, 504}
print("7. PROVIDER ERROR HANDLING OK")

# 8. Execute tool with lowercase name
from tool_registry import execute_tool as reg_execute
result = reg_execute("bash", {"command": "echo integrity_check_pass"}, {})
assert "integrity_check_pass" in result, f"FAIL: bash execution returned: {result}"
print("8. EXECUTE TOOL LOWERCASE OK")

print()
print("=" * 50)
print("ALL 8 INTEGRITY CHECKS PASSED")
print("=" * 50)
