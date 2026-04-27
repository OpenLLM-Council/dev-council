from pathlib import Path

import dev_council
from config import CONFIG_DIR, DEFAULTS
from providers import bare_model, detect_provider
from skill.loader import load_skills


def test_config_uses_dev_council_home():
    assert CONFIG_DIR.name in {".dev-council", ".dev-council-home"}
    assert DEFAULTS["model"].startswith("local/")


def test_provider_detection_is_ollama_only():
    assert detect_provider("local/qwen2.5-coder") == "local"
    assert detect_provider("cloud/llama3.3") == "cloud"
    assert detect_provider("qwen2.5-coder") == "local"
    assert bare_model("cloud/llama3.3") == "llama3.3"


def test_builtin_skills_match_SDLC_flow():
    names = {skill.name for skill in load_skills()}
    assert {
        "srs",
        "milestones",
        "techstack",
        "qa",
        "deploy",
        "pipeline",
        "implementation-core",
        "frontend-builder",
        "backend-builder",
        "fullstack-builder",
        "testing-guard",
    } <= names


def test_large_product_intent_detection():
    assert dev_council._looks_like_large_product_request(
        "Build a full stack SaaS dashboard with auth and APIs"
    )
    assert not dev_council._looks_like_large_product_request("Create sha1.py")


def test_relevant_skill_selection_prefers_coding_skills():
    skills = dev_council._select_relevant_skills(
        "Create a React frontend dashboard and FastAPI backend with tests",
        force_coding=True,
    )
    names = {skill.name for skill in skills}
    assert "implementation-core" in names
    assert "frontend-builder" in names or "backend-builder" in names
