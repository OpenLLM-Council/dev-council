"""Tests for pipeline checkpoint save/resume system."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dev_council import (
    _save_pipeline_state,
    _load_pipeline_state,
    _clear_pipeline_state,
    _pipeline_state_path,
    _PIPELINE_STAGES,
)

# Use a workspace-local temp dir to avoid Windows permission errors with tmp_path
_TEST_DIR = Path(__file__).resolve().parent.parent / ".test-tmp" / "pipeline-ckpt"


@pytest.fixture(autouse=True)
def _use_test_dir(monkeypatch):
    """Run every test inside a workspace-local temp directory."""
    if _TEST_DIR.exists():
        shutil.rmtree(_TEST_DIR, ignore_errors=True)
    _TEST_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(_TEST_DIR)
    yield
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


class TestPipelineCheckpoint:
    """Pipeline state persistence."""

    def test_save_and_load(self):
        _save_pipeline_state("build a todo app", ["srs", "techstack"], {"model": "test/model"})
        state = _load_pipeline_state()
        assert state is not None
        assert state["query"] == "build a todo app"
        assert state["completed_stages"] == ["srs", "techstack"]
        assert state["model"] == "test/model"
        assert "saved_at" in state

    def test_load_returns_none_when_no_checkpoint(self):
        assert _load_pipeline_state() is None

    def test_clear_removes_checkpoint(self):
        _save_pipeline_state("test query", ["srs"], {"model": "m"})
        assert _load_pipeline_state() is not None
        _clear_pipeline_state()
        assert _load_pipeline_state() is None

    def test_clear_no_error_when_no_checkpoint(self):
        _clear_pipeline_state()  # should not raise

    def test_save_creates_SDLC_dir(self):
        """Even if SDLC/ doesn't exist, save should create it."""
        sub = _TEST_DIR / "fresh_sub"
        sub.mkdir()
        old_cwd = os.getcwd()
        os.chdir(sub)
        try:
            _save_pipeline_state("q", [], {"model": "m"})
            assert _load_pipeline_state() is not None
        finally:
            os.chdir(old_cwd)

    def test_corrupted_checkpoint_returns_none(self):
        path = _pipeline_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT VALID JSON", encoding="utf-8")
        assert _load_pipeline_state() is None

    def test_state_preserves_consensus_models(self):
        config = {
            "model": "local/gemini-3-flash-preview:cloud",
            "llm_mode": "consensus",
            "consensus_models": ["local/model-a", "local/model-b"],
        }
        _save_pipeline_state("test", ["srs"], config)
        state = _load_pipeline_state()
        assert state["llm_mode"] == "consensus"
        assert state["consensus_models"] == ["local/model-a", "local/model-b"]

    def test_pipeline_stages_order(self):
        """Verify the canonical stage ordering."""
        assert _PIPELINE_STAGES == ["srs", "techstack", "code", "qa", "deploy"]

    def test_remaining_stages_calculation(self):
        """Verify remaining stages are correctly derived."""
        completed = ["srs", "techstack"]
        remaining = [s for s in _PIPELINE_STAGES if s not in completed]
        assert remaining == ["code", "qa", "deploy"]

    def test_all_completed(self):
        completed = list(_PIPELINE_STAGES)
        remaining = [s for s in _PIPELINE_STAGES if s not in completed]
        assert remaining == []

    def test_overwrite_existing_checkpoint(self):
        """Saving again overwrites the previous checkpoint."""
        _save_pipeline_state("q1", ["srs"], {"model": "m1"})
        _save_pipeline_state("q2", ["srs", "techstack"], {"model": "m2"})
        state = _load_pipeline_state()
        assert state["query"] == "q2"
        assert state["completed_stages"] == ["srs", "techstack"]
