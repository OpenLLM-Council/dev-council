"""Configuration management for dev-council."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _resolve_config_dir() -> Path:
    env_home = os.environ.get("DEV_COUNCIL_HOME")
    candidates = []
    if env_home:
        candidates.append(Path(env_home).expanduser())
    candidates.append(Path.home() / ".dev-council")
    candidates.append(Path.cwd() / ".dev-council-home")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return Path.cwd() / ".dev-council-home"


CONFIG_DIR = _resolve_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "input_history.txt"
SESSIONS_DIR = CONFIG_DIR / "sessions"
DAILY_DIR = SESSIONS_DIR / "daily"
SESSION_HIST_FILE = SESSIONS_DIR / "history.json"
MR_SESSION_DIR = SESSIONS_DIR / "mr_sessions"


DEFAULTS = {
    "model": "local/qwen2.5-coder:latest",
    "max_tokens": 32000,
    "permission_mode": "auto",
    "verbose": False,
    "thinking": False,
    "thinking_budget": 8000,
    "max_tool_output": 32000,
    "session_daily_limit": 10,
    "session_history_limit": 200,
    "ollama_local_base_url": "http://localhost:11434",
    "ollama_cloud_base_url": "",
    "ollama_cloud_api_key": "",
    "active_ollama_endpoint": "local",
}


def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {key: value for key, value in cfg.items() if not key.startswith("_")}
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def current_provider(cfg: dict) -> str:
    from providers import detect_provider

    return detect_provider(cfg.get("model", DEFAULTS["model"]))


def has_api_key(cfg: dict) -> bool:
    from providers import detect_provider, get_api_key

    provider_name = detect_provider(cfg.get("model", DEFAULTS["model"]))
    return bool(get_api_key(provider_name, cfg))


def calc_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    from providers import calc_cost as _calc_cost

    return _calc_cost(model, in_tokens, out_tokens)
