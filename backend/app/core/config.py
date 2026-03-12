"""
Settings loaded from ~/.dev-council/llm_config.json (written by `onboard`).
No .env file needed.
"""
import json
import os

CONFIG_DIR = os.path.expanduser("~/.dev-council")
CONFIG_FILE = os.path.join(CONFIG_DIR, "llm_config.json")


def _load_config() -> dict:
    """Read the onboarding config file. Returns {} if missing."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


_cfg = _load_config()
_models: list[str] = _cfg.get("ollama_models", [])


def _pick(keyword: str) -> str:
    """Return the first model whose name contains *keyword* (case-insensitive),
    or fall back to the first available model, or a sensible default."""
    kw = keyword.lower()
    for m in _models:
        if kw in m.lower():
            return m
    return _models[0] if _models else "qwen2.5:1.5b"


class Settings:
    OLLAMA_URL: str = _cfg.get("ollama_base_url", "http://localhost:11434")
    OLLAMA_TEMPERATURE: float = float(_cfg.get("ollama_temperature", 0))

    # Best-effort model selection from the onboarded list.
    # Each property tries to find a model matching its keyword,
    # then falls back to the first available model.
    GPT_LLM: str = _pick("gpt")
    QWEN_LLM: str = _pick("qwen")
    DEEPSEEK_LLM: str = _pick("deepseek")
    MISTRAL_LLM: str = _pick("mistral")


settings = Settings()


# Lazy firecrawl — only created when web_search.py actually imports it.
def _make_firecrawl():
    try:
        from firecrawl import Firecrawl
        return Firecrawl(api_url="http://localhost:3002/")
    except Exception:
        return None


class _LazyFirecrawl:
    _instance = None

    def __getattr__(self, name):
        if _LazyFirecrawl._instance is None:
            _LazyFirecrawl._instance = _make_firecrawl()
        return getattr(_LazyFirecrawl._instance, name)


firecrawl = _LazyFirecrawl()
