import os
import json
from langchain_core.tools import tool

CONFIG_DIR = os.path.expanduser("~/.dev-council")
CONFIG_FILE = os.path.join(CONFIG_DIR, "llm_config.json")


def load_ollama_models() -> list[dict]:
    """Helper to load Ollama models from config file."""
    models = []
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                ollama_models = config.get("ollama_models", [])
                for model in ollama_models:
                    safe_name = f"OLLAMA_{model.replace(':', '_').replace('.', '_').replace('-', '_').upper()}_LLM"
                    models.append({"name": safe_name, "model": model})
        except Exception:
            pass
    return models


@tool
def list_llms(query: str = "") -> str:
    """Useful for getting available team of LLMs.
    Action Input must be an empty string."""
    result = []
    ollama_models = load_ollama_models()
    for m in ollama_models:
        result.append(m["model"])
    return str(result)


def get_available_llms() -> list[dict]:
    """Returns a list of available LLMs with their config name and model.
    Used internally for dynamic graph node creation.
    Example: [{"name": "OLLAMA_QWEN2_5_1_5B_LLM", "model": "qwen2.5:1.5b"}, ...]
    """
    return load_ollama_models()


def get_coder_llms() -> list[dict]:
    """Returns available LLMs, prioritizing coding-specialized LLMs first (e.g. CODER or GRANITE)
    if they appear in the name string, but surfacing all models."""
    llms = load_ollama_models()
    coder_keywords = ["CODER", "GRANITE"]

    def _priority(model_info):
        name_upper = model_info["name"].upper()
        if any(kw in name_upper for kw in coder_keywords):
            return 0
        return 1

    llms.sort(key=_priority)
    return llms
