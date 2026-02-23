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
    for key, value in os.environ.items():
        if key.endswith("_LLM"):
            result.append(value)

    ollama_models = load_ollama_models()
    for m in ollama_models:
        result.append(m["model"])

    return str(result)


def get_available_llms() -> list[dict]:
    """Returns a list of available LLMs with their env var name and model.
    Used internally for dynamic graph node creation.
    Example: [{"name": "QWEN_LLM", "model": "qwen2.5:1.5b"}, ...]
    """
    llms = []
    for key, value in os.environ.items():
        if key.endswith("_LLM"):
            llms.append({"name": key, "model": value})

    llms.extend(load_ollama_models())

    return llms
