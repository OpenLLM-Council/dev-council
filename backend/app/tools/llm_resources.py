import os
from langchain_core.tools import tool


@tool
def list_llms(query: str = "") -> str:
    """Useful for getting available team of LLMs.
    Action Input must be an empty string."""
    result = []
    for key, value in os.environ.items():
        if key.endswith("_LLM"):
            result.append(value)
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
    return llms
