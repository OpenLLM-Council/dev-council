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


def get_coder_llms() -> list[dict]:
    """Returns only coding-specialized LLMs (keys containing CODER or GRANITE),
    sorted so CODER models (larger, e.g. qwen2.5-coder:7b) come before GRANITE."""
    coder_keywords = {"CODER", "GRANITE"}
    llms = [
        {"name": key, "model": value}
        for key, value in os.environ.items()
        if key.endswith("_LLM") and any(kw in key.upper() for kw in coder_keywords)
    ]
    # Prefer keys with 'CODER' first (usually larger/better coding models)
    llms.sort(key=lambda x: (0 if "CODER" in x["name"].upper() else 1))
    return llms
