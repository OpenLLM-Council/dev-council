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
