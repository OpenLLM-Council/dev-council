import os
import json
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Union


@tool
def save_file(file_name: str, content: Union[str, dict, BaseModel]) -> str:
    """
    Useful for saving files. Supports string, dict, or Pydantic model content.
    Args:
        file_name (str): The name of the file to save.
        content (Union[str, dict, BaseModel]): The content to save to the file.
    """
    os.makedirs("outputs", exist_ok=True)

    # Handle Pydantic models
    if isinstance(content, BaseModel):
        content = content.model_dump_json(indent=2)
    # Handle dicts
    elif isinstance(content, dict):
        content = json.dumps(content, indent=2)
    # Otherwise assume string

    with open(f"outputs/{file_name}", "w", encoding="utf-8") as f:
        f.write(content)
    return f"File saved to outputs/{file_name}"
