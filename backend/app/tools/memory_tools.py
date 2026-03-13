"""
Memory persistence tools for tracking project progress and state across sessions.
All memory data is stored in .memory/memory.json for persistence.
"""
import os
import json
from typing import Any
from langchain.tools import tool

from app.tools.path_utils import ensure_within_workspace, get_code_dir


def _get_memory_dir(project_path: str) -> str:
    """Ensure .memory directory exists and return path."""
    memory_dir = os.path.join(project_path, ".memory")
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir


def _resolve_project_path(project_path: str) -> tuple[str, str | None]:
    """Resolve project_path relative to the active code directory when available."""
    code_dir = get_code_dir()
    base_dir = os.path.dirname(code_dir) if code_dir else None
    return ensure_within_workspace(project_path, base_dir=base_dir)


@tool
def update_memory(project_path: str, topic: str, data: Any) -> str:
    """
    Persists structured data to the agent's memory store for this project.
    Use to save progress, decisions, or context across sessions.
    
    Data is stored as JSON in .memory/memory.json under the given topic key.
    If project_path is relative, it's resolved from the current working directory.
    
    Args:
        project_path: Root path of the project (absolute or relative; where .memory/ is/will be created)
        topic: Key name for the data (e.g., 'progress', 'architecture', 'todo', 'milestone_1')
        data: Any JSON-serializable data (str, dict, list, int, bool, etc.)
    
    Examples:
        update_memory(project_path, "progress", "Milestone 1: Core setup complete")
        update_memory(project_path, "architecture", {"layers": ["backend", "frontend"]})
    """
    try:
        project_path, path_error = _resolve_project_path(project_path)
        if path_error:
            return path_error

        if isinstance(data, (dict, list)):
            data = json.dumps(data, indent=4)
        else:
            data = str(data)
        
        mem_file = os.path.join(_get_memory_dir(project_path), "memory.json")
        mem_data: dict = {}
        
        # Load existing memory if present
        if os.path.exists(mem_file):
            with open(mem_file, "r", encoding="utf-8") as f:
                try:
                    mem_data = json.load(f)
                except (ValueError, json.JSONDecodeError):
                    pass
        
        # Update or add topic
        mem_data[topic] = data
        
        # Persist to disk
        with open(mem_file, "w", encoding="utf-8") as f:
            json.dump(mem_data, f, indent=4)
        
        return f"✓ Memory topic '{topic}' saved to {mem_file}"
    except Exception as e:
        return f"Error saving memory: {e}"


@tool
def get_memory(project_path: str, topic: str) -> str:
    """
    Retrieves a saved memory topic for this project.
    
    Args:
        project_path: Root path of the project (absolute or relative; where .memory/ is located)
        topic: Key name for the data to retrieve (e.g., 'progress', 'architecture')
    
    Returns:
        The stored data as a string, or an error/not-found message.
    
    Examples:
        get_memory(project_path, "progress")
        get_memory(project_path, "architecture")
    """
    try:
        project_path, path_error = _resolve_project_path(project_path)
        if path_error:
            return path_error

        mem_file = os.path.join(_get_memory_dir(project_path), "memory.json")
        
        if not os.path.exists(mem_file):
            return "No memory saved for this project yet. Start by calling update_memory."
        
        with open(mem_file, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        
        value = mem_data.get(topic)
        if value is None:
            available = ", ".join(mem_data.keys()) if mem_data else "none"
            return f"Topic '{topic}' not found. Available topics: {available}"
        
        return str(value)
    except Exception as e:
        return f"Error retrieving memory: {e}"


def read_all_memory(project_path: str) -> dict:
    """
    Helper function (not a tool) to read all memory topics at once.
    Useful for agents that want to check full state before proceeding.
    
    Returns:
        Dictionary of all topics -> values, or empty dict if no memory exists.
    """
    try:
        project_path, path_error = _resolve_project_path(project_path)
        if path_error:
            return {}
        mem_file = os.path.join(_get_memory_dir(project_path), "memory.json")
        if not os.path.exists(mem_file):
            return {}
        with open(mem_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
