import os
from langchain.tools import tool

from app.tools.path_utils import ensure_within_workspace, get_code_dir


@tool
def show_tree(start_path: str = ".") -> str:
    """Show the file tree of the project code directory.

    Args:
        start_path (str): Relative path within the code directory to show (default: root of code dir).

    Returns:
        str: The file tree of the given path as a formatted string.
    """
    code_dir = get_code_dir()
    if not code_dir:
        return "Error: code directory not set."

    abs_path, path_error = ensure_within_workspace(start_path, base_dir=code_dir)
    if path_error:
        return path_error

    if not os.path.exists(abs_path):
        return f"Path does not exist: {start_path} (checked in codebase: {abs_path})"

    skip = {".git", "__pycache__", "node_modules", "venv", ".venv", ".memory"}
    lines = []

    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if d not in skip]

        level = root.replace(abs_path, "").count(os.sep)
        indent = "    " * level
        lines.append(f"{indent}{os.path.basename(root)}/")

        sub_indent = "    " * (level + 1)
        for f in sorted(files):
            lines.append(f"{sub_indent}{f}")

    return "\n".join(lines) if lines else "(empty directory)"
