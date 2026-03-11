import os
from langchain.tools import tool


@tool
def show_tree(start_path: str = ".") -> str:
    """Show the file tree of the project code directory.

    Args:
        start_path (str): Relative path within the code directory to show (default: root of code dir).

    Returns:
        str: The file tree of the given path as a formatted string.
    """
    from app.agents.coder_agent import _code_dir

    if not _code_dir:
        return "Error: code directory not set."

    abs_path = os.path.abspath(os.path.join(_code_dir, start_path))
    if not abs_path.startswith(os.path.abspath(_code_dir)):
        return f"Error: Cannot access directory {start_path}. Stick to the code/ directory."

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