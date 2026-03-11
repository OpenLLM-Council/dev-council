import os
from langchain.tools import tool


@tool
def show_tree(start_path: str) -> str:
    """Show the file tree of the given directory path.

    Args:
        start_path (str): The path to show the file tree of.

    Returns:
        str: The file tree of the given path as a formatted string.
    """
    if not os.path.exists(start_path):
        return f"Path does not exist: {start_path}"

    skip = {".git", "__pycache__", "node_modules", "venv", ".venv", ".memory"}
    lines = []

    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in skip]

        level = root.replace(start_path, "").count(os.sep)
        indent = "    " * level
        lines.append(f"{indent}{os.path.basename(root)}/")

        sub_indent = "    " * (level + 1)
        for f in sorted(files):
            lines.append(f"{sub_indent}{f}")

    return "\n".join(lines) if lines else "(empty directory)"