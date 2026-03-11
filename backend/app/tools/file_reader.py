import os
from langchain.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read the full contents of a single file.

    ALWAYS call this before modifying any existing file so you know exactly
    what is already there and can produce a correct SEARCH/REPLACE block.

    Args:
        file_path (str): Absolute or relative path to the file to read.

    Returns:
        str: The file contents, or an error message if the file is unavailable.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def read_directory(directory: str) -> str:
    """Read all text files in a directory and return their contents formatted for an LLM.

    Use this to get the full current codebase context when working on a new milestone
    or when you need to understand the existing code structure before making changes.

    Args:
        directory (str): Path to the directory to read.

    Returns:
        str: A string containing the relative path and content of each readable file.
    """
    if not os.path.exists(directory):
        return f"Directory does not exist: {directory}"

    skip_dirs = {".git", "__pycache__", "node_modules", "venv", ".venv", ".memory"}
    skip_exts = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe", ".pyc", ".ico", ".svg"}

    output = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in sorted(files):
            if file.startswith(".") or os.path.splitext(file)[1].lower() in skip_exts:
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                output.append(
                    f"### File: {relative_path}\n```{relative_path}\n{content}\n```\n"
                )
            except Exception as e:
                output.append(f"### File: {relative_path}\nError reading file: {e}\n")

    return "\n".join(output) if output else "No readable files found in directory."

def read_file_raw(file_path: str) -> str:
    """Non-tool version of read_file for use by internal Python code."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def read_code_directory(directory: str) -> str:
    """Non-tool version of read_directory for use by internal Python code."""
    return read_directory.invoke(directory)