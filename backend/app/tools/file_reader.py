import os
from langchain.tools import tool


_file_cache: dict = {}
MAX_FILE_CHARS = 3000

@tool
def read_file(file_path: str) -> str:
    """Read the full contents of a single file in the code directory.

    ALWAYS call this before modifying any existing file so you know exactly
    what is already there and can produce a correct SEARCH/REPLACE block.

    Args:
        file_path (str): Relative path to the file to read (e.g., 'backend/src/index.ts')

    Returns:
        str: The file contents, or an error message if the file is unavailable.
    """
    from app.agents.coder_agent import _code_dir

    if not _code_dir:
        return "Error: code directory not set."

    abs_path = os.path.abspath(os.path.join(_code_dir, file_path))
    if not abs_path.startswith(os.path.abspath(_code_dir)):
        return f"Error: Cannot access file {file_path}. Stick to the code/ directory."

    if abs_path in _file_cache:
        return _file_cache[abs_path]

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + f"\n...(truncated {len(content) - MAX_FILE_CHARS} chars)..."
            _file_cache[abs_path] = content
            return content
    except FileNotFoundError:
        return f"File not found: {file_path} (checked in codebase: {abs_path})"
    except Exception as e:
        return f"Error reading file {file_path}: {e}"


@tool
def read_directory(directory: str) -> str:
    """Read all text files in a directory and return their contents formatted for an LLM.

    Use this to get the full current codebase context when working on a new milestone
    or when you need to understand the existing code structure before making changes.

    Args:
        directory (str): Relative path to the directory to read (inside the code folder).

    Returns:
        str: A string containing the relative path and content of each readable file.
    """
    from app.agents.coder_agent import _code_dir

    if not _code_dir:
        return "Error: code directory not set."

    # Prevent reading outside the _code_dir
    abs_dir = os.path.abspath(os.path.join(_code_dir, directory))
    if not abs_dir.startswith(os.path.abspath(_code_dir)):
        return f"Error: Cannot access directory {directory}. Stick to the code/ directory."

    if not os.path.isdir(abs_dir):
        return f"Directory does not exist: {directory} (checked in codebase: {abs_dir})"

    skip_dirs = {".git", "__pycache__", "node_modules", "venv", ".venv", ".memory"}
    skip_exts = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe", ".pyc", ".ico", ".svg"}

    output = []
    for root, dirs, files in os.walk(abs_dir):
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