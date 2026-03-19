import os
import re
import difflib
from typing import List
from langchain.tools import tool

from app.tools.path_utils import ensure_within_workspace, get_code_dir


_file_cache: dict = {}
MAX_FILE_CHARS = 3000
_SEARCH_SKIP_DIRS = {".git", ".memory", ".venv", "__pycache__", "node_modules", ".mypy_cache"}
_SEARCH_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".exe", ".pyc", ".ico",
    ".svg", ".lock", ".zip", ".tar", ".gz", ".woff", ".woff2",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_lines(filepath: str) -> list[str]:
    """Read file as list of lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.readlines()


def _normalize_path(path: str) -> tuple[str, str | None]:
    """Convert relative paths to absolute and keep them inside the active workspace."""
    return ensure_within_workspace(path)


def _assert_is_file(path: str) -> "str | None":
    """Returns an error string if path is not a regular file, else None."""
    path, path_error = _normalize_path(path)
    if path_error:
        return path_error
    if not os.path.exists(path):
        return f"Error: '{path}' does not exist."
    if os.path.isdir(path):
        return (
            f"Error: '{path}' is a directory, not a file. "
            "Use get_project_tree to list files, then pass a specific file path."
        )
    return None


def _nearby_hint(content: str, old_text: str) -> str:
    """Finds the closest matching block and returns a hint."""
    old_lines = old_text.splitlines()
    content_lines = content.splitlines()
    best_ratio, best_start = 0.0, 0
    window = max(len(old_lines), 1)

    for i in range(max(1, len(content_lines) - window + 1)):
        block = content_lines[i : i + window]
        ratio = difflib.SequenceMatcher(None, old_lines, block).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i

    if best_ratio < 0.3:
        return ""

    snippet = "\n".join(
        f"  L{best_start + j + 1}: {line}"
        for j, line in enumerate(content_lines[best_start : best_start + window])
    )
    return (
        f"\nClosest match at line ~{best_start + 1} (similarity {best_ratio:.0%}):\n"
        f"{snippet}\n"
        "-> Check indentation/whitespace, or use get_lines on those lines.\n"
    )


# ---------------------------------------------------------------------------
# EXPLORE tools
# ---------------------------------------------------------------------------

@tool
def get_project_tree(project_path: str) -> str:
    """
    Returns a directory tree of the project.
    Skips .git, .memory, .venv, __pycache__, node_modules.
    Always call this first to understand the workspace.
    """
    project_path, path_error = _normalize_path(project_path)
    if path_error:
        return path_error
    if not os.path.exists(project_path):
        return f"Error: Path '{project_path}' does not exist."

    lines = [f"Project: {os.path.abspath(project_path)}"]

    for root, dirs, files in os.walk(project_path):
        dirs[:] = sorted(d for d in dirs if d not in _SEARCH_SKIP_DIRS and not d.startswith("."))
        level = root.replace(project_path, "").count(os.sep)
        lines.append(f"{'    ' * level}{os.path.basename(root)}/")
        for f in sorted(files):
            lines.append(f"{'    ' * (level + 1)}{f}")

    return "\n".join(lines)


@tool
def search_in_project(
    project_path: str,
    pattern: str,
    is_regex: bool = False,
    context_lines: int = 1,
    max_results: int = 50,
) -> str:
    """
    Search across the project for a string or regex and return file/line matches.

    Use this when you know the symbol or text to change but not the exact file yet.
    """
    project_path, path_error = _normalize_path(project_path)
    if path_error:
        return path_error
    if not os.path.isdir(project_path):
        return f"Error: '{project_path}' is not a directory."

    try:
        regex = re.compile(pattern) if is_regex else None
    except re.error as e:
        return f"Invalid regex: {e}"

    matches: list[str] = []
    file_hits = 0

    for root, dirs, files in os.walk(project_path):
        dirs[:] = sorted(d for d in dirs if d not in _SEARCH_SKIP_DIRS and not d.startswith("."))
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext in _SEARCH_SKIP_EXTS:
                continue

            filepath = os.path.join(root, filename)
            try:
                lines = _read_lines(filepath)
            except (UnicodeDecodeError, OSError):
                continue

            file_match_blocks = []
            for i, line in enumerate(lines):
                matched = bool(regex.search(line)) if regex else pattern in line
                if not matched:
                    continue

                lo = max(0, i - context_lines)
                hi = min(len(lines) - 1, i + context_lines)
                block = [f"{os.path.relpath(filepath, project_path)}:"]
                for j in range(lo, hi + 1):
                    marker = ">>>" if j == i else "   "
                    block.append(f"  {marker} L{j + 1}: {lines[j].rstrip()}")
                file_match_blocks.append("\n".join(block))

                if len(matches) + len(file_match_blocks) >= max_results:
                    break

            if file_match_blocks:
                file_hits += 1
                matches.extend(file_match_blocks)

            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches for '{pattern}' in '{project_path}'."

    suffix = ""
    if len(matches) >= max_results:
        suffix = f"\n\n... results capped at {max_results}; refine the pattern if needed."

    return (
        f"Found {len(matches)} match(es) across {file_hits} file(s) for '{pattern}' in '{project_path}'.\n\n"
        + "\n\n".join(matches)
        + suffix
    )


@tool
def search_in_file(
    filepath: str,
    pattern: str,
    is_regex: bool = False,
    context_lines: int = 2,
) -> str:
    """
    Searches for a string (or regex) inside a FILE and returns matching line numbers
    plus surrounding context. filepath MUST be a file path, not a directory.

    Use context_lines=3 when searching for a function definition to see its full signature.
    Use context_lines=0 when just collecting call-site line numbers for delete_lines.

    Returns up to 20 matches. Use a more specific pattern if you get too many.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err

    try:
        lines = _read_lines(filepath)
        total = len(lines)
        hit_indices: list[int] = []

        for i, line in enumerate(lines):
            matched = bool(re.search(pattern, line)) if is_regex else pattern in line
            if matched:
                hit_indices.append(i)
                if len(hit_indices) >= 20:
                    break

        if not hit_indices:
            return f"No matches for '{pattern}' in '{filepath}'."

        parts = [
            f"Found {len(hit_indices)} match(es) for '{pattern}' in '{filepath}' "
            f"(total {total} lines):\n"
        ]
        for idx in hit_indices:
            lo = max(0, idx - context_lines)
            hi = min(total - 1, idx + context_lines)
            block = []
            for j in range(lo, hi + 1):
                marker = ">>>" if j == idx else "   "
                block.append(f"  {marker} L{j+1}: {lines[j].rstrip()}")
            parts.append("\n".join(block))

        if len(hit_indices) == 20:
            parts.append("  ... (capped at 20 -- refine pattern if needed)")

        return "\n\n".join(parts)
    except re.error as e:
        return f"Invalid regex: {e}"
    except Exception as e:
        return f"Error searching file: {e}"


@tool
def get_file_length(filepath: str) -> str:
    """
    Returns the total number of lines in a file.
    Use before read_file_chunk on large files to plan chunk ranges.
    Not needed before read_whole_file or delete_lines.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err
    try:
        return f"'{filepath}' has {len(_read_lines(filepath))} lines."
    except Exception as e:
        return f"Error: {e}"


@tool
def read_whole_file(filepath: str) -> str:
    """
    Reads the entire file with line numbers prefixed.
    Use when you need full context before adding or refactoring code (files <= 300 lines).
    Refuses files over 300 lines -- use read_file_chunk for those.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err
    try:
        lines = _read_lines(filepath)
        total = len(lines)
        if total > 300:
            return (
                f"'{filepath}' has {total} lines -- too large for read_whole_file. "
                "Use get_file_length then read_file_chunk in overlapping 60-line windows."
            )
        numbered = "".join(f"L{i+1:>4}: {line}" for i, line in enumerate(lines))
        return f"--- '{filepath}' ({total} lines) ---\n{numbered}--- end ---"
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file_chunk(filepath: str, start_line: int, end_line: int) -> str:
    """
    Reads a specific line range from a file (1-indexed, inclusive).
    Use for files over 300 lines where read_whole_file is refused.
    Use overlapping ranges (e.g. 1-70, 60-130) to avoid cutting functions in half.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err
    try:
        lines = _read_lines(filepath)
        total = len(lines)
        start = max(1, start_line)
        end = min(total, end_line)
        if start > end:
            return f"Error: start_line ({start}) > end_line ({end})."
        numbered = "".join(
            f"L{i:>4}: {line}"
            for i, line in enumerate(lines[start - 1 : end], start=start)
        )
        return f"--- '{filepath}' lines {start}-{end} of {total} ---\n{numbered}--- end of chunk ---"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_lines(filepath: str, line_numbers: List[int]) -> str:
    """
    Returns the exact content of specific line numbers (max 50).
    Use this instead of read_file_chunk when you only need a few exact lines
    to copy verbatim into old_text for replace_in_file.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err
    try:
        lines = _read_lines(filepath)
        total = len(lines)
        unique_numbers = sorted(set(line_numbers))[:50]
        if not unique_numbers:
            return "Error: No line numbers were provided."

        invalid = [n for n in unique_numbers if not (1 <= n <= total)]
        if invalid:
            return (
                f"Error: line number(s) {invalid} out of range for '{filepath}' "
                f"(file has {total} lines)."
            )

        blocks = []
        start = prev = unique_numbers[0]
        for n in unique_numbers[1:]:
            if n == prev + 1:
                prev = n
                continue
            blocks.append((start, prev))
            start = prev = n
        blocks.append((start, prev))

        parts = []
        for start, end in blocks:
            exact = "".join(lines[start - 1 : end])
            label = f"{start}" if start == end else f"{start}-{end}"
            parts.append(f"Lines {label} (exact):\n```text\n{exact}```")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Legacy tools (kept for backward compatibility)
# ---------------------------------------------------------------------------

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
    code_dir = get_code_dir()
    if not code_dir:
        return "Error: code directory not set."

    abs_path, path_error = ensure_within_workspace(file_path, base_dir=code_dir)
    if path_error:
        return path_error

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
    code_dir = get_code_dir()
    if not code_dir:
        return "Error: code directory not set."

    abs_dir, path_error = ensure_within_workspace(directory, base_dir=code_dir)
    if path_error:
        return path_error

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
