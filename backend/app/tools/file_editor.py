"""
Production-grade file editing tools for targeted patching and precise modifications.
Adapted from code-agent's strategy for chunked reading and surgical edits.

Core philosophy:
- NEVER rewrite entire files for small changes
- Use line-based deletion and text-based replacement
- Support atomic multi-operation changes
- Provide helpful error messages with candidate suggestions
"""
import os
import json
import difflib
from typing import List, Any
from langchain.tools import tool

from app.tools.path_utils import ensure_within_workspace


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
            "Use search_in_file to find files, then pass a specific file path."
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
# DELETE tool — removes line ranges
# ---------------------------------------------------------------------------

@tool
def delete_lines(filepath: str, line_ranges: List[dict]) -> str:
    """
    Deletes specific line ranges from a file BY LINE NUMBER -- no text matching needed.
    Use for removing functions, call sites, imports, or any unwanted block.

    Pass ALL ranges (definition + every call site) in ONE call for atomic removal.
    Line numbers refer to the ORIGINAL file before any deletions in this call.
    Automatically cleans up double-blank-lines left behind.

    Args:
        filepath:    Absolute or relative path to the file (relative to code_dir).
        line_ranges: List of {"start": N, "end": M} dicts (1-indexed, inclusive).
                     Single line: {"start": N, "end": N}

    Example: delete_lines(filepath, [{"start":4,"end":6},{"start":10,"end":10},{"start":20,"end":20}])
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

        errors, parsed = [], []
        for i, r in enumerate(line_ranges):
            s, e = r.get("start"), r.get("end")
            if s is None or e is None:
                errors.append(f"Range #{i+1}: missing 'start' or 'end'.")
            elif not (1 <= s <= total and 1 <= e <= total):
                errors.append(f"Range #{i+1}: ({s}-{e}) out of bounds (file has {total} lines).")
            elif s > e:
                errors.append(f"Range #{i+1}: start ({s}) > end ({e}).")
            else:
                parsed.append((s - 1, e - 1))

        if errors:
            return "Aborted (file NOT modified):\n" + "\n".join(errors)

        to_delete = set()
        for lo, hi in parsed:
            to_delete.update(range(lo, hi + 1))

        remaining = [line for i, line in enumerate(lines) if i not in to_delete]

        cleaned, prev_blank = [], False
        for line in remaining:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(cleaned)

        return (
            f"Deleted {len(to_delete)} line(s) from '{filepath}'. "
            f"File now has {len(cleaned)} lines."
        )
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# INSERT tool — adds lines at specific positions
# ---------------------------------------------------------------------------

@tool
def insert_after_line(filepath: str, after_line: int, new_code: str) -> str:
    """
    Inserts new_code after a specific line number in an existing file.
    Use this when adding a new function, block, or import at a known position.
    Much simpler than replace_in_file for pure insertions.

    Args:
        filepath:   Absolute or relative path to the file (relative to code_dir).
        after_line: Insert after this 1-indexed line. Use 0 to insert at top of file.
        new_code:   The full code string to insert (include a leading blank line for spacing).
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

        if not (0 <= after_line <= total):
            return f"Error: after_line ({after_line}) out of range (file has {total} lines)."

        if new_code and not new_code.endswith("\n"):
            new_code += "\n"

        new_lines = lines[:after_line] + [new_code] + lines[after_line:]

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        inserted = new_code.count("\n")
        return (
            f"Inserted {inserted} line(s) after line {after_line} in '{filepath}'. "
            f"File now has {total + inserted} lines."
        )
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# REPLACE tool — targeted text replacement
# ---------------------------------------------------------------------------

@tool
def replace_in_file(filepath: str, old_text: str, new_text: Any) -> str:
    """
    Replaces the FIRST occurrence of old_text with new_text in a file.
    old_text MUST match the file exactly -- every space, tab, and newline counts.

    Prefer insert_after_line for adding new code.
    Prefer delete_lines for removing code.
    Use this for in-place modifications: renaming, changing logic, updating a value.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err
    try:
        if isinstance(new_text, (dict, list)):
            new_text = json.dumps(new_text, indent=4)
        new_text = str(new_text)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if old_text not in content:
            hint = _nearby_hint(content, old_text)
            return (
                f"Error: old_text NOT found in '{filepath}'.\n"
                f"Cause: wrong indentation, extra whitespace, or stale read.\n"
                f"{hint}"
                "-> Use get_lines on the specific lines and copy the text exactly."
            )

        new_content = content.replace(old_text, new_text, 1)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        approx_line = new_content[: new_content.find(new_text)].count("\n") + 1
        return f"Replaced in '{filepath}' at approximately line {approx_line}."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# MULTI_REPLACE tool — atomic multi-replacement
# ---------------------------------------------------------------------------

@tool
def multi_replace_in_file(filepath: str, replacements: List[dict]) -> str:
    """
    Applies multiple find-and-replace operations ATOMICALLY to a single file.
    If ANY old_text is not found, the ENTIRE operation is aborted -- file unchanged.

    Use when renaming a symbol that appears in multiple places, or updating
    several related lines at once.

    Args:
        filepath:     Absolute or relative path to the file (relative to code_dir).
        replacements: List of {"old_text": "...", "new_text": "..."} dicts.
    """
    filepath, path_error = _normalize_path(filepath)
    if path_error:
        return path_error
    err = _assert_is_file(filepath)
    if err:
        return err
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        missing = []
        for i, rep in enumerate(replacements):
            old = rep.get("old_text", "")
            new = rep.get("new_text", "")
            if isinstance(new, (dict, list)):
                rep["new_text"] = json.dumps(new, indent=4)
            if old not in content:
                hint = _nearby_hint(content, old)
                missing.append(f"  Replacement #{i+1}: old_text not found.{hint}")

        if missing:
            return (
                "Aborted (file NOT modified):\n"
                + "\n".join(missing)
                + "\n-> Fix old_text values and retry."
            )

        for rep in replacements:
            content = content.replace(str(rep["old_text"]), str(rep["new_text"]), 1)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Applied {len(replacements)} replacement(s) to '{filepath}'."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# CREATE tool — only for brand-new files
# ---------------------------------------------------------------------------

@tool
def write_file(filepath: str, content: Any) -> str:
    """
    Creates a BRAND NEW file and writes content to it.
    Refuses if the file already exists -- use replace_in_file, insert_after_line,
    or delete_lines to edit existing files.
    """
    try:
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=4)
        content = str(content)

        abs_path, path_error = _normalize_path(filepath)
        if path_error:
            return path_error
        if os.path.exists(abs_path):
            return (
                f"Error: '{filepath}' already exists. "
                "Use replace_in_file or insert_after_line to edit it."
            )

        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Created '{filepath}' ({content.count(chr(10)) + 1} lines)."
    except Exception as e:
        return f"Error: {e}"
