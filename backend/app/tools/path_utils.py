import os


def get_code_dir() -> str:
    """Return the active code directory configured by the coder agent."""
    try:
        from app.agents.coder_agent import _code_dir
    except Exception:
        return ""
    return os.path.abspath(_code_dir) if _code_dir else ""


def resolve_path(path: str, base_dir: str | None = None) -> str:
    """Resolve a path relative to the active code directory when available."""
    base = os.path.abspath(base_dir) if base_dir else get_code_dir()
    if os.path.isabs(path):
        return os.path.abspath(path)
    if base:
        return os.path.abspath(os.path.join(base, path))
    return os.path.abspath(path)


def ensure_within_workspace(path: str, base_dir: str | None = None) -> tuple[str, str | None]:
    """
    Resolve a path and ensure it stays inside the active workspace when one exists.

    Returns the normalized absolute path plus an optional error string.
    """
    resolved = resolve_path(path, base_dir=base_dir)
    workspace = os.path.abspath(base_dir) if base_dir else get_code_dir()

    if workspace:
        try:
            common = os.path.commonpath([resolved, workspace])
        except ValueError:
            common = ""
        if common != workspace:
            return resolved, (
                f"Error: '{path}' resolves outside the active code directory "
                f"('{workspace}')."
            )

    return resolved, None
