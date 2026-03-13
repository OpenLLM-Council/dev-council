"""Shell command execution and process management tools for the coder agent."""
import subprocess
import platform
import signal
import os
import re
from langchain.tools import tool

from app.tools.path_utils import ensure_within_workspace, get_code_dir


# ---------------------------------------------------------------------------
# Background process registry
# ---------------------------------------------------------------------------

_processes: dict[str, subprocess.Popen] = {}
_CODE_WRITE_EXTENSIONS = {
    ".env", ".css", ".csv", ".html", ".ini", ".js", ".json", ".jsx",
    ".md", ".mjs", ".py", ".sql", ".svg", ".toml", ".ts", ".tsx",
    ".txt", ".xml", ".yaml", ".yml",
}


def _resolve_work_dir(cwd: str) -> tuple[str, str | None]:
    """Resolve cwd relative to the active code directory when available."""
    code_dir = get_code_dir()
    target = cwd.strip() if cwd.strip() else (code_dir or os.getcwd())
    return ensure_within_workspace(target, base_dir=code_dir or None)


def _validate_shell_command(command: str) -> str | None:
    """Reject shell commands that are likely to cause tool loops or shell portability issues."""
    normalized = command.lower()

    if os.name == "nt":
        issues = []
        if "mkdir -p" in normalized:
            issues.append("`mkdir -p` is bash syntax and creates the wrong folders on Windows cmd.")
        if re.search(r"\bpwd\b", normalized):
            issues.append("`pwd` is not available in Windows cmd; use the working directory already passed via `cwd` or use `cd`.")
        if re.search(r"mkdir\s+[^\n]*\{[^\n]*,[^\n]*\}", normalized):
            issues.append("brace expansion like `src/{a,b}` is not supported in Windows cmd.")
        if issues:
            joined = " ".join(issues)
            return (
                f"ERROR: Unsupported Windows shell syntax detected. {joined} "
                "run_shell executes commands through Windows cmd.exe; use Windows-safe commands."
            )

    write_ops = (">", ">>", "add-content", "set-content", "out-file", "type nul")
    if any(op in normalized for op in write_ops):
        matches = re.findall(r"([^\s\"']+\.[A-Za-z0-9]+)", command)
        if any(os.path.splitext(path)[1].lower() in _CODE_WRITE_EXTENSIONS for path in matches):
            return (
                "ERROR: Do not use run_shell to create or edit project files. "
                "Use write_file for new files, insert_after_line for additions, "
                "and replace_in_file/delete_lines for edits."
            )

    return None


def _run_command(command: str, work_dir: str, env: dict, timeout: int) -> subprocess.CompletedProcess:
    """Execute a shell command using a deterministic shell for the current platform."""
    if os.name == "nt":
        return subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", command],
            cwd=work_dir,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=env,
        )

    return subprocess.run(
        command,
        shell=True,
        cwd=work_dir,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def _start_command(command: str, work_dir: str, env: dict) -> subprocess.Popen:
    """Start a background command using a deterministic shell for the current platform."""
    kwargs = dict(
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if os.name == "nt":
        return subprocess.Popen(
            ["cmd.exe", "/d", "/s", "/c", command],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            **kwargs,
        )

    return subprocess.Popen(command, shell=True, **kwargs)


@tool
def run_shell(command: str, cwd: str = "") -> str:
    """Run a shell command in the specified directory.

    Use this for installing dependencies, scaffolding projects, running build
    tools, or any CLI operation (e.g. npm, npx, pip, cargo, dotnet, etc.).

    Args:
        command: The shell command to execute (e.g. "npm install", "npx create-react-app my-app").
        cwd: The working directory to run the command in. Defaults to the project code directory.

    Returns:
        Combined stdout + stderr output of the command, or an error message.
    """
    work_dir, path_error = _resolve_work_dir(cwd)
    if path_error:
        return path_error

    validation_error = _validate_shell_command(command)
    if validation_error:
        return validation_error

    # Ensure the directory exists
    os.makedirs(work_dir, exist_ok=True)

    # Force non-interactive mode: close stdin so prompts get EOF,
    # and set CI=true which many tools (npm, npx, yarn) check.
    env = {**os.environ, "CI": "true"}

    try:
        result = _run_command(command, work_dir, env, timeout=120)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 120 seconds."
    except Exception as e:
        return f"ERROR: {e}"


@tool
def start_background_process(command: str, process_name: str, cwd: str = "") -> str:
    """Start a long-running background process (e.g. a dev server).

    The process runs in the background and can be stopped later with
    stop_background_process.  Use this for servers, watchers, or any
    command that doesn't exit on its own.

    Args:
        command: The shell command to run (e.g. "node server.js", "npm run dev").
        process_name: A short label to identify this process later (e.g. "backend-server").
        cwd: Working directory. Defaults to the project code directory.

    Returns:
        A confirmation string with the process name and PID.
    """
    work_dir, path_error = _resolve_work_dir(cwd)
    if path_error:
        return path_error

    validation_error = _validate_shell_command(command)
    if validation_error:
        return validation_error
    os.makedirs(work_dir, exist_ok=True)

    # Stop an existing process with the same name first
    if process_name in _processes:
        old = _processes[process_name]
        if old.poll() is None:
            old.terminate()
            try:
                old.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old.kill()
        del _processes[process_name]

    env = {**os.environ, "CI": "true"}

    try:
        proc = _start_command(command, work_dir, env)
        _processes[process_name] = proc
        return f"Started '{process_name}' (PID {proc.pid}): {command}"
    except Exception as e:
        return f"ERROR starting process: {e}"


@tool
def stop_background_process(process_name: str) -> str:
    """Stop a background process that was started with start_background_process.

    Sends a graceful termination signal first (CTRL+C / SIGINT), then forces
    termination if the process doesn't exit within 5 seconds.

    Args:
        process_name: The label given when the process was started.

    Returns:
        A status message indicating whether the process was stopped.
    """
    if process_name not in _processes:
        alive = [n for n, p in _processes.items() if p.poll() is None]
        return f"No process named '{process_name}'. Running processes: {alive or 'none'}"

    proc = _processes.pop(process_name)

    if proc.poll() is not None:
        return f"'{process_name}' already exited (code {proc.returncode})."

    # Try graceful shutdown first
    try:
        if os.name == "nt":
            # On Windows, send CTRL_BREAK_EVENT to the process group
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)

        try:
            proc.wait(timeout=5)
            return f"Stopped '{process_name}' gracefully (code {proc.returncode})."
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass

    # Force kill if graceful shutdown failed
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return f"Force-killed '{process_name}' (PID {proc.pid})."


@tool
def get_platform_info() -> str:
    """Detect the current operating system, version, architecture, and shell.

    Call this FIRST before running shell commands so you can use
    the correct syntax for the platform (e.g. PowerShell vs bash).

    Returns:
        A string describing the OS, version, architecture, and default shell.
    """
    system = platform.system()
    release = platform.release()
    version = platform.version()
    arch = platform.machine()
    if system == "Windows":
        shell = "run_shell uses Windows cmd.exe"
    elif system == "Darwin":
        shell = "run_shell uses /bin/sh"
    else:
        shell = "run_shell uses /bin/sh"
    return (
        f"OS: {system}\n"
        f"Release: {release}\n"
        f"Version: {version}\n"
        f"Architecture: {arch}\n"
        f"Shell: {shell}"
    )
