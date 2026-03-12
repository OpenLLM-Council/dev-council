"""Shell command execution and process management tools for the coder agent."""
import subprocess
import platform
import signal
import os
from langchain.tools import tool


# ---------------------------------------------------------------------------
# Background process registry
# ---------------------------------------------------------------------------

_processes: dict[str, subprocess.Popen] = {}


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
    from app.agents.coder_agent import _code_dir

    work_dir = cwd.strip() if cwd.strip() else _code_dir
    if not work_dir:
        work_dir = os.getcwd()

    # Ensure the directory exists
    os.makedirs(work_dir, exist_ok=True)

    # Force non-interactive mode: close stdin so prompts get EOF,
    # and set CI=true which many tools (npm, npx, yarn) check.
    env = {**os.environ, "CI": "true"}

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            stdin=subprocess.DEVNULL,
            env=env,
        )
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
    from app.agents.coder_agent import _code_dir

    work_dir = cwd.strip() if cwd.strip() else _code_dir
    if not work_dir:
        work_dir = os.getcwd()
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
        kwargs = dict(
            shell=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        # On Windows, create a new process group so we can terminate cleanly
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(command, **kwargs)
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
        shell = "PowerShell / cmd.exe"
    elif system == "Darwin":
        shell = "zsh (default on macOS)"
    else:
        shell = "bash"
    return (
        f"OS: {system}\n"
        f"Release: {release}\n"
        f"Version: {version}\n"
        f"Architecture: {arch}\n"
        f"Shell: {shell}"
    )
