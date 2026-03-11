from langchain_ollama import ChatOllama
from langchain.tools import tool
from app.core.config import settings
from app.tools.file_tree import show_tree
from app.tools.file_reader import read_file, read_directory
from app.tools.file_writer import apply_search_replace_blocks
import os

_code_dir: str = ""


def set_code_dir(path: str):
    """Called by manager.py to set the output code directory for the write_file tool."""
    global _code_dir
    _code_dir = path


@tool
def read_memory(memory_dir: str) -> str:
    """Read all .memory/ files to check current project progress.

    CALL THIS FIRST before writing any code.

    Args:
        memory_dir (str): Absolute path to the .memory/ folder

    Returns:
        str: Contents of all memory files.
    """
    if not os.path.exists(memory_dir):
        return "No memory directory found. This is a fresh project."

    output = []
    for filename in sorted(os.listdir(memory_dir)):
        if not filename.endswith(".md"):
            continue
        file_path = os.path.join(memory_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            output.append(f"### {filename}\n{content}")
        except Exception as e:
            output.append(f"### {filename}\nError reading: {e}")

    return "\n\n".join(output) if output else "Memory folder exists but is empty."


@tool
def write_file(file_path: str, content: str | dict | list) -> str:
    """Write or update a single file in the project code directory.

    For NEW files: pass the full file contents.
    For EXISTING files: pass SEARCH/REPLACE blocks using <<<<, ====, >>>> markers.

    Args:
        file_path (str): Relative file path (e.g. 'src/index.ts', 'package.json').
        content: Full file content OR SEARCH/REPLACE blocks for existing files.

    Returns:
        str: Confirmation message.
    """
    import json as _json

    if not _code_dir:
        return "Error: code directory not set. Cannot write file."

    if isinstance(content, (dict, list)):
        content = _json.dumps(content, indent=2)

    # LLMs sometimes pass paths like "code/backend/..." because the tree showed "code/" as the root
    if file_path.startswith("code/"):
        file_path = file_path[5:]
    elif file_path.startswith("code\\"):
        file_path = file_path[5:]
        
    abs_path = os.path.join(_code_dir, file_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # Try SEARCH/REPLACE first if file exists
    if os.path.exists(abs_path):
        patched = apply_search_replace_blocks(abs_path, content)
        if patched:
            return f"✓ Patched: {file_path}"

    # Full file write (new file or no SEARCH/REPLACE blocks found)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"✓ Written: {file_path}"


CODER_SYSTEM_PROMPT = """You are a senior software developer implementing a project milestone.

## MANDATORY WORKFLOW
1. Call `read_memory` with the .memory/ directory path to check project state.
2. Call `show_tree` on the code directory to see existing files.
3. If modifying an existing file, call `read_file` first.
4. Use `write_file` to create/update each file ONE BY ONE.

## WRITE_FILE USAGE
- For NEW files: pass full contents to `write_file(file_path, content)`
- For EXISTING files: pass SEARCH/REPLACE blocks:
  <<<<
  old code exactly as it appears
  ====
  new replacement code
  >>>>

## RULES
- Always use scalable folder architecture and structure.
- Use ONLY the approved tech stack.
- Write complete, production-ready code.
- Create each file individually using `write_file` — do NOT dump all code at once.
- After writing all files, respond with a brief summary of what you created.
"""

CODER_REVISION_PROMPT = """You are a senior software developer fixing code based on reviewer feedback.

## MANDATORY WORKFLOW
1. Call `read_memory` to check project state.
2. Call `read_file` on each file you need to fix.
3. Use `write_file` to update each file ONE BY ONE.

## WRITE_FILE USAGE
- For EXISTING files use SEARCH/REPLACE blocks in `write_file`:
  <<<<
  old code
  ====
  new code
  >>>>
- For NEW files pass full contents.

## RULES
- Address ALL reviewer feedback.
- Create/update each file individually using `write_file`.
- After writing all files, respond with a brief summary of what you fixed.
"""

CODER_TOOLS = [read_memory, show_tree, read_file, read_directory, write_file]


def get_coder_agent(model_name: str, revision: bool = False):
    """
    Factory: creates a tool-calling coder agent.

    Tools:
      - read_memory  : reads .memory/ for project progress (MUST call first)
      - show_tree    : inspect directory file structure
      - read_file    : read a single file before editing
      - read_directory : read all files in a directory
      - write_file   : write/update a single file in the code directory
    """
    from langchain.agents import create_agent

    llm = ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_URL,
        num_predict=8192,
    )

    system_prompt = CODER_REVISION_PROMPT if revision else CODER_SYSTEM_PROMPT

    agent = create_agent(
        model=llm,
        tools=CODER_TOOLS,
        system_prompt=system_prompt,
    )

    return agent
