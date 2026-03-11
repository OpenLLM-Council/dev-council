import os


MEMORY_DIR = ".memory"
TREE_FILE = "tree.md"
PROGRESS_FILE = "progress.md"
BASE_PATH_FILE = "base_path.md"


def _memory_path(project_path: str, filename: str) -> str:
    return os.path.join(project_path, MEMORY_DIR, filename)


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


class CoderMemory:
    """
    Manages persistent coder agent memory stored as markdown files inside
    a .memory/ folder within the project output directory.

    Files maintained:
      - base_path.md   : absolute base path of the project
      - tree.md        : current file tree of the code/ directory
      - progress.md    : milestone completion log
    """

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.memory_dir = os.path.join(project_path, MEMORY_DIR)


    def initialize(self):
        """
        Creates the .memory/ folder and seeds initial files.
        If files already exist (resumed session), existing content is kept.
        """
        os.makedirs(self.memory_dir, exist_ok=True)

        base_path_file = _memory_path(self.project_path, BASE_PATH_FILE)
        if not os.path.exists(base_path_file):
            abs_path = os.path.abspath(self.project_path)
            _write(base_path_file, f"# Base Path\n\n{abs_path}\n")

        progress_file = _memory_path(self.project_path, PROGRESS_FILE)
        if not os.path.exists(progress_file):
            _write(progress_file, "# Milestone Progress\n\n_No milestones completed yet._\n")
        self.update_tree()


    def update_tree(self):
        """
        Regenerates tree.md to reflect the current state of the code/ directory.
        Called after every file write so the agent always has the latest snapshot.
        """
        code_dir = os.path.join(self.project_path, "code")
        tree_str = self._build_tree(code_dir)
        content = f"# File Tree\n\n```\n{tree_str}\n```\n"
        _write(_memory_path(self.project_path, TREE_FILE), content)

    def _build_tree(self, start_path: str) -> str:
        """Builds a readable tree string for start_path."""
        if not os.path.exists(start_path):
            return "(no files yet)"

        lines = []
        skip = {".git", "__pycache__", "node_modules", "venv", ".venv", ".memory"}

        for root, dirs, files in os.walk(start_path):
            dirs[:] = [d for d in dirs if d not in skip]

            level = root.replace(start_path, "").count(os.sep)
            indent = "    " * level
            lines.append(f"{indent}{os.path.basename(root)}/")

            sub_indent = "    " * (level + 1)
            for f in sorted(files):
                lines.append(f"{sub_indent}{f}")

        return "\n".join(lines)


    def update_progress(self, milestone_name: str, status: str = "✅ Complete"):
        """
        Appends or updates the progress.md entry for a given milestone.

        Args:
            milestone_name: Short name/title of the milestone.
            status: Status string, e.g. '✅ Complete' or '🔄 In Progress'.
        """
        progress_file = _memory_path(self.project_path, PROGRESS_FILE)
        existing = _read(progress_file)

        entry = f"- **{milestone_name}**: {status}"

        if milestone_name in existing:
            lines = existing.splitlines()
            updated_lines = []
            for line in lines:
                if f"**{milestone_name}**" in line:
                    updated_lines.append(entry)
                else:
                    updated_lines.append(line)
            new_content = "\n".join(updated_lines) + "\n"
        else:
            cleaned = existing.replace("_No milestones completed yet._", "").strip()
            new_content = cleaned + f"\n{entry}\n" if cleaned else f"# Milestone Progress\n\n{entry}\n"

        _write(progress_file, new_content)


    def read_context(self) -> str:
        """
        Returns a formatted string of all memory files to be injected
        into the coder agent prompt as '## CURRENT PROGRESS MEMORY'.
        """
        base_path = _read(_memory_path(self.project_path, BASE_PATH_FILE))
        tree = _read(_memory_path(self.project_path, TREE_FILE))
        progress = _read(_memory_path(self.project_path, PROGRESS_FILE))

        if not any([base_path.strip(), tree.strip(), progress.strip()]):
            return ""

        sections = []
        if base_path.strip():
            sections.append(f"### Base Path\n{base_path.strip()}")
        if tree.strip():
            sections.append(f"### Current File Tree\n{tree.strip()}")
        if progress.strip():
            sections.append(f"### Milestone Progress\n{progress.strip()}")

        return "\n\n".join(sections)
