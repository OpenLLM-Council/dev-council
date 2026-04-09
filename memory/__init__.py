"""Memory package for dev-council.

Storage layout:
  user scope    : ~/.dev-council/memory/<slug>.md
  project scope : .dev-council/memory/<slug>.md
"""
from .store import (  # noqa: F401
    MemoryEntry,
    save_memory,
    delete_memory,
    load_index,
    load_entries,
    search_memory,
    get_index_content,
    parse_frontmatter,
    USER_MEMORY_DIR,
    INDEX_FILENAME,
    MAX_INDEX_LINES,
    MAX_INDEX_BYTES,
)
from .scan import (  # noqa: F401
    MemoryHeader,
    scan_memory_dir,
    scan_all_memories,
    format_memory_manifest,
    memory_age_days,
    memory_age_str,
    memory_freshness_text,
)
from .context import (  # noqa: F401
    get_memory_context,
    find_relevant_memories,
    truncate_index_content,
)
from .types import (  # noqa: F401
    MEMORY_TYPES,
    MEMORY_TYPE_DESCRIPTIONS,
    MEMORY_SYSTEM_PROMPT,
    WHAT_NOT_TO_SAVE,
)
from .consolidator import consolidate_session  # noqa: F401

__all__ = [
    "MemoryEntry",
    "save_memory",
    "delete_memory",
    "load_index",
    "load_entries",
    "search_memory",
    "get_index_content",
    "parse_frontmatter",
    "USER_MEMORY_DIR",
    "INDEX_FILENAME",
    "MAX_INDEX_LINES",
    "MAX_INDEX_BYTES",
    "MemoryHeader",
    "scan_memory_dir",
    "scan_all_memories",
    "format_memory_manifest",
    "memory_age_days",
    "memory_age_str",
    "memory_freshness_text",
    "get_memory_context",
    "find_relevant_memories",
    "truncate_index_content",
    "MEMORY_TYPES",
    "MEMORY_TYPE_DESCRIPTIONS",
    "MEMORY_SYSTEM_PROMPT",
    "WHAT_NOT_TO_SAVE",
    "consolidate_session",
]
