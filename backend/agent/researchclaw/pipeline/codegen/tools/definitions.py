"""Tool specifications for the claw-code agentic turn loop.

Ported from claw-code ``rust/crates/tools/src/lib.rs`` ``mvp_tool_specs()``.
Each tool has a name, description, and JSON-Schema ``input_schema`` that the
LLM sees via the API ``tools`` field — NOT embedded in the system prompt.
"""

from __future__ import annotations

from typing import Any


def tool_spec(name: str, description: str, properties: dict[str, Any],
              required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOL_SPECS: list[dict[str, Any]] = [
    tool_spec(
        name="bash",
        description=(
            "Execute a shell command in the experiment workspace. "
            "Use for running Python scripts, installing packages, checking output, etc. "
            "Commands run in bash -lc with a timeout. Prefer short, targeted commands."
        ),
        properties={
            "command": {"type": "string", "description": "The shell command to execute."},
            "timeout": {
                "type": "integer", "minimum": 1, "default": 60,
                "description": "Timeout in seconds (default 60).",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what this command does (5-10 words).",
            },
        },
        required=["command"],
    ),

    tool_spec(
        name="read_file",
        description=(
            "Read a text file from the workspace or allowed data directories. "
            "Returns numbered lines. Use offset/limit for large files."
        ),
        properties={
            "path": {"type": "string", "description": "Path to the file to read."},
            "offset": {
                "type": "integer", "minimum": 0,
                "description": "Line offset to start reading from (0-based).",
            },
            "limit": {
                "type": "integer", "minimum": 1,
                "description": "Max number of lines to return.",
            },
        },
        required=["path"],
    ),

    tool_spec(
        name="write_file",
        description=(
            "Write a text file in the workspace. Creates parent directories if needed. "
            "Use for creating new experiment files (main.py, helpers, configs)."
        ),
        properties={
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Full content to write to the file."},
        },
        required=["path", "content"],
    ),

    tool_spec(
        name="edit_file",
        description=(
            "Replace text in an existing workspace file. Finds old_string and replaces "
            "with new_string. Use for targeted fixes instead of rewriting entire files."
        ),
        properties={
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact text to find and replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean", "default": False,
                "description": "Replace all occurrences (default: first only).",
            },
        },
        required=["path", "old_string", "new_string"],
    ),

    tool_spec(
        name="glob_search",
        description=(
            "Find files by glob pattern in the workspace or data directories. "
            "Returns matching file paths sorted by modification time (newest first). "
            "Capped at 100 results."
        ),
        properties={
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py', '*.yaml')."},
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: workspace root).",
            },
        },
        required=["pattern"],
    ),

    tool_spec(
        name="grep_search",
        description=(
            "Search file contents with a regex pattern. Returns matching lines with "
            "file paths and line numbers. Use for finding function definitions, imports, "
            "variable usages in codebases."
        ),
        properties={
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {
                "type": "string",
                "description": "File or directory to search in (default: workspace root).",
            },
            "glob": {
                "type": "string",
                "description": "File glob filter (e.g. '*.py').",
            },
            "context": {
                "type": "integer", "minimum": 0, "default": 2,
                "description": "Number of context lines before and after each match.",
            },
        },
        required=["pattern"],
    ),
]

TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in TOOL_SPECS)
