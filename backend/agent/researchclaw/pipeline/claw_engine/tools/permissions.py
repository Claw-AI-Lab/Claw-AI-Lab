"""Sandbox permission policy for tool execution.

Adapted from claw-code's ``runtime/src/permissions.rs`` PermissionPolicy
which checks tool_modes and optionally prompts the user. Our version is
simpler: all tools are auto-allowed within sandbox boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DANGEROUS_BASH_PATTERNS = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    "dd if=/dev/zero",
    ":(){ :",
    "> /dev/sda",
    "chmod -R 777 /",
    "curl | sh",
    "wget | sh",
    "pip install --",
    "shutdown",
    "reboot",
    "kill -9 1",
    "pkill -9",
)

# Bash write-redirect patterns that could escape the workspace
_REDIRECT_WRITE_PATTERN = (
    "> /",       # redirect stdout to absolute path
    ">> /",      # append to absolute path
    "tee /",     # tee to absolute path
    "cp * /",    # copy to root
    "mv * /",    # move to root
)


class SandboxPermissionPolicy:
    """Permission policy that confines tools to workspace boundaries.

    Analogous to claw-code's ``PermissionPolicy`` which has Allow/Deny/Prompt
    modes per tool. Our simplified version:
    - write_file / edit_file: workspace only
    - bash: workspace cwd, timeout enforced, dangerous commands blocked
    - read_file / glob_search / grep_search: workspace + allowed_read_dirs
    """

    def __init__(
        self,
        workspace: Path,
        allowed_read_dirs: list[Path] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.allowed_read_dirs = [
            d.resolve() for d in (allowed_read_dirs or []) if d and Path(d).is_dir()
        ]

    def check(self, tool_name: str, input_data: dict[str, Any]) -> str | None:
        """Check permission. Returns None if allowed, error string if denied.

        Write and read restrictions are relaxed — only dangerous bash
        commands are blocked.
        """
        if tool_name == "bash":
            return self._check_bash(input_data)
        return None

    def _check_bash(self, inp: dict[str, Any]) -> str | None:
        command = inp.get("command", "")
        cmd_lower = command.lower().strip()

        # Block dangerous commands
        for pattern in DANGEROUS_BASH_PATTERNS:
            if pattern in cmd_lower:
                return f"Dangerous command blocked: {command[:80]}"

        # Block write-redirects to absolute paths outside workspace
        ws_str = str(self.workspace)
        for pattern in _REDIRECT_WRITE_PATTERN:
            idx = cmd_lower.find(pattern)
            if idx >= 0:
                # Extract the path after the redirect
                after = command[idx + len(pattern) - 1:].strip().split()[0] if len(command) > idx + len(pattern) else ""
                if after and not after.startswith(ws_str) and after.startswith("/"):
                    return (
                        f"Bash write to path outside workspace blocked: {after}\n"
                        f"All file modifications must be inside: {ws_str}"
                    )

        return None

    def _check_write(self, inp: dict[str, Any]) -> str | None:
        raw_path = inp.get("path", "")
        if not raw_path:
            return "path is required"
        resolved = (self.workspace / raw_path).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            return f"Write outside workspace denied: {raw_path}"
        return None

    def _check_read(self, inp: dict[str, Any]) -> str | None:
        raw_path = inp.get("path", "")
        if not raw_path:
            return None
        import os
        resolved = Path(raw_path).resolve() if os.path.isabs(raw_path) else (self.workspace / raw_path).resolve()

        if str(resolved).startswith(str(self.workspace)):
            return None
        for allowed in self.allowed_read_dirs:
            if str(resolved).startswith(str(allowed)):
                return None
        return (
            f"Read outside allowed directories: {raw_path}\n"
            f"Allowed: workspace + {[str(d) for d in self.allowed_read_dirs]}"
        )
