"""Tool executor — dispatch + 6 implementations.

Ported from claw-code ``rust/crates/tools/src/lib.rs`` ``execute_tool()``
and the runtime helpers in ``bash.rs`` / ``file_ops.rs``.

All file operations are sandboxed: paths are resolved relative to the
workspace root and validated by the permission policy before execution.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 16000
MAX_BASH_RESULT_CHARS = 24000


class ToolExecutor:
    """Execute tools within a workspace.

    Analogous to claw-code's ``CliToolExecutor`` which checks an
    allowlist then calls ``tools::execute_tool(name, &input)``.
    """

    def __init__(
        self,
        workspace: Path,
        allowed_read_dirs: list[Path] | None = None,
        bash_timeout: int = 60,
        python_path: str = "",
    ) -> None:
        self.workspace = workspace.resolve()
        self.allowed_read_dirs = [
            d.resolve() for d in (allowed_read_dirs or []) if d and Path(d).is_dir()
        ]
        self.bash_timeout = bash_timeout
        self.python_path = python_path
        self.call_count = 0
        self._snapshot_count = 0

    def execute(self, name: str, input_data: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool call. Returns (result_string, is_error)."""
        self.call_count += 1
        try:
            if name == "bash":
                return self._bash(input_data), False
            elif name == "read_file":
                return self._read_file(input_data), False
            elif name == "write_file":
                result = self._write_file(input_data)
                self._save_snapshot(input_data.get("path", ""))
                return result, False
            elif name == "edit_file":
                result = self._edit_file(input_data)
                self._save_snapshot(input_data.get("path", ""))
                return result, False
            elif name == "glob_search":
                return self._glob_search(input_data), False
            elif name == "grep_search":
                return self._grep_search(input_data), False
            else:
                return f"Unknown tool: {name}", True
        except PermissionError as e:
            return f"Permission denied: {e}", True
        except FileNotFoundError as e:
            return f"File not found: {e}", True
        except Exception as e:
            return f"Tool error ({type(e).__name__}): {e}", True

    # ------------------------------------------------------------------
    # bash — ported from claw-code runtime/src/bash.rs
    # ------------------------------------------------------------------

    def _bash(self, inp: dict[str, Any]) -> str:
        command = inp.get("command", "")
        if not command:
            raise ValueError("command is required")
        timeout = min(inp.get("timeout", self.bash_timeout), 300)

        _dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"]
        cmd_lower = command.lower()
        if any(d in cmd_lower for d in _dangerous):
            raise PermissionError(f"Dangerous command blocked: {command[:80]}")

        env = os.environ.copy()
        env["WORKSPACE"] = str(self.workspace)

        # Use python_path from config (experiment.sandbox.python_path) to
        # ensure bash runs in the correct conda/venv environment.
        if self.python_path and os.path.isfile(self.python_path):
            python_bin_dir = os.path.dirname(os.path.realpath(self.python_path))
            env["PATH"] = python_bin_dir + ":" + env.get("PATH", "")
            env_prefix = os.path.dirname(python_bin_dir)
            env["CONDA_PREFIX"] = env_prefix
            env["VIRTUAL_ENV"] = env_prefix

        try:
            # Use bash -c (not -lc) to avoid login shell resetting PATH
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"[exit_code: {result.returncode}]")
            output = "\n".join(output_parts) or "(no output)"
        except subprocess.TimeoutExpired:
            output = f"Command timed out after {timeout}s: {command[:100]}"

        return self._truncate_bash(output)

    # ------------------------------------------------------------------
    # read_file — ported from claw-code runtime/src/file_ops.rs
    # ------------------------------------------------------------------

    def _read_file(self, inp: dict[str, Any]) -> str:
        path = self._resolve_read_path(inp["path"])
        offset = inp.get("offset", 0)
        limit = inp.get("limit")

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        end = min(offset + limit, total) if limit else total
        selected = lines[offset:end]

        numbered = "\n".join(
            f"{offset + i + 1:6d} | {line}" for i, line in enumerate(selected)
        )
        header = f"File: {path.name} ({total} lines total, showing {offset+1}-{end})"
        return self._truncate(f"{header}\n{numbered}")

    # ------------------------------------------------------------------
    # write_file — ported from claw-code runtime/src/file_ops.rs
    # ------------------------------------------------------------------

    def _write_file(self, inp: dict[str, Any]) -> str:
        path = self._resolve_write_path(inp["path"])
        content = inp.get("content", "")
        existed = path.exists()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        kind = "updated" if existed else "created"
        line_count = len(content.splitlines())
        try:
            display = str(path.relative_to(self.workspace))
        except ValueError:
            display = str(path)
        return f"File {kind}: {display} ({line_count} lines)"

    # ------------------------------------------------------------------
    # edit_file — ported from claw-code runtime/src/file_ops.rs
    # ------------------------------------------------------------------

    def _edit_file(self, inp: dict[str, Any]) -> str:
        path = self._resolve_write_path(inp["path"])
        old_string = inp["old_string"]
        new_string = inp["new_string"]
        replace_all = inp.get("replace_all", False)

        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if old_string == new_string:
            return "old_string and new_string are identical — no change needed"

        text = path.read_text(encoding="utf-8")
        if old_string not in text:
            snippet = old_string[:100].replace("\n", "\\n")
            return f"old_string not found in {path.name}: '{snippet}...'"

        if replace_all:
            count = text.count(old_string)
            new_text = text.replace(old_string, new_string)
        else:
            count = 1
            new_text = text.replace(old_string, new_string, 1)

        path.write_text(new_text, encoding="utf-8")
        try:
            display = str(path.relative_to(self.workspace))
        except ValueError:
            display = str(path)
        return f"Edited {display}: {count} replacement(s)"

    # ------------------------------------------------------------------
    # glob_search — ported from claw-code runtime/src/file_ops.rs
    # ------------------------------------------------------------------

    def _glob_search(self, inp: dict[str, Any]) -> str:
        import time as _time

        pattern = inp["pattern"]
        base_str = inp.get("path")
        base = self._resolve_read_path(base_str) if base_str else self.workspace

        if not base.is_dir():
            return f"Not a directory: {base}"

        is_recursive = "**" in pattern
        matches: list[tuple[float, Path]] = []
        cap = 200
        scan_limit = cap * 5
        deadline = _time.monotonic() + 8.0
        timed_out = False
        scanned = 0

        for p in base.glob(pattern):
            scanned += 1
            if _time.monotonic() > deadline or scanned > 50000:
                timed_out = True
                break
            if p.is_symlink() and is_recursive:
                continue
            if p.is_file() and not any(
                part.startswith(".") or part == "__pycache__" for part in p.parts
            ):
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    mtime = 0
                matches.append((mtime, p))
                if len(matches) >= scan_limit:
                    break

        matches.sort(key=lambda x: -x[0])
        truncated = len(matches) > cap
        matches = matches[:cap]

        lines: list[str] = []
        for _, p in matches:
            try:
                lines.append(str(p.relative_to(base)))
            except ValueError:
                lines.append(str(p))

        header = f"Found {len(lines)} file(s)"
        if truncated:
            header += f" (showing first {cap})"
        if timed_out:
            header += " [TIMEOUT: directory too large — use a more specific pattern or path]"
        return header + "\n" + "\n".join(lines) if lines else "No files matched."

    # ------------------------------------------------------------------
    # grep_search — ported from claw-code runtime/src/file_ops.rs
    # ------------------------------------------------------------------

    def _grep_search(self, inp: dict[str, Any]) -> str:
        import time as _time

        pattern_str = inp["pattern"]
        base_str = inp.get("path")
        file_glob = inp.get("glob")
        context_lines = inp.get("context", 2)

        base = self._resolve_read_path(base_str) if base_str else self.workspace

        try:
            regex = re.compile(pattern_str, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"

        if base.is_file():
            files = [base]
        else:
            deadline = _time.monotonic() + 8.0
            collected: list[Path] = []
            for f in base.rglob("*"):
                if _time.monotonic() > deadline or len(collected) > 5000:
                    break
                if f.is_symlink():
                    continue
                if f.is_file() and not any(
                    p.startswith(".") or p == "__pycache__"
                    for p in f.relative_to(base).parts
                ):
                    collected.append(f)
            files = sorted(collected)
            if file_glob:
                files = [f for f in files if fnmatch.fnmatch(f.name, file_glob)]

        results: list[str] = []
        match_count = 0
        max_matches = 200

        for fpath in files:
            if match_count >= max_matches:
                break
            try:
                if fpath.stat().st_size > 2 * 1024 * 1024:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            file_matches: list[str] = []
            for i, line in enumerate(lines):
                if regex.search(line):
                    match_count += 1
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    for j in range(start, end):
                        prefix = ":" if j == i else "-"
                        file_matches.append(f"  {j+1}{prefix} {lines[j]}")
                    if end < len(lines):
                        file_matches.append("  ---")
            if file_matches:
                try:
                    rel = fpath.relative_to(base)
                except ValueError:
                    rel = fpath
                results.append(f"{rel}:\n" + "\n".join(file_matches))

        if not results:
            return f"No matches for /{pattern_str}/"

        header = f"{match_count} match(es) in {len(results)} file(s)"
        if match_count >= max_matches:
            header += " (truncated)"
        return self._truncate(header + "\n\n" + "\n\n".join(results))

    # ------------------------------------------------------------------
    # Path validation helpers
    # ------------------------------------------------------------------

    def _resolve_write_path(self, raw: str) -> Path:
        """Resolve a path for writing.

        Absolute paths are accepted as-is (no workspace restriction).
        Relative paths are resolved against the workspace.
        """
        if os.path.isabs(raw):
            return Path(raw).resolve()
        return (self.workspace / raw).resolve()

    def _resolve_read_path(self, raw: str) -> Path:
        """Resolve a path for reading. No directory restriction."""
        p = Path(raw).resolve() if os.path.isabs(raw) else (self.workspace / raw).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{raw} not found")
        return p

    def _save_snapshot(self, raw_path: str) -> None:
        """Save a versioned snapshot of .py files after each write/edit.

        Snapshots go to workspace/.snapshots/ for post-hoc debugging —
        you can diff consecutive versions to see exactly what each edit changed.
        """
        if not raw_path or not raw_path.endswith(".py"):
            return
        try:
            path = self._resolve_write_path(raw_path)
            if not path.exists():
                return
            self._snapshot_count += 1
            snap_dir = self.workspace / ".snapshots"
            snap_dir.mkdir(exist_ok=True)
            fname = path.stem
            snap_path = snap_dir / f"{fname}_v{self._snapshot_count:03d}.py"
            snap_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _truncate(text: str, max_chars: int = MAX_RESULT_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"

    @staticmethod
    def _truncate_bash(text: str, max_chars: int = MAX_BASH_RESULT_CHARS) -> str:
        """Truncate bash output keeping both head and tail.

        Errors and tracebacks are always at the end of output, so we
        keep the first 30% and last 70% to ensure the LLM sees both
        the initial output and the final error/traceback.
        """
        if len(text) <= max_chars:
            return text
        head_budget = max_chars * 3 // 10
        tail_budget = max_chars - head_budget - 200
        head = text[:head_budget]
        tail = text[-tail_budget:]
        return (
            f"{head}\n\n"
            f"... [{len(text)} total chars, middle truncated] ...\n\n"
            f"{tail}"
        )
