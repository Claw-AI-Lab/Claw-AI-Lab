"""Generic agentic turn loop for claw-code style LLM tool use.

Ported from claw-code ``ConversationRuntime::run_turn()``. The loop:
    user_message → (LLM call → tool execution →)* → done

This is the shared engine used by all agentic pipeline stages.
Stage-specific behaviour (verification gates, custom prompts) is
injected via the ``verification_hooks`` parameter.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from researchclaw.pipeline.claw_engine.tools.definitions import TOOL_SPECS
from researchclaw.pipeline.claw_engine.tools.executor import ToolExecutor
from researchclaw.pipeline.claw_engine.tools.permissions import SandboxPermissionPolicy

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 40


@dataclass
class TurnResult:
    """Result of a complete turn loop execution."""
    files: dict[str, str] = field(default_factory=dict)
    iterations: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)
    final_text: str = ""
    elapsed_sec: float = 0.0


class SessionProtocol(Protocol):
    """Minimal interface for session logging."""
    llm_calls: int
    def log(self, phase: str, message: str) -> None: ...
    def log_error(self, phase: str, message: str, exc: Exception | None = None) -> None: ...


VerificationHook = Callable[
    [Path, list[dict[str, Any]], list[str]],
    str | None,
]
"""Callable(workspace, tool_uses_this_iteration, workspace_files) -> inject_message | None.

If a hook returns a non-None string, it is injected as a ``user`` message
into the conversation to steer the LLM. Used for anti-simulation gates,
plan compliance checks, etc.
"""


class _TraceLog:
    """Step-by-step trace logger for debugging agentic loops."""

    def __init__(self, trace_dir: Path, prefix: str = "generation") -> None:
        self._path = trace_dir / f"{prefix}_trace.md"
        self._step = 0
        self._write(f"# {prefix.title()} Trace\n")
        self._write(f"Started: {datetime.now(timezone.utc).isoformat()}\n")

    def iteration_start(self, i: int, total: int) -> None:
        self._step += 1
        self._write(f"\n---\n## Iteration {i}/{total}  (step {self._step})\n")

    def llm_request(self, n_messages: int, n_tools: int, model: str) -> None:
        self._write(
            f"### LLM Request\n"
            f"- Model: `{model}`\n"
            f"- Messages in context: {n_messages}\n"
            f"- Tools available: {n_tools}\n"
        )

    def llm_response(self, text: str, tool_calls: list[dict], tokens: dict | None) -> None:
        self._write("### LLM Response\n")
        if tokens:
            self._write(
                f"- Prompt tokens: {tokens.get('prompt_tokens', '?')}\n"
                f"- Completion tokens: {tokens.get('completion_tokens', '?')}\n"
            )
        if text:
            preview = text[:500] + ("..." if len(text) > 500 else "")
            self._write(f"**Text** ({len(text)} chars):\n```\n{preview}\n```\n")
        if tool_calls:
            self._write(f"**Tool calls**: {len(tool_calls)}\n")
        else:
            self._write("**No tool calls** — generation complete.\n")

    def tool_call(
        self, name: str, input_data: dict, result: str, is_error: bool, elapsed_ms: int,
    ) -> None:
        status = "ERROR" if is_error else "OK"
        self._write(f"\n#### Tool: `{name}` [{status}] ({elapsed_ms}ms)\n")
        if name == "bash":
            cmd = input_data.get("command", "")
            self._write(f"**Command:**\n```bash\n{cmd}\n```\n")
        elif name == "write_file":
            path = input_data.get("path", "?")
            content = input_data.get("content", "")
            n_lines = len(content.splitlines())
            self._write(f"**Path:** `{path}` ({n_lines} lines, {len(content)} chars)\n")
        elif name == "edit_file":
            path = input_data.get("path", "?")
            self._write(f"**Path:** `{path}`\n")
        elif name == "read_file":
            self._write(f"**Path:** `{input_data.get('path', '?')}`\n")
        elif name in ("glob_search", "grep_search"):
            self._write(f"**Pattern:** `{input_data.get('pattern', '?')}`\n")
        result_preview = result[:1000] + ("\n... [truncated]" if len(result) > 1000 else "")
        self._write(f"**Result:**\n```\n{result_preview}\n```\n")

    def permission_denied(self, name: str, reason: str) -> None:
        self._write(f"\n#### Tool: `{name}` [DENIED]\n**Reason:** {reason}\n")

    def iteration_end(self, files_in_workspace: list[str]) -> None:
        if files_in_workspace:
            self._write(
                f"\n**Workspace files:** "
                f"{', '.join(f'`{f}`' for f in files_in_workspace[:30])}\n"
            )

    def loop_end(self, result: TurnResult) -> None:
        self._write(
            f"\n---\n## Summary\n"
            f"- Iterations: {result.iterations}\n"
            f"- Tool calls: {result.tool_calls}\n"
            f"- Files produced: {sorted(result.files.keys())}\n"
            f"- Errors: {len(result.errors)}\n"
            f"- Elapsed: {result.elapsed_sec:.1f}s\n"
        )
        self._write(f"\nCompleted: {datetime.now(timezone.utc).isoformat()}\n")

    def _write(self, text: str) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


class AgentTurnLoop:
    """Generic agentic turn loop: LLM iteratively calls tools.

    Parameters
    ----------
    verification_hooks : list[VerificationHook]
        Called after each tool-execution round.  Each hook receives
        (workspace, tool_uses, workspace_files) and may return a
        ``user`` message to inject or ``None`` to skip.  Hooks are
        one-shot: once they fire they are removed.
    tool_specs : list[dict]
        Override default TOOL_SPECS if the stage needs extra/fewer tools.
    trace_prefix : str
        Prefix for the trace markdown file (e.g. "sanity", "experiment").
    """

    def __init__(
        self,
        *,
        llm_config: Any,
        workspace: Path,
        system_prompt: str,
        session: SessionProtocol,
        allowed_read_dirs: list[Path] | None = None,
        bash_timeout: int = 60,
        max_iterations: int = MAX_ITERATIONS,
        python_path: str = "",
        tool_specs: list[dict[str, Any]] | None = None,
        verification_hooks: list[VerificationHook] | None = None,
        trace_prefix: str = "generation",
    ) -> None:
        self._llm_config = llm_config
        self._workspace = workspace
        self._system_prompt = system_prompt
        self._session = session
        self._max_iterations = max_iterations
        self._messages: list[dict[str, Any]] = []
        self._tool_specs = tool_specs or TOOL_SPECS
        self._verification_hooks = list(verification_hooks or [])

        self._executor = ToolExecutor(
            workspace=workspace,
            allowed_read_dirs=allowed_read_dirs,
            bash_timeout=bash_timeout,
            python_path=python_path,
        )
        self._permissions = SandboxPermissionPolicy(
            workspace=workspace,
            allowed_read_dirs=allowed_read_dirs,
        )

        self._api_tools = self._build_api_tools()
        _coding = getattr(llm_config, "coding_model", "") or ""
        self._use_text_tools = (
            self._is_claude_model(llm_config.primary_model)
            or (bool(_coding) and self._is_claude_model(_coding))
        )
        if self._use_text_tools:
            self._text_tool_prompt = self._build_text_tool_prompt()
            logger.info(
                "[agent_loop] Claude model detected (primary=%s, coding=%s) "
                "— using text-based tool calling",
                llm_config.primary_model, _coding,
            )
        trace_dir = workspace.parent if workspace.parent.is_dir() else workspace
        self._trace = _TraceLog(trace_dir, prefix=trace_prefix)

    @property
    def workspace(self) -> Path:
        return self._workspace

    def run_turn(self, user_message: str) -> TurnResult:
        t0 = time.monotonic()
        self._session.log("EXECUTE", "Turn loop started")
        self._messages.append({"role": "user", "content": user_message})

        result = TurnResult()

        for iteration in range(self._max_iterations):
            iter_num = iteration + 1
            self._trace.iteration_start(iter_num, self._max_iterations)
            self._session.log(
                "EXECUTE", f"Turn {iter_num}/{self._max_iterations}: calling LLM...",
            )

            self._trace.llm_request(
                n_messages=len(self._messages),
                n_tools=len(self._api_tools),
                model=self._llm_config.primary_model,
            )

            response = None
            for _retry in range(3):
                try:
                    response = self._call_llm()
                    break
                except Exception as exc:
                    self._session.log(
                        "EXECUTE",
                        f"LLM call attempt {_retry + 1}/3 failed: {exc}",
                    )
                    if _retry < 2:
                        time.sleep(2 ** _retry)
                    else:
                        error_msg = f"LLM call failed after 3 retries at iteration {iter_num}: {exc}"
                        self._session.log_error("EXECUTE", error_msg, exc)
                        result.errors.append(error_msg)
            if response is None:
                break

            result.iterations = iter_num

            assistant_text, tool_uses = self._parse_response(response)
            usage = response.get("usage")
            self._trace.llm_response(assistant_text, tool_uses, usage)

            if assistant_text:
                result.final_text = assistant_text
                self._session.log(
                    "EXECUTE", f"Turn {iter_num}: LLM text ({len(assistant_text)} chars)",
                )

            if self._use_text_tools:
                self._messages.append({"role": "assistant", "content": assistant_text})
            else:
                self._messages.append(self._build_assistant_message(response))

            if not tool_uses:
                self._session.log(
                    "EXECUTE", f"Turn {iter_num}: no tool calls — loop complete",
                )
                break

            self._session.log(
                "EXECUTE",
                f"Turn {iter_num}: {len(tool_uses)} tool call(s): "
                f"{[tu['function']['name'] for tu in tool_uses]}",
            )

            for tu in tool_uses:
                tool_name = tu["function"]["name"]
                tool_id = tu.get("id", f"call_{result.tool_calls}")
                try:
                    tool_input = json.loads(tu["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    tool_input = {}

                result.tool_calls += 1
                self._session.llm_calls += 1

                perm_error = self._permissions.check(tool_name, tool_input)
                if perm_error:
                    self._session.log("EXECUTE", f"  DENIED {tool_name}: {perm_error}")
                    self._trace.permission_denied(tool_name, perm_error)
                    if self._use_text_tools:
                        self._messages.append({
                            "role": "user",
                            "content": self._format_text_tool_feedback(
                                tool_name,
                                tool_input,
                                f"PERMISSION DENIED: {perm_error}",
                            ),
                        })
                    else:
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": f"PERMISSION DENIED: {perm_error}",
                        })
                    continue

                self._session.log(
                    "EXECUTE",
                    f"  Executing {tool_name}({self._summarize_input(tool_name, tool_input)})",
                )
                tool_t0 = time.monotonic()
                tool_result, is_error = self._executor.execute(tool_name, tool_input)
                tool_elapsed_ms = int((time.monotonic() - tool_t0) * 1000)

                self._trace.tool_call(
                    tool_name, tool_input, tool_result, is_error, tool_elapsed_ms,
                )

                if is_error:
                    self._session.log(
                        "EXECUTE",
                        f"  {tool_name} ERROR ({tool_elapsed_ms}ms): {tool_result[:200]}",
                    )
                else:
                    self._session.log(
                        "EXECUTE",
                        f"  {tool_name} OK ({tool_elapsed_ms}ms, {len(tool_result)} chars)",
                    )

                if self._use_text_tools:
                    self._messages.append({
                        "role": "user",
                        "content": self._format_text_tool_feedback(
                            tool_name, tool_input, tool_result,
                        ),
                    })
                else:
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result,
                    })

            ws_files = self._list_workspace_files()
            self._trace.iteration_end(ws_files)

            # Run verification hooks (one-shot: remove after firing)
            fired: list[int] = []
            for idx, hook in enumerate(self._verification_hooks):
                try:
                    inject = hook(self._workspace, tool_uses, ws_files)
                except Exception as exc:
                    self._session.log("VERIFY", f"Hook {idx} error: {exc}")
                    inject = None
                if inject:
                    self._session.log("VERIFY", f"Hook {idx} fired, injecting message")
                    self._messages.append({"role": "user", "content": inject})
                    fired.append(idx)
            for idx in reversed(fired):
                self._verification_hooks.pop(idx)

        else:
            self._session.log(
                "EXECUTE", f"Turn loop hit max iterations ({self._max_iterations})",
            )

        result.files = self._collect_workspace_files()
        result.elapsed_sec = time.monotonic() - t0

        self._trace.loop_end(result)
        self._session.log(
            "EXECUTE",
            f"Turn loop done: {result.iterations} iters, "
            f"{result.tool_calls} tool calls, "
            f"{len(result.files)} files, {result.elapsed_sec:.1f}s",
        )
        self._save_conversation_log()
        return result

    # ------------------------------------------------------------------
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(self) -> dict[str, Any]:
        cfg = self._llm_config
        base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"

        system_prompt = self._system_prompt
        if self._use_text_tools:
            system_prompt += self._text_tool_prompt

        model = cfg.primary_model
        if self._use_text_tools:
            _coding = getattr(cfg, "coding_model", "") or ""
            if _coding and self._is_claude_model(_coding):
                model = _coding

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *self._messages,
            ],
            "max_tokens": 8192,
        }

        if not self._use_text_tools:
            body["tools"] = self._api_tools
            body["tool_choice"] = "auto"

        if any(cfg.primary_model.startswith(p) for p in ("o3", "o4", "gpt-5")):
            body["max_tokens"] = 16384

        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }

        req = urllib.request.Request(url, data=payload, headers=headers)
        timeout = getattr(cfg, "timeout_sec", 600)

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data

    def _parse_response(
        self, data: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        choices = data.get("choices", [])
        if not choices:
            return "", []
        message = choices[0].get("message", {})
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls", [])

        if not tool_calls and text.strip():
            recovered = self._try_recover_tool_calls_from_text(text)
            if recovered:
                logger.info(
                    "[turn_loop] Recovered %d tool call(s) from text content",
                    len(recovered),
                )
                tool_calls = recovered
                if not self._use_text_tools:
                    message["tool_calls"] = recovered

        return text, tool_calls

    def _try_recover_tool_calls_from_text(
        self, text: str,
    ) -> list[dict[str, Any]]:
        """Parse tool call JSON embedded in assistant text as fallback."""
        import re
        import uuid

        valid_tool_names = {spec["name"] for spec in TOOL_SPECS}

        candidates: list[dict[str, Any]] = []
        stripped = text.strip()

        blobs: list[Any] = []
        try:
            parsed = json.loads(stripped)
            blobs = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, ValueError):
            for m in re.finditer(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}', stripped):
                try:
                    blobs.append(json.loads(m.group()))
                except (json.JSONDecodeError, ValueError):
                    pass

        for blob in blobs:
            if not isinstance(blob, dict):
                continue
            tool_name = blob.get("tool") or blob.get("name") or blob.get("function", {}).get("name")
            if not tool_name or tool_name not in valid_tool_names:
                continue
            params = blob.get("parameters") or blob.get("arguments") or blob.get("input") or {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, ValueError):
                    params = {}

            if isinstance(params, dict):
                params = self._normalize_tool_params(tool_name, params)

            candidates.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(params),
                },
            })

        return candidates

    @staticmethod
    def _normalize_tool_params(tool_name: str, params: dict) -> dict:
        """Map common parameter name aliases to canonical names."""
        _ALIASES: dict[str, dict[str, str]] = {
            "read_file": {"filename": "path", "file": "path", "file_path": "path", "filepath": "path"},
            "write_file": {"filename": "path", "file": "path", "file_path": "path",
                           "contents": "content", "text": "content", "data": "content"},
            "edit_file": {"filename": "path", "file": "path", "file_path": "path",
                          "find": "old_string", "search": "old_string",
                          "replace": "new_string", "replacement": "new_string"},
            "glob_search": {"glob": "pattern", "glob_pattern": "pattern",
                            "directory": "path", "dir": "path"},
            "grep_search": {"regex": "pattern", "query": "pattern",
                            "directory": "path", "dir": "path"},
            "bash": {"cmd": "command", "script": "command", "shell": "command"},
        }
        aliases = _ALIASES.get(tool_name, {})
        if not aliases:
            return params
        normalized = {}
        for k, v in params.items():
            canonical = aliases.get(k, k)
            normalized[canonical] = v
        return normalized

    @staticmethod
    def _build_assistant_message(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {"role": "assistant", "content": ""})
        return {"role": "assistant", "content": ""}

    def _build_api_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in self._tool_specs
        ]

    @staticmethod
    def _is_claude_model(model_name: str) -> bool:
        return "claude" in model_name.lower()

    def _build_text_tool_prompt(self) -> str:
        """Build tool descriptions for system prompt (text-based mode).

        Used when the proxy doesn't reliably translate structured
        tool calls (e.g. Claude via OpenAI-compatible proxy).
        """
        lines = [
            "\n\n---\n## Tool Calling\n",
            "To use a tool, output EXACTLY one JSON object per message:",
            '{"tool": "<tool_name>", "parameters": {<params>}}',
            "",
            "IMPORTANT: Output ONLY the JSON object. No extra text.",
            "After receiving the tool result, decide the next action.",
            "",
            "Available tools:",
        ]
        for spec in self._tool_specs:
            name = spec["name"]
            desc = spec["description"]
            schema = spec["input_schema"]
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            lines.append(f"\n### {name}")
            lines.append(desc)
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                req = " **(required)**" if pname in required else ""
                pdesc = pinfo.get("description", "")
                ptype = pinfo.get("type", "string")
                lines.append(f"  - `{pname}` ({ptype}): {pdesc}{req}")
        return "\n".join(lines)

    @staticmethod
    def _format_text_tool_feedback(
        tool_name: str, tool_input: dict[str, Any], tool_result: str,
    ) -> str:
        """Return a text-only tool transcript for Claude-style loops."""
        return (
            f"Tool result ({tool_name})\n"
            f"Arguments: {json.dumps(tool_input, ensure_ascii=False, sort_keys=True)}\n"
            f"Result:\n{tool_result}"
        )

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _collect_workspace_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for py_file in sorted(self._workspace.rglob("*.py")):
            if py_file.is_symlink():
                continue
            rel = py_file.relative_to(self._workspace)
            if any(
                p.startswith(".") or p == "__pycache__" or p == "codebases"
                for p in rel.parts
            ):
                continue
            try:
                files[str(rel)] = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        for extra in ("requirements.txt", "results.json"):
            p = self._workspace / extra
            if p.exists():
                try:
                    files[extra] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
        return files

    def _list_workspace_files(self) -> list[str]:
        result = []
        for f in sorted(self._workspace.rglob("*")):
            if f.is_file() and not f.is_symlink():
                rel = f.relative_to(self._workspace)
                if not any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
                    result.append(str(rel))
        return result

    @staticmethod
    def _summarize_input(tool_name: str, inp: dict[str, Any]) -> str:
        if tool_name == "bash":
            cmd = inp.get("command", "")
            return cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif tool_name in ("write_file", "edit_file"):
            path = inp.get("path", "?")
            size = len(inp.get("content", inp.get("new_string", "")))
            return f"{path} ({size} chars)"
        elif tool_name == "read_file":
            return inp.get("path", "?")
        elif tool_name in ("glob_search", "grep_search"):
            return inp.get("pattern", "?")
        return json.dumps(inp)[:80]

    def _save_conversation_log(self) -> None:
        trace_dir = self._workspace.parent if self._workspace.parent.is_dir() else self._workspace
        try:
            full_path = trace_dir / "turn_loop_conversation_full.json"
            full_path.write_text(
                json.dumps(self._messages, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            log_path = trace_dir / "turn_loop_conversation.json"
            safe_messages = []
            for msg in self._messages:
                safe = dict(msg)
                content = safe.get("content", "")
                if isinstance(content, str) and len(content) > 3000:
                    safe["content"] = content[:3000] + f"\n... [{len(content)} total chars]"
                if "tool_calls" in safe:
                    for tc in safe.get("tool_calls", []):
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            args = fn.get("arguments", "")
                            if isinstance(args, str) and len(args) > 2000:
                                fn["arguments"] = args[:2000] + f"... [{len(args)} total]"
                safe_messages.append(safe)
            log_path.write_text(
                json.dumps(safe_messages, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass
