"""Claw-code agentic turn loop for experiment code generation.

Ported from claw-code ``rust/crates/runtime/src/conversation.rs``
``ConversationRuntime::run_turn()``. The loop:

    user_message → (LLM call → tool execution →)* → done

The LLM receives tools via the API ``tools`` field and returns
``tool_use`` content blocks. Each tool call is executed, and the result
is fed back as a ``tool`` role message for the next iteration.
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
from typing import Any

from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.tools.definitions import TOOL_SPECS
from researchclaw.pipeline.codegen.tools.executor import ToolExecutor
from researchclaw.pipeline.codegen.tools.permissions import SandboxPermissionPolicy
from researchclaw.pipeline.codegen.types import CodegenPhase, GeneratedFiles

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 40
MAX_RESULT_CHARS = 8000


@dataclass
class TurnResult:
    """Result of a complete turn loop execution."""
    files: GeneratedFiles = field(default_factory=dict)
    iterations: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)
    final_text: str = ""
    elapsed_sec: float = 0.0


class _TraceLog:
    """Step-by-step trace logger for debugging code generation.

    Writes a human-readable markdown file (``generation_trace.md``)
    showing every LLM call, tool invocation, input/output, and file
    change in chronological order — like a git log for code generation.
    """

    def __init__(self, trace_dir: Path) -> None:
        self._path = trace_dir / "generation_trace.md"
        self._step = 0
        self._write("# Code Generation Trace\n")
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
        self, name: str, input_data: dict, result: str, is_error: bool, elapsed_ms: int
    ) -> None:
        status = "ERROR" if is_error else "OK"
        self._write(f"\n#### Tool: `{name}` [{status}] ({elapsed_ms}ms)\n")

        # Log input
        if name == "bash":
            cmd = input_data.get("command", "")
            self._write(f"**Command:**\n```bash\n{cmd}\n```\n")
        elif name == "write_file":
            path = input_data.get("path", "?")
            content = input_data.get("content", "")
            n_lines = len(content.splitlines())
            self._write(f"**Path:** `{path}` ({n_lines} lines, {len(content)} chars)\n")
            preview = content[:800] + ("\n... [truncated]" if len(content) > 800 else "")
            self._write(f"```python\n{preview}\n```\n")
        elif name == "edit_file":
            path = input_data.get("path", "?")
            old = input_data.get("old_string", "")[:200]
            new = input_data.get("new_string", "")[:200]
            self._write(
                f"**Path:** `{path}`\n"
                f"**old_string:** `{old}`\n"
                f"**new_string:** `{new}`\n"
            )
        elif name == "read_file":
            self._write(f"**Path:** `{input_data.get('path', '?')}`\n")
        elif name in ("glob_search", "grep_search"):
            self._write(f"**Pattern:** `{input_data.get('pattern', '?')}`\n")
            if input_data.get("path"):
                self._write(f"**In:** `{input_data['path']}`\n")

        # Log result
        result_preview = result[:1000] + ("\n... [truncated]" if len(result) > 1000 else "")
        self._write(f"**Result:**\n```\n{result_preview}\n```\n")

    def permission_denied(self, name: str, reason: str) -> None:
        self._write(f"\n#### Tool: `{name}` [DENIED]\n**Reason:** {reason}\n")

    def iteration_end(self, files_in_workspace: list[str]) -> None:
        if files_in_workspace:
            self._write(
                f"\n**Workspace files after this iteration:** "
                f"{', '.join(f'`{f}`' for f in files_in_workspace)}\n"
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
        if result.errors:
            self._write("### Errors\n")
            for e in result.errors:
                self._write(f"- {e}\n")
        self._write(f"\nCompleted: {datetime.now(timezone.utc).isoformat()}\n")

    def _write(self, text: str) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


class ClawTurnLoop:
    """Agentic turn loop: LLM iteratively calls tools to generate code.

    Ported from claw-code's ``ConversationRuntime``:
    - Max 16 iterations (same as claw-code)
    - Tool results fed back as ``tool`` role messages
    - Stops when LLM responds without tool calls
    """

    def __init__(
        self,
        *,
        llm_config: Any,
        workspace: Path,
        system_prompt: str,
        session: CodegenSession,
        allowed_read_dirs: list[Path] | None = None,
        bash_timeout: int = 60,
        max_iterations: int = MAX_ITERATIONS,
        python_path: str = "",
    ) -> None:
        self._llm_config = llm_config
        self._workspace = workspace
        self._system_prompt = system_prompt
        self._session = session
        self._max_iterations = max_iterations
        self._messages: list[dict[str, Any]] = []

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
        self._simulation_check_done = False
        self._plan_check_done = False
        self._exp_plan = ""

        # Step-by-step trace log for debugging
        trace_dir = workspace.parent if workspace.parent.is_dir() else workspace
        self._trace = _TraceLog(trace_dir)

    def set_exp_plan(self, plan: str) -> None:
        """Set the experiment plan for plan compliance checking."""
        self._exp_plan = plan

    def run_turn(self, user_message: str) -> TurnResult:
        """Execute the full turn loop with detailed step-by-step tracing."""
        t0 = time.monotonic()
        self._session.log(CodegenPhase.GENERATE, "Turn loop started")
        self._messages.append({"role": "user", "content": user_message})

        result = TurnResult()

        for iteration in range(self._max_iterations):
            iter_num = iteration + 1
            self._trace.iteration_start(iter_num, self._max_iterations)
            self._session.log(
                CodegenPhase.GENERATE,
                f"Turn {iter_num}/{self._max_iterations}: calling LLM...",
            )

            # ── LLM call ──
            self._trace.llm_request(
                n_messages=len(self._messages),
                n_tools=len(self._api_tools),
                model=self._llm_config.primary_model,
            )

            try:
                response = self._call_llm()
            except Exception as exc:
                error_msg = f"LLM call failed at iteration {iter_num}: {exc}"
                self._session.log_error(CodegenPhase.GENERATE, error_msg, exc)
                result.errors.append(error_msg)
                break

            result.iterations = iter_num

            # ── Parse response ──
            assistant_text, tool_uses = self._parse_response(response)
            usage = response.get("usage")
            self._trace.llm_response(assistant_text, tool_uses, usage)

            if assistant_text:
                result.final_text = assistant_text
                self._session.log(
                    CodegenPhase.GENERATE,
                    f"Turn {iter_num}: LLM text ({len(assistant_text)} chars)",
                )

            self._messages.append(self._build_assistant_message(response))

            if not tool_uses:
                self._session.log(
                    CodegenPhase.GENERATE,
                    f"Turn {iter_num}: no tool calls — loop complete",
                )
                break

            self._session.log(
                CodegenPhase.GENERATE,
                f"Turn {iter_num}: {len(tool_uses)} tool call(s): "
                f"{[tu['function']['name'] for tu in tool_uses]}",
            )

            # ── Execute each tool call ──
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
                    self._session.log(
                        CodegenPhase.GENERATE,
                        f"  DENIED {tool_name}: {perm_error}",
                    )
                    self._trace.permission_denied(tool_name, perm_error)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": f"PERMISSION DENIED: {perm_error}",
                    })
                    continue

                self._session.log(
                    CodegenPhase.GENERATE,
                    f"  Executing {tool_name}({self._summarize_input(tool_name, tool_input)})",
                )
                tool_t0 = time.monotonic()
                tool_result, is_error = self._executor.execute(tool_name, tool_input)
                tool_elapsed_ms = int((time.monotonic() - tool_t0) * 1000)

                # Write to trace log
                self._trace.tool_call(
                    tool_name, tool_input, tool_result, is_error, tool_elapsed_ms,
                )

                if is_error:
                    self._session.log(
                        CodegenPhase.GENERATE,
                        f"  {tool_name} ERROR ({tool_elapsed_ms}ms): {tool_result[:200]}",
                    )
                else:
                    self._session.log(
                        CodegenPhase.GENERATE,
                        f"  {tool_name} OK ({tool_elapsed_ms}ms, {len(tool_result)} chars)",
                    )

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result,
                })

            # Log workspace state after each iteration
            ws_files = self._list_workspace_files()
            self._trace.iteration_end(ws_files)

            # ── Anti-simulation gate ──
            # After the agent writes main.py and runs it successfully,
            # auto-inject a verification message forcing the self-check.
            if (
                "main.py" in ws_files
                and not self._simulation_check_done
                and any(
                    tu["function"]["name"] == "bash"
                    and "python" in tu["function"].get("arguments", "")
                    for tu in tool_uses
                )
            ):
                check_result = self._run_simulation_check()
                if check_result:
                    self._simulation_check_done = True
                    self._session.log(
                        CodegenPhase.VALIDATE,
                        f"Anti-simulation gate: {check_result[:200]}",
                    )
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "ANTI-SIMULATION VERIFICATION FAILED. Your code uses "
                            "forbidden simulation patterns. Here are the results:\n\n"
                            f"{check_result}\n\n"
                            "You MUST rewrite main.py to use REAL pretrained models "
                            "(from_pretrained, torchvision.models, timm.create_model) "
                            "instead of nn.Linear or numpy transformations. "
                            "The experiment plan specifies specific methods — implement them "
                            "using the actual ML libraries and local checkpoints. "
                            "Do NOT use brightness/contrast/augmentation as experimental conditions."
                        ),
                    })

            # ── Plan compliance gate ──
            # Like claw-code's CLAUDE.md instructions that persist across turns,
            # we check the code against the experiment plan's key requirements
            # and inject corrective messages when violations are found.
            if (
                "main.py" in ws_files
                and not self._plan_check_done
                and self._exp_plan
                and any(
                    tu["function"]["name"] == "bash"
                    and "python" in tu["function"].get("arguments", "")
                    for tu in tool_uses
                )
            ):
                plan_violations = self._run_plan_compliance_check()
                if plan_violations:
                    self._plan_check_done = True
                    self._session.log(
                        CodegenPhase.VALIDATE,
                        f"Plan compliance gate: {len(plan_violations)} violation(s)",
                    )
                    violation_text = "\n".join(f"- {v}" for v in plan_violations)
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "PLAN COMPLIANCE CHECK FAILED. Your code does NOT follow "
                            "the experiment plan. Violations found:\n\n"
                            f"{violation_text}\n\n"
                            "You MUST fix these violations. Re-read the EXPERIMENT_PLAN.yaml "
                            "and implement what it specifies. Do NOT take shortcuts — "
                            "if the plan says to load reference frames from a directory, "
                            "load them; if it defines 4 conditions, implement all 4 with "
                            "genuinely different logic."
                        ),
                    })

        else:
            self._session.log(
                CodegenPhase.GENERATE,
                f"Turn loop hit max iterations ({self._max_iterations})",
            )

        result.files = self._collect_workspace_files()
        result.elapsed_sec = time.monotonic() - t0

        self._trace.loop_end(result)

        self._session.log(
            CodegenPhase.GENERATE,
            f"Turn loop done: {result.iterations} iterations, "
            f"{result.tool_calls} tool calls, "
            f"{len(result.files)} files, {result.elapsed_sec:.1f}s",
        )

        self._save_conversation_log()
        return result

    # ------------------------------------------------------------------
    # LLM API call with tool support
    # ------------------------------------------------------------------

    def _call_llm(self) -> dict[str, Any]:
        """Call the LLM API with tool definitions (OpenAI-compatible)."""
        cfg = self._llm_config
        base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"

        body: dict[str, Any] = {
            "model": cfg.primary_model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                *self._messages,
            ],
            "max_tokens": 8192,
            "tools": self._api_tools,
            "tool_choice": "auto",
        }

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
        self, data: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract text content and tool_use calls from API response."""
        choices = data.get("choices", [])
        if not choices:
            return "", []

        message = choices[0].get("message", {})
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls", [])
        return text, tool_calls

    @staticmethod
    def _build_assistant_message(data: dict[str, Any]) -> dict[str, Any]:
        """Build the assistant message to append to conversation history."""
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {"role": "assistant", "content": ""})
        return {"role": "assistant", "content": ""}

    def _build_api_tools(self) -> list[dict[str, Any]]:
        """Convert tool specs to OpenAI-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in TOOL_SPECS
        ]

    # ------------------------------------------------------------------
    # Workspace file collection
    # ------------------------------------------------------------------

    def _collect_workspace_files(self) -> GeneratedFiles:
        """Collect all .py files from workspace (new or modified)."""
        files: GeneratedFiles = {}
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

        for extra in ("requirements.txt",):
            p = self._workspace / extra
            if p.exists():
                try:
                    files[extra] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        return files

    def _list_workspace_files(self) -> list[str]:
        """Quick list of workspace files (for trace logging)."""
        result = []
        for f in sorted(self._workspace.rglob("*")):
            if f.is_file() and not f.is_symlink():
                rel = f.relative_to(self._workspace)
                if not any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
                    result.append(str(rel))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_simulation_check(self) -> str | None:
        """Run anti-simulation checks on main.py.

        Returns a failure message if simulation is detected, None if clean.
        """
        main_py = self._workspace / "main.py"
        if not main_py.exists():
            return None

        code = main_py.read_text(encoding="utf-8")
        lines = code.splitlines()
        violations: list[str] = []

        # Check 1: nn.Linear used as primary model (not LoRA adapter)
        for i, line in enumerate(lines, 1):
            if "nn.Linear" in line:
                line_lower = line.lower()
                if not any(kw in line_lower for kw in (
                    "lora", "adapter", "projection", "head", "classifier",
                    "fc", "linear_probe", "to_out", "to_q", "to_k", "to_v",
                )):
                    violations.append(f"FORBIDDEN nn.Linear as model → Line {i}: {line.strip()}")

        # Check 2: Mock functions
        for i, line in enumerate(lines, 1):
            if any(pat in line for pat in ("_mock", "mock_", "random.uniform")):
                violations.append(f"FORBIDDEN mock/random metric → Line {i}: {line.strip()}")

        # Check 3: try/except returning hardcoded metrics
        # Pattern: except ... return {"fid": 0.8, ...}
        in_except = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("except"):
                in_except = True
            elif in_except and stripped.startswith("return"):
                # Check if the return contains hardcoded numbers
                import re
                hardcoded = re.findall(r'"(?:fid|clip|metric|score|loss)":\s*[\d.]+', stripped)
                if hardcoded:
                    violations.append(
                        f"FORBIDDEN hardcoded fallback metric in except block → "
                        f"Line {i}: {stripped[:100]}"
                    )
                in_except = False
            elif in_except and not stripped.startswith(("print", "#", "")):
                if not stripped.startswith(" ") and stripped:
                    in_except = False

        # Check 4: bare except that silently swallows errors
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped in ("except:", "except Exception:", "except Exception as e:"):
                # Look ahead for return with numbers or pass
                for j in range(i, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line == "pass":
                        violations.append(
                            f"FORBIDDEN silent except/pass → Line {i}: {stripped}"
                        )
                        break
                    if next_line.startswith("return") and any(
                        c.isdigit() for c in next_line
                    ):
                        violations.append(
                            f"FORBIDDEN except returning hardcoded value → "
                            f"Line {i}-{j+1}: {stripped} ... {next_line[:80]}"
                        )
                        break

        # Check 5: No real model loading
        has_real_model = any(pat in code for pat in (
            "from_pretrained", "load_state_dict", "create_model",
            "timm.", "torchvision.models",
        ))
        if not has_real_model:
            violations.append(
                "MISSING: No real model loading found (from_pretrained, "
                "load_state_dict, create_model, torchvision.models)"
            )

        # Check 6: Image augmentation as "experimental condition"
        augment_patterns = ["brightness", "contrast", "augment_image", "enhance("]
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            if any(pat in line_lower for pat in augment_patterns):
                if "condition" in line_lower or "def condition_" in line_lower:
                    violations.append(
                        f"FORBIDDEN augmentation as condition → Line {i}: {line.strip()}"
                    )

        if not violations:
            return None

        return (
            "ANTI-SIMULATION CHECK FAILED — " + str(len(violations)) + " violation(s):\n\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\n\nYou MUST fix ALL violations:\n"
            "- Remove ALL try/except blocks around model code. Let errors crash.\n"
            "- Remove ALL hardcoded fallback return values.\n"
            "- Use the correct model loading approach based on what you discovered in the workspace.\n"
            "- Each condition must have genuinely different code, not copy-paste with minor changes."
        )

    def _run_plan_compliance_check(self) -> list[str]:
        """Check code against experiment plan requirements.

        Inspired by claw-code's CLAUDE.md instruction files which persist
        across all turns and enforce project-specific rules. We extract
        verifiable constraints from the experiment plan and check the code.
        """
        main_py = self._workspace / "main.py"
        if not main_py.exists() or not self._exp_plan:
            return []

        code = main_py.read_text(encoding="utf-8")
        plan = self._exp_plan.lower()
        violations: list[str] = []

        # 1. Check all methods/conditions from plan are implemented
        import re
        # Extract method names from plan (look for method definitions in YAML)
        plan_methods = set()
        for match in re.finditer(r'^\s{2}(\w+):\s*$', self._exp_plan, re.MULTILINE):
            name = match.group(1)
            if name not in (
                "name", "source", "path", "format", "description", "algorithm",
                "trainable_params", "notes", "class", "architecture",
                "lora_target_modules",
            ):
                plan_methods.add(name)

        # Check each plan method has a corresponding function in code
        for method in plan_methods:
            # Look for def method_name or def run_method_name etc
            if method not in code.lower().replace("-", "_"):
                violations.append(
                    f"MISSING CONDITION: Plan defines '{method}' but no matching "
                    f"function found in code. You must implement ALL conditions from the plan."
                )

        # 2. Check reference frames: if plan mentions first_frames, code must load them
        if "first_frames" in plan and "first_frames" in code.lower():
            if "torch.randint" in code and "fid" in code.lower():
                violations.append(
                    "FAKE REFERENCE FRAMES: Code uses torch.randint() as FID reference "
                    "frames instead of loading real images from first_frames directory. "
                    "You MUST load actual PNG files: Image.open(path).convert('RGB')"
                )

        # 2b. Check for hardcoded metric return values (e.g. "clip_score": 0.0)
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            if "return" in stripped:
                import re as _re
                hardcoded = _re.findall(
                    r'"(?:clip_score|fid|metric|score|loss|accuracy)":\s*0\.0',
                    stripped,
                )
                if hardcoded:
                    violations.append(
                        f"HARDCODED METRIC: Line {i} returns hardcoded 0.0 for {hardcoded[0].split(':')[0]}. "
                        f"Every metric must be computed from real model output, not hardcoded."
                    )

        # 2c. Optional metrics may be skipped, but should not silently become NaN.
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip().lower()
            if any(pat in stripped for pat in ('"clip_score": math.nan', "'clip_score': math.nan", '"clip_score": float("nan")', "'clip_score': float(\"nan\")")):
                violations.append(
                    f"UNCLEAR METRIC STATUS: Line {i} returns NaN for clip_score. "
                    "If CLIP score is unavailable offline, return an explicit skipped reason/status instead of NaN."
                )

        # 2d. Check for fake attribute assignments (only if plan mentions adaptive LoRA)
        if any(kw in plan for kw in ("adaptive", "rank_pattern", "per-layer", "per_layer")):
            for i, line in enumerate(code.splitlines(), 1):
                stripped = line.strip()
                if "module.r =" in stripped or "module.lora_alpha =" in stripped:
                    violations.append(
                        f"FAKE ADAPTIVE LORA: Line {i}: '{stripped}' — setting module.r/module.lora_alpha "
                        f"does NOT change the LoRA matrix dimensions. To get different ranks per layer, "
                        f"you must apply separate LoraConfig objects to different layer groups with "
                        f"different adapter_name parameters."
                    )

        # 2e. Video experiments should not silently collapse to a single frame in evaluation.
        if any(kw in plan for kw in ("video", "num_frames", "t2v", "i2v")):
            for i, line in enumerate(code.splitlines(), 1):
                stripped = line.strip().replace(" ", "")
                if "frames[0][0]" in stripped or ".frames[0][0]" in stripped:
                    violations.append(
                        f"SINGLE-FRAME VIDEO EVAL: Line {i} only uses the first generated frame. "
                        f"Video evaluation should iterate over `output.frames[0]` unless the plan explicitly says first-frame-only."
                    )
                    break

        # 3. Check training: if plan specifies training steps, code must have training loop
        if any(kw in plan for kw in ("max_steps", "training", "train_", "optimizer")):
            has_training = any(kw in code for kw in (
                "loss.backward()", "optimizer.step()", "train()",
            ))
            if not has_training:
                violations.append(
                    "MISSING TRAINING: Plan specifies training but code has no "
                    "training loop (no loss.backward() or optimizer.step() found). "
                    "You MUST include the full training loop as specified in the plan."
                )

        # 3b. S11 code must support a lightweight smoke mode without changing algorithm semantics.
        if any(kw in plan for kw in ("training", "evaluation", "max_steps", "num_frames")):
            if "smoke_test" not in code.lower():
                violations.append(
                    "MISSING SMOKE MODE: main.py must support `SMOKE_TEST=1` for lightweight verification "
                    "while keeping the default execution path as the full experiment."
                )

        # 4. Check each condition is genuinely different (not copy-paste)
        # Extract function bodies and compare
        func_bodies: dict[str, str] = {}
        for match in re.finditer(
            r'^def (\w+)\(.*?\):\s*\n((?:[ \t]+.*\n)*)',
            code, re.MULTILINE,
        ):
            fname = match.group(1)
            body = match.group(2).strip()
            if fname.startswith("lora_") or fname.startswith("baseline"):
                # Normalize: remove comments, whitespace variations
                normalized = re.sub(r'#.*$', '', body, flags=re.MULTILINE)
                normalized = re.sub(r'\s+', ' ', normalized).strip()
                func_bodies[fname] = normalized

        # Check for duplicate function bodies
        seen_bodies: dict[str, str] = {}
        for fname, body in func_bodies.items():
            # Compare ignoring small differences (numbers, variable names)
            body_sig = re.sub(r'\d+', 'N', body)[:500]
            for prev_name, prev_sig in seen_bodies.items():
                if body_sig == prev_sig:
                    violations.append(
                        f"DUPLICATE CONDITION: '{fname}' has identical logic to '{prev_name}'. "
                        f"Each condition MUST implement a genuinely different algorithm."
                    )
                    break
            seen_bodies[fname] = body_sig

        return violations

    @staticmethod
    def _summarize_input(tool_name: str, inp: dict[str, Any]) -> str:
        """One-line summary of tool input for logging."""
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
        """Save conversation history in two formats for debugging.

        1. ``turn_loop_conversation.json`` — full messages (truncated for size)
        2. ``turn_loop_conversation_full.json`` — completely untruncated
        """
        trace_dir = self._workspace.parent if self._workspace.parent.is_dir() else self._workspace

        # Full untruncated conversation (for deep debugging)
        try:
            full_path = trace_dir / "turn_loop_conversation_full.json"
            full_path.write_text(
                json.dumps(self._messages, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

        # Truncated version (for quick inspection)
        try:
            log_path = trace_dir / "turn_loop_conversation.json"
            safe_messages = []
            for msg in self._messages:
                safe = dict(msg)
                content = safe.get("content", "")
                if isinstance(content, str) and len(content) > 3000:
                    safe["content"] = content[:3000] + f"\n... [{len(content)} total chars]"
                # Also truncate tool_calls arguments
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
