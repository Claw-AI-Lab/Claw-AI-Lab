"""Aider Beast Mode bridge for experiment code generation.

Uses Aider CLI (headless --message mode) to generate experiment code
via a TODO-driven loop: first generates a skeleton with TODO markers,
then iteratively fills in each TODO until none remain.

Aider reads the workspace codebase via its repo-map feature, understanding
the full code structure before generating/modifying files.

Re-exports complexity scoring and result types from opencode_bridge
so executor.py can import from this module interchangeably.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.pipeline.opencode_bridge import (
    ComplexityScore,
    OpenCodeResult,
    count_historical_failures,
    score_complexity,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenHandsBridge",
    "OpenCodeResult",
    "ComplexityScore",
    "score_complexity",
    "count_historical_failures",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RULES = """\
CRITICAL: Read EXPERIMENT_PLAN.yaml FIRST — it defines method names, algorithm steps, baselines, metrics.
Read GUIDANCE.md for ABSOLUTE paths to data/checkpoints/codebases. Read codebase .py files to learn the API.

RULES:
- Build ON TOP of existing codebase. Use ABSOLUTE paths from GUIDANCE.md.
- Load ALL models from LOCAL CHECKPOINTS_DIR paths. NEVER download from internet (no HuggingFace IDs, no `pretrained=True`, no URLs, no `torch.hub.load`).
- Each condition must implement a genuinely DIFFERENT algorithm, not just different hyperparameters.
- Metrics MUST be computed from ACTUAL model outputs. NEVER hardcode fake values or compute from formulas/profiles.
- NO try/except anywhere (except save_outputs for file I/O). Let all errors crash with full traceback.
- NO argparse. NO wrapper classes with empty methods. NO files named `utils.py`/`models.py`/`config.py` that shadow codebase modules.
- sys.path.insert: add the REPOSITORY ROOT only, NOT subdirectories.
- When calling codebase functions, READ function bodies to understand expected args (e.g. pass pipeline object, not pipeline.unet).
"""

_RULES_NO_CODEBASE = """\
NO-CODEBASE: You MUST use real ML libraries (torch, diffusers, transformers, peft) — NEVER simulate with numpy/PIL.
Load checkpoints via `from_pretrained(CHECKPOINTS_DIR)`. Prioritize topic requirements over generic benchmark suggestions.
"""

_SKELETON_PROMPT = _RULES + """
TASK: Generate a SHORT main.py SKELETON (<130 lines). Do NOT implement any logic — use `pass` for function bodies.

1. Read EXPERIMENT_PLAN.yaml. Pick 1 baseline + 2 proposed methods (3 total).
2. Create main.py with plain functions (no classes):
   - Imports + sys.path.insert(0, REPO_ROOT)
   - Constants: DATASETS_DIR, CHECKPOINTS_DIR, OUTPUT_DIR='outputs', TIME_BUDGET={time_budget_sec}, SEEDS=[42,123,456]
   - set_seed(seed) — IMPLEMENT (3 lines)
   - should_stop() — IMPLEMENT (2 lines)
   - load_pipeline(), load_data(), compute_metric(), save_outputs() — each: `# TODO: <what>` then `pass`
   - 3 condition functions (REAL names from plan) — each: `# TODO: <algorithm steps>` then `pass`
   - run_condition() — IMPLEMENT (dispatch dict + compute_metric + save_outputs, NO try/except)
   - main() — IMPLEMENT (loop conditions/seeds, print results, NO try/except around run_condition)
   - if __name__ == '__main__': main()
"""

_FILL_TODO_PROMPT = _RULES + """
TASK: Implement ONE TODO in main.py. The TODO:

{todo_line}

- Read codebase .py files and EXPERIMENT_PLAN.yaml first. Implement ONLY this function, keep output SHORT.
- Remove `# TODO:` and replace `pass` with real implementation using ABSOLUTE paths from constants.
- save_outputs(): save visual artifacts (PNG/curves/text) proving model ran. try/except allowed ONLY here.
"""

_FIX_PROMPT = """\
The file main.py has a syntax error or import error:

{error_output}

Fix the error. Make the MINIMAL change needed. Do NOT rewrite the entire file.
"""

# Single-shot fallback
_FALLBACK_PROMPT = _RULES + """
Create a COMPLETE main.py using EXPERIMENT_PLAN.yaml method names, algorithm steps, and metric definitions.
3 conditions (1 baseline + 2 proposed) with genuinely different algorithms.
Print: {metric}: <value> for each condition and seed. Time budget: {time_budget_sec}s.
Save visual results to `outputs/{{condition}}_{{seed}}.png`.
"""

_FIX_SANITY_PROMPT = _RULES + """
TASK: Fix a sanity check failure in main.py — make the smallest surgical change, do NOT rewrite the file.

**Failed test:** `{test_name}`
**Test code:**
```python
{test_code}
```
**Error (stderr tail):**
```
{stderr}
```
{repeat_hint}
## DIAGNOSE FIRST (read relevant files before fixing):
- Path errors: read the ACTUAL config YAML and GUIDANCE.md directory tree. Use `os.path.basename()` to extract filenames, rebuild paths from DATASETS_DIR. Never blindly join nested relative paths.
- NoneType errors: read the reference implementation (e.g. `inference.py`) for correct values. Search ALL attribute accesses on that object in codebase, not just the crash site.
- External library errors: fix CALLING code, not the library. Read codebase source to understand expected params.

## FIX RULES:
- Fix ONLY the error lines. No try/except. No DummyPipeline. No hardcoded metrics. Keep print/metric statements.
- Verify your fix works for ALL loop entries (not just the first) and doesn't introduce new errors.
"""

_MAX_TODO_ITERATIONS = 10
_MAX_FIX_ATTEMPTS = 2


@dataclass
class OpenHandsBridge:
    """Manages Aider CLI invocations for beast mode code generation.

    Despite the class name (kept for backward compatibility with executor.py),
    this now uses Aider with a TODO-driven loop: generate skeleton with TODO
    markers, then iteratively implement each TODO until none remain.
    """

    model: str = "openai/claude-opus-4-6"
    llm_base_url: str = ""
    api_key_env: str = ""
    api_key: str = ""
    timeout_sec: int = 1200
    max_retries: int = 0

    @staticmethod
    def _find_binary() -> str:
        """Locate the aider binary.

        Search order:
        1. PATH (shutil.which)
        2. Same bin directory as the running Python interpreter
           (handles conda/venv where aider is co-installed)
        3. ~/.local/bin/aider (pip install --user)
        """
        import sys

        for candidate in (
            shutil.which("aider"),
            str(Path(sys.executable).parent / "aider"),
            os.path.expanduser("~/.local/bin/aider"),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate
        return "aider"

    def check_available(self) -> bool:
        try:
            result = subprocess.run(
                [self._find_binary(), "--version"],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    def _prepare_workspace(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        extra_guidance: str,
        time_budget_sec: int,
        codebases_dir: str = "",
        datasets_dir: str = "",
        checkpoints_dir: str = "",
        selected_repos: list[str] | None = None,
    ) -> Path:
        ws = stage_dir / f"aider_beast_{int(time.time())}_{os.getpid()}"
        ws.mkdir(parents=True, exist_ok=True)

        (ws / "EXPERIMENT_PLAN.yaml").write_text(
            exp_plan or "# No experiment plan provided\n", encoding="utf-8",
        )

        guidance_parts = [
            "# Experiment Guidance\n",
            f"## Topic\n{topic}\n",
            f"## Primary Metric\n{metric}\n",
            f"## Time Budget\n{time_budget_sec} seconds\n",
        ]
        if pkg_hint:
            guidance_parts.append(f"## Environment\n{pkg_hint}\n")
        if extra_guidance:
            guidance_parts.append(f"## Additional Guidance\n{extra_guidance}\n")
        (ws / "GUIDANCE.md").write_text(
            "\n".join(guidance_parts), encoding="utf-8",
        )

        if codebases_dir:
            cb_path = Path(codebases_dir).resolve()
            if cb_path.is_dir():
                codebases_ws = ws / "codebases"
                codebases_ws.mkdir(exist_ok=True)
                _ignore = shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc",
                    "node_modules", ".eggs", "_manifest.json",
                )
                dest = codebases_ws / cb_path.name
                shutil.copytree(cb_path, dest, ignore=_ignore)

        if datasets_dir and Path(datasets_dir).is_dir():
            ds_path = Path(datasets_dir).resolve()
            link = ws / "datasets" / ds_path.name
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                link.symlink_to(ds_path)

        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            ck_path = Path(checkpoints_dir).resolve()
            link = ws / "checkpoints" / ck_path.name
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                link.symlink_to(ck_path)

        usage_hints: list[str] = []
        usage_hints.append(
            "## IMPORTANT: Use ABSOLUTE paths in your code\n"
            "The code will be copied to a different directory for execution. "
            "Do NOT rely on relative paths or `__file__`-based resolution.\n"
        )

        def _dir_tree(root: Path, max_depth: int = 3, max_items: int = 15) -> str:
            """Generate a directory tree string showing the actual file layout."""
            lines: list[str] = [f"{root}/"]
            count = 0
            for item in sorted(root.rglob("*")):
                if count >= max_items:
                    lines.append("  ... (truncated)")
                    break
                rel = item.relative_to(root)
                if any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
                    continue
                depth = len(rel.parts)
                if depth > max_depth:
                    continue
                indent = "  " * depth
                suffix = "/" if item.is_dir() else f" ({item.stat().st_size} bytes)" if item.is_file() else ""
                lines.append(f"{indent}{item.name}{suffix}")
                count += 1
            return "\n".join(lines)

        ws_abs = str(ws.resolve())
        if datasets_dir and Path(datasets_dir).is_dir():
            try:
                ds_path = Path(datasets_dir).resolve()
                tree = _dir_tree(ds_path, max_depth=3, max_items=40)
                usage_hints.append(
                    f"## Local Data — ACTUAL directory structure\n"
                    f"Original path: \"{ds_path}\"\n"
                    f"Workspace symlink: \"{ws_abs}/datasets/{ds_path.name}\"\n"
                    f"In code use: `DATASETS_DIR = \"{ds_path}\"`\n"
                    f"```\n{tree}\n```\n"
                )
            except OSError:
                pass
        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            try:
                ck_path = Path(checkpoints_dir).resolve()
                tree = _dir_tree(ck_path, max_depth=2, max_items=30)
                usage_hints.append(
                    f"\n## Local Checkpoints — ACTUAL directory structure\n"
                    f"Original path: \"{ck_path}\"\n"
                    f"Workspace symlink: \"{ws_abs}/checkpoints/{ck_path.name}\"\n"
                    f"In code use: `CHECKPOINTS_DIR = \"{ck_path}\"`\n"
                    f"NEVER use HuggingFace model IDs — load ALL models from this local path.\n"
                    f"```\n{tree}\n```\n"
                )
            except OSError:
                pass
        # When no codebase is provided, add a concise hint with loading example
        if not codebases_dir and checkpoints_dir and Path(checkpoints_dir).is_dir():
            _ck_name = Path(checkpoints_dir).name.lower()
            if any(kw in _ck_name for kw in ("stable-diffusion", "sd-", "sdxl", "diffusion")):
                usage_hints.append(
                    "\n## No codebase — load model from checkpoints:\n"
                    "```python\n"
                    "from diffusers import StableDiffusionPipeline\n"
                    f"pipe = StableDiffusionPipeline.from_pretrained('{checkpoints_dir}', "
                    "local_files_only=True).to('cuda')\n"
                    "```\n"
                )

        if codebases_dir and Path(codebases_dir).is_dir():
            cb_abs = str(Path(codebases_dir).resolve())
            codebases_ws = ws / "codebases"
            repo_names = []
            if codebases_ws.is_dir():
                repo_names = sorted(d.name for d in codebases_ws.iterdir() if d.is_dir())
            hint_lines = [
                f"\n## Local Codebases\n"
                f"Original absolute path: \"{cb_abs}\"\n"
                f"Repos: {', '.join(repo_names)}\n\n"
                "IMPORTANT: In main.py, use the ORIGINAL absolute path for sys.path.insert:\n"
                f"  `sys.path.insert(0, '{cb_abs}')`\n\n"
                "Do NOT use workspace-relative paths — the code will be copied to a sandbox directory.\n"
                "The original path is permanent and always accessible.\n"
            ]
            hint_lines.append(
                "\nThis keeps each codebase's internal imports (e.g. `from utils.utils import ...`) working correctly.\n"
                "Do NOT add subdirectories like `.../freecustom` or `.../utils` to sys.path.\n"
            )

            # Find and include example/demo/entry scripts FIRST (highest priority)
            example_lines: list[str] = []
            _example_budget = 80  # max lines for examples
            for rn in repo_names:
                repo_dir = codebases_ws / rn
                example_patterns = [
                    # Explicit example/demo directories
                    "**/example*/**/*.py", "**/demo*/**/*.py",
                    "**/sample*/**/*.py", "**/tutorial*/**/*.py",
                    # Root-level scripts (common entry points)
                    "run*.py", "main*.py", "train*.py", "infer*.py",
                    "generate*.py", "test_*.py", "predict*.py", "eval*.py",
                    # Named patterns
                    "*example*.py", "*demo*.py", "*inference*.py",
                    # Scripts directory
                    "**/scripts/**/*.py",
                    # Quickstart / getting started
                    "**/quickstart*/**/*.py",
                ]
                seen_examples: set[str] = set()
                for pattern in example_patterns:
                    for ex_file in sorted(repo_dir.glob(pattern)):
                        if _example_budget <= 0:
                            break
                        if ex_file.is_symlink() or not ex_file.is_file():
                            continue
                        rel = str(ex_file.relative_to(codebases_ws))
                        if rel in seen_examples:
                            continue
                        parts = ex_file.relative_to(repo_dir).parts
                        if any(p.startswith("__pycache__") or p.startswith(".") or p == "tests" for p in parts):
                            continue
                        # Skip large files (likely not simple examples)
                        try:
                            if ex_file.stat().st_size > 15000:
                                continue
                        except OSError:
                            continue
                        seen_examples.add(rel)
                        try:
                            content = ex_file.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            continue
                        lines = content.splitlines()
                        if len(lines) > _example_budget:
                            lines = lines[:_example_budget]
                            lines.append("# ... (truncated)")
                        example_lines.append(f"\n**Example: `{rel}`** — KEY code patterns (non-essential lines removed):\n")
                        example_lines.append("```python\n")
                        example_lines.extend(l + "\n" for l in lines)
                        example_lines.append("```\n")
                        _example_budget -= len(lines)

            if example_lines:
                hint_lines.append("\n## Working example scripts from codebase (USE THESE AS REFERENCE)\n")
                hint_lines.append("Copy loading patterns. Adapt HuggingFace IDs to LOCAL CHECKPOINTS_DIR paths.\n\n")
                hint_lines.extend(example_lines)

            # API signatures AFTER examples (lower priority, budget-limited)
            api_lines: list[str] = ["\n## Key API signatures from codebase\n"]
            _api_line_budget = 30
            for rn in repo_names:
                repo_dir = codebases_ws / rn
                py_files = sorted(
                    (f for f in repo_dir.rglob("*.py") if not f.is_symlink()),
                    key=lambda f: len(f.relative_to(repo_dir).parts),
                )
                for py_file in py_files:
                    if _api_line_budget <= 0:
                        break
                    rel = py_file.relative_to(codebases_ws)
                    parts = rel.parts
                    if any(p.startswith("__pycache__") or p.startswith(".") or p == "examples" or p == "tests" for p in parts):
                        continue
                    try:
                        source = py_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    sigs = [
                        line.strip().rstrip(":").rstrip()
                        for line in source.splitlines()
                        if line.strip().startswith("def ") or line.strip().startswith("class ")
                    ]
                    if sigs:
                        api_lines.append(f"\n**`{rel}`**: ")
                        api_lines.append(", ".join(f"`{s}`" for s in sigs[:5]))
                        api_lines.append("\n")
                        _api_line_budget -= 2
                        if _api_line_budget <= 0:
                            api_lines.append("  ... (truncated)\n")
                            break
            if len(api_lines) > 1:
                hint_lines.extend(api_lines)

            usage_hints.append("".join(hint_lines))
        if usage_hints:
            hint_block = "\n\n" + "\n".join(usage_hints) + "\n"
            guidance_path = ws / "GUIDANCE.md"
            existing = guidance_path.read_text(encoding="utf-8")
            guidance_path.write_text(hint_block + "\n" + existing, encoding="utf-8")

        # Trim GUIDANCE.md to avoid bloating the LLM context.
        # Our hints (prepended above) are the most important; the executor-generated
        # content after them can be very long with irrelevant framework docs.
        _MAX_GUIDANCE_LINES = 300
        guidance_path = ws / "GUIDANCE.md"
        if guidance_path.exists():
            g_lines = guidance_path.read_text(encoding="utf-8").splitlines()
            if len(g_lines) > _MAX_GUIDANCE_LINES:
                g_lines = g_lines[:_MAX_GUIDANCE_LINES]
                g_lines.append("\n<!-- GUIDANCE trimmed to fit token budget -->")
                guidance_path.write_text("\n".join(g_lines), encoding="utf-8")

        # Snapshot file hashes for _collect_files filtering
        snapshot: dict[str, str] = {}
        for f in ws.rglob("*.py"):
            if f.is_symlink():
                continue
            try:
                snapshot[str(f.relative_to(ws))] = hashlib.md5(
                    f.read_bytes()
                ).hexdigest()
            except OSError:
                pass
        (ws / ".codebase_snapshot.json").write_text(
            json.dumps(snapshot), encoding="utf-8",
        )

        main_py = ws / "main.py"
        if not main_py.exists():
            main_py.write_text(
                "# main.py — experiment entry point (to be implemented)\n",
                encoding="utf-8",
            )

        return ws

    @staticmethod
    def _collect_files(workspace: Path) -> dict[str, str]:
        """Collect new/modified Python files from the workspace."""
        _snapshot_file = workspace / ".codebase_snapshot.json"
        _original_hashes: dict[str, str] = {}
        if _snapshot_file.exists():
            try:
                _original_hashes = json.loads(
                    _snapshot_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

        files: dict[str, str] = {}
        py_files = sorted(
            workspace.rglob("*.py"),
            key=lambda p: len(p.relative_to(workspace).parts),
        )
        for py_file in py_files:
            if py_file.is_symlink():
                continue
            rel = py_file.relative_to(workspace)
            parts = rel.parts
            if any(p.startswith("__pycache__") or p.startswith(".") or p == "codebases" for p in parts):
                continue
            rel_str = str(rel)
            if rel_str in _original_hashes:
                try:
                    current_hash = hashlib.md5(py_file.read_bytes()).hexdigest()
                except OSError:
                    continue
                if current_hash == _original_hashes[rel_str]:
                    continue
            basename = rel.name
            if basename not in files:
                try:
                    files[basename] = py_file.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as exc:
                    logger.warning("Aider: failed to read %s: %s", py_file, exc)

        for extra in ("requirements.txt", "setup.py"):
            p = workspace / extra
            if p.exists() and extra not in files:
                if extra in _original_hashes:
                    try:
                        cur = hashlib.md5(p.read_bytes()).hexdigest()
                    except OSError:
                        continue
                    if cur == _original_hashes.get(extra):
                        continue
                files[extra] = p.read_text(encoding="utf-8", errors="replace")

        return files

    @staticmethod
    def _find_core_source_files(workspace: Path, max_files: int = 10, max_lines: int = 150) -> list[str]:
        """Find small, core .py files in codebases/ to pass as --read to aider.

        These give aider full visibility into key functions (e.g. mrsa_forward,
        hack_self_attention_to_mrsa) so it can subclass or override them.
        """
        codebases_dir = workspace / "codebases"
        if not codebases_dir.is_dir():
            return []

        candidates: list[tuple[int, Path]] = []
        for py_file in codebases_dir.rglob("*.py"):
            if py_file.is_symlink():
                continue
            parts = py_file.relative_to(codebases_dir).parts
            if any(p.startswith("__pycache__") or p.startswith(".") or p == "tests" or p == "examples" for p in parts):
                continue
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            try:
                line_count = len(py_file.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                continue
            if 10 < line_count <= max_lines:
                candidates.append((line_count, py_file))

        # Sort by line count (smallest first = most focused/core modules)
        candidates.sort(key=lambda x: x[0])
        return [str(p) for _, p in candidates[:max_files]]

    def _build_aider_cmd(
        self,
        workspace: Path,
        message: str,
        api_key: str,
        edit_format: str = "diff",
    ) -> list[str]:
        """Build the aider CLI command for a single invocation."""
        model = self.model
        if "/" not in model:
            model = f"openai/{model}"

        msg_file = workspace / ".aider_task.md"
        msg_file.write_text(message, encoding="utf-8")

        # Editable files: only main.py and GUIDANCE.md (Phase 0 may edit it)
        add_files: list[str] = []
        main_py = workspace / "main.py"
        if main_py.exists():
            add_files.append(str(main_py))
        guidance_path = workspace / "GUIDANCE.md"
        if guidance_path.exists():
            add_files.append(str(guidance_path))

        # Read-only context: EXPERIMENT_PLAN.yaml + codebase source files
        read_files: list[str] = []
        exp_plan_path = workspace / "EXPERIMENT_PLAN.yaml"
        if exp_plan_path.exists():
            read_files.extend(["--read", str(exp_plan_path)])
        for rf in self._find_core_source_files(workspace):
            read_files.extend(["--read", rf])

        return [
            self._find_binary(),
            "--model", model,
            "--openai-api-base", self.llm_base_url,
            "--openai-api-key", api_key,
            "--message-file", str(msg_file),
            "--yes",
            "--no-auto-commits",
            "--no-stream",
            "--no-git",
            "--no-show-model-warnings",
            "--no-show-release-notes",
            "--no-check-update",
            "--no-browser",
            "--edit-format", edit_format,
            "--map-tokens", "2048",
            *read_files,
            *add_files,
        ]

    def _invoke_aider(
        self,
        workspace: Path,
        message: str,
        api_key: str,
        step_timeout: int = 0,
        edit_format: str = "diff",
    ) -> tuple[bool, str, float]:
        """Run a single aider invocation in the workspace."""
        workspace = workspace.resolve()
        env = os.environ.copy()

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(var, None)
        env["no_proxy"] = "*"
        env["NO_PROXY"] = "*"
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        cmd = self._build_aider_cmd(workspace, message, api_key, edit_format)
        timeout = step_timeout or self.timeout_sec

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            elapsed = time.monotonic() - t0
            log = result.stdout + "\n" + result.stderr
            return result.returncode == 0, log, elapsed
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return False, f"Timeout after {elapsed:.1f}s", elapsed
        except FileNotFoundError:
            elapsed = time.monotonic() - t0
            return False, "aider CLI not found", elapsed
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return False, f"Unexpected error: {exc}", elapsed

    @staticmethod
    def _scan_todos(main_py: Path) -> list[str]:
        """Scan main.py for TODO markers. Returns list of TODO lines with context."""
        if not main_py.exists():
            return []
        lines = main_py.read_text(encoding="utf-8").splitlines()
        todos: list[str] = []
        for i, line in enumerate(lines):
            if "# TODO:" in line:
                # Include surrounding context: function def above + the TODO line
                context_start = max(0, i - 3)
                context = "\n".join(
                    f"  L{j+1}: {lines[j]}" for j in range(context_start, min(i + 2, len(lines)))
                )
                todos.append(f"Line {i+1}: {line.strip()}\nContext:\n{context}")
        return todos

    @staticmethod
    def _check_syntax(workspace: Path) -> str | None:
        """Run `python3 -c 'import main'` in the workspace; return error or None."""
        main_py = workspace / "main.py"
        if not main_py.exists():
            return "main.py does not exist"
        try:
            result = subprocess.run(
                ["python3", "-c", "import main"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return (result.stderr or result.stdout)[-1000:]
            return None
        except Exception as exc:
            return str(exc)

    def _resolve_api_key(self) -> str:
        # Prefer explicit api_key from project config (matches the hardcoded
        # base_url for aider), fall back to env var for generic setups.
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def _run_todo_loop(
        self,
        workspace: Path,
        metric: str,
        time_budget_sec: int,
    ) -> tuple[bool, str, float]:
        """Generate code via TODO-driven loop.

        1. Generate skeleton with TODO markers
        2. Repeatedly scan for TODOs and implement one at a time
        3. Syntax-check and fix at the end
        """
        api_key = self._resolve_api_key()
        all_logs: list[str] = []
        total_elapsed = 0.0
        per_call_timeout = max(300, self.timeout_sec // (_MAX_TODO_ITERATIONS + 2))
        main_py = workspace / "main.py"

        _has_codebases = (workspace / "codebases").is_dir() and any((workspace / "codebases").iterdir())
        _extra_rules = "" if _has_codebases else _RULES_NO_CODEBASE

        # --- Phase 1: Generate skeleton ---
        skeleton_prompt = (_SKELETON_PROMPT + _extra_rules).replace(
            "{metric}", metric
        ).replace(
            "{time_budget_sec}", str(time_budget_sec)
        )

        logger.info("Aider TODO loop: generating skeleton...")
        ok, log, elapsed = self._invoke_aider(
            workspace, skeleton_prompt, api_key,
            step_timeout=per_call_timeout,
            edit_format="diff",
        )
        total_elapsed += elapsed
        all_logs.append(f"=== Skeleton (ok={ok}, {elapsed:.1f}s) ===\n{log}")

        if not main_py.exists() or len(main_py.read_text(encoding="utf-8").strip().splitlines()) < 5:
            logger.warning("Aider TODO loop: skeleton generation failed")
            return False, "\n".join(all_logs), total_elapsed

        # --- Phase 2: TODO fill loop ---
        prev_todo_count = -1
        stuck_count = 0
        _MAX_STUCK = 3  # allow a few retries before giving up
        for iteration in range(_MAX_TODO_ITERATIONS):
            todos = self._scan_todos(main_py)

            if not todos:
                logger.info("Aider TODO loop: no TODOs remaining after %d iterations", iteration)
                break

            if len(todos) == prev_todo_count:
                stuck_count += 1
                if stuck_count >= _MAX_STUCK:
                    logger.warning(
                        "Aider TODO loop: TODO count unchanged (%d) for %d iterations — giving up",
                        len(todos), _MAX_STUCK,
                    )
                    break
                logger.info(
                    "Aider TODO loop: TODO count unchanged (%d), retry %d/%d",
                    len(todos), stuck_count, _MAX_STUCK,
                )
            else:
                stuck_count = 0
            prev_todo_count = len(todos)

            first_todo = todos[0]
            fill_prompt = (_FILL_TODO_PROMPT + _extra_rules).replace(
                "{todo_line}", first_todo
            ).replace(
                "{metric}", metric
            ).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )

            logger.info(
                "Aider TODO loop: iteration %d/%d — %d TODOs remaining, implementing first...",
                iteration + 1, _MAX_TODO_ITERATIONS, len(todos),
            )
            ok, log, elapsed = self._invoke_aider(
                workspace, fill_prompt, api_key,
                step_timeout=per_call_timeout,
                edit_format="diff",
            )
            total_elapsed += elapsed
            all_logs.append(
                f"=== TODO iter {iteration+1} ({len(todos)} left, ok={ok}, {elapsed:.1f}s) ===\n{log}"
            )

        # --- Phase 3: Syntax fix ---
        for fix_attempt in range(_MAX_FIX_ATTEMPTS):
            syntax_err = self._check_syntax(workspace)
            if not syntax_err:
                break
            logger.info("Aider TODO loop: fix attempt %d — syntax error detected", fix_attempt + 1)
            fix_prompt = _FIX_PROMPT.replace("{error_output}", syntax_err[:2000])
            ok, log, elapsed = self._invoke_aider(
                workspace, fix_prompt, api_key,
                step_timeout=per_call_timeout,
                edit_format="diff",
            )
            total_elapsed += elapsed
            all_logs.append(f"=== Fix attempt {fix_attempt+1} (ok={ok}, {elapsed:.1f}s) ===\n{log}")

        combined_log = "\n".join(all_logs)
        has_main = (
            main_py.exists()
            and len(main_py.read_text(encoding="utf-8").strip().splitlines()) > 10
        )
        return has_main, combined_log, total_elapsed

    def generate(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str = "",
        extra_guidance: str = "",
        time_budget_sec: int = 3600,
        codebases_dir: str = "",
        datasets_dir: str = "",
        checkpoints_dir: str = "",
        selected_repos: list[str] | None = None,
    ) -> OpenCodeResult:
        """Run Aider to generate experiment code via TODO-driven loop."""
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="Aider CLI not installed or not callable",
            )

        workspace: Path | None = None
        last_error = ""
        last_log = ""
        last_elapsed = 0.0

        for attempt in range(1 + self.max_retries):
            try:
                workspace = self._prepare_workspace(
                    stage_dir=stage_dir,
                    topic=topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint,
                    extra_guidance=extra_guidance,
                    time_budget_sec=time_budget_sec,
                    codebases_dir=codebases_dir,
                    datasets_dir=datasets_dir,
                    checkpoints_dir=checkpoints_dir,
                    selected_repos=selected_repos,
                )
            except OSError as exc:
                last_error = f"Failed to prepare workspace: {exc}"
                logger.warning("Aider beast mode: %s", last_error)
                continue

            logger.info(
                "Aider beast mode (TODO loop): attempt %d/%d",
                attempt + 1, 1 + self.max_retries,
            )

            # --- TODO-driven generation ---
            success, log, elapsed = self._run_todo_loop(
                workspace, metric, time_budget_sec,
            )

            files = self._collect_files(workspace) if workspace.exists() else {}

            if success and "main.py" in files:
                (stage_dir / "aider_log.txt").write_text(log, encoding="utf-8")
                logger.info(
                    "Aider beast mode: SUCCESS (TODO loop) — %d files in %.1fs",
                    len(files), elapsed,
                )
                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            # --- Single-shot fallback ---
            logger.warning(
                "Aider TODO loop produced no valid main.py — trying single-shot fallback"
            )

            api_key = self._resolve_api_key()
            _has_cb = (workspace / "codebases").is_dir() and any((workspace / "codebases").iterdir())
            _fb_extra = "" if _has_cb else _RULES_NO_CODEBASE
            fallback_prompt = (_FALLBACK_PROMPT + _fb_extra).replace(
                "{metric}", metric
            ).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )
            ok, fb_log, fb_elapsed = self._invoke_aider(
                workspace, fallback_prompt, api_key,
                step_timeout=self.timeout_sec,
                edit_format="whole",
            )
            elapsed += fb_elapsed
            log += "\n=== Single-shot fallback ===\n" + fb_log

            files = self._collect_files(workspace) if workspace.exists() else {}

            if "main.py" in files and len(files["main.py"].strip().splitlines()) > 10:
                (stage_dir / "aider_log.txt").write_text(log, encoding="utf-8")
                logger.info(
                    "Aider beast mode: SUCCESS (fallback) — %d files in %.1fs",
                    len(files), elapsed,
                )
                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = log[:500] if log else "No main.py produced"
            last_log = log
            last_elapsed = elapsed
            logger.info(
                "Aider beast mode: attempt %d failed (%.1fs, files=%s): %s",
                attempt + 1, elapsed, list(files.keys()), last_error[:200],
            )

        # Persist log even on failure so the error can be diagnosed
        if last_log:
            try:
                (stage_dir / "aider_log.txt").write_text(
                    last_log, encoding="utf-8",
                )
            except OSError:
                pass

        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            elapsed_sec=last_elapsed,
            error=f"Aider failed after {1 + self.max_retries} attempt(s)",
        )

    # ------------------------------------------------------------------
    # Stage 12: Aider-based sanity check fix
    # ------------------------------------------------------------------

    def _prepare_fix_workspace(
        self,
        stage_dir: Path,
        run_dir: Path,
        experiment_dir: Path,
        codebases_dir: str = "",
    ) -> Path:
        """Prepare a workspace for Aider-based sanity fix.

        Copies experiment .py files, reuses GUIDANCE.md and EXPERIMENT_PLAN.yaml
        from the Stage 11 aider workspace, and symlinks codebases for --read.
        """
        ws = stage_dir / f"aider_fix_{int(time.time())}_{os.getpid()}"
        ws.mkdir(parents=True, exist_ok=True)

        for py_file in sorted(experiment_dir.glob("*.py")):
            try:
                shutil.copy2(py_file, ws / py_file.name)
            except OSError:
                pass

        for aider_ws in sorted(run_dir.glob("stage-*/aider_beast_*"), reverse=True):
            for ctx_name in ("GUIDANCE.md", "EXPERIMENT_PLAN.yaml"):
                src = aider_ws / ctx_name
                dst = ws / ctx_name
                if src.exists() and not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                    except OSError:
                        pass
            cb_src = aider_ws / "codebases"
            cb_dst = ws / "codebases"
            if cb_src.is_dir() and not cb_dst.exists():
                try:
                    cb_dst.symlink_to(cb_src.resolve())
                except OSError:
                    pass
            if (ws / "GUIDANCE.md").exists():
                break

        if codebases_dir and not (ws / "codebases").exists():
            cb_path = Path(codebases_dir).resolve()
            if cb_path.is_dir():
                link = ws / "codebases"
                try:
                    link.symlink_to(cb_path)
                except OSError:
                    pass

        # Copy dataset config YAMLs as read-only context so the model can
        # inspect actual config keys (e.g. ref_image_infos path strings).
        _datasets_dir = ""
        _gm = ws / "GUIDANCE.md"
        if _gm.exists():
            import re as _re_ws
            _gm_text = _gm.read_text(encoding="utf-8")
            _match = _re_ws.search(r'DATASETS_DIR\s*=\s*["\']([^"\']+)["\']', _gm_text)
            if _match:
                _datasets_dir = _match.group(1)
        if _datasets_dir:
            _ds_path = Path(_datasets_dir)
            _ctx_dir = ws / "dataset_configs"
            _copied = 0
            if _ds_path.is_dir():
                for _cfg_yaml in sorted(_ds_path.rglob("*.yaml")):
                    if _copied >= 10:
                        break
                    try:
                        _rel = _cfg_yaml.relative_to(_ds_path)
                        _dst = _ctx_dir / _rel
                        _dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(_cfg_yaml, _dst)
                        _copied += 1
                    except (OSError, ValueError):
                        pass

        # Copy codebase config YAMLs (e.g. config_for_visualization.yaml)
        if codebases_dir:
            _cb_configs = Path(codebases_dir) / "configs"
            _cb_ctx = ws / "codebase_configs"
            if _cb_configs.is_dir():
                _copied = 0
                for _cfg_yaml in sorted(_cb_configs.glob("*.yaml")):
                    if _copied >= 5:
                        break
                    try:
                        _cb_ctx.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(_cfg_yaml, _cb_ctx / _cfg_yaml.name)
                        _copied += 1
                    except OSError:
                        pass

        return ws

    def _build_aider_fix_cmd(
        self,
        workspace: Path,
        message: str,
        api_key: str,
    ) -> list[str]:
        """Build Aider CLI command for a sanity fix invocation.

        All .py files in workspace are editable; GUIDANCE.md, EXPERIMENT_PLAN.yaml,
        and codebase .py files are read-only context.
        """
        model = self.model
        if "/" not in model:
            model = f"openai/{model}"

        msg_file = workspace / ".aider_task.md"
        msg_file.write_text(message, encoding="utf-8")

        add_files: list[str] = []
        for py_file in sorted(workspace.glob("*.py")):
            add_files.append(str(py_file))

        read_files: list[str] = []
        for ctx in ("GUIDANCE.md", "EXPERIMENT_PLAN.yaml"):
            ctx_path = workspace / ctx
            if ctx_path.exists():
                read_files.extend(["--read", str(ctx_path)])
        for rf in self._find_core_source_files(workspace):
            read_files.extend(["--read", rf])
        # Include dataset config YAMLs and codebase config YAMLs as read context
        for _ctx_dir_name in ("dataset_configs", "codebase_configs"):
            _ctx_dir = workspace / _ctx_dir_name
            if _ctx_dir.is_dir():
                for _yaml_file in sorted(_ctx_dir.rglob("*.yaml"))[:8]:
                    read_files.extend(["--read", str(_yaml_file)])

        return [
            self._find_binary(),
            "--model", model,
            "--openai-api-base", self.llm_base_url,
            "--openai-api-key", api_key,
            "--message-file", str(msg_file),
            "--yes",
            "--no-auto-commits",
            "--no-stream",
            "--no-git",
            "--no-show-model-warnings",
            "--no-show-release-notes",
            "--no-check-update",
            "--no-browser",
            "--edit-format", "diff",
            "--map-tokens", "2048",
            *read_files,
            *add_files,
        ]

    def fix_sanity_error(
        self,
        stage_dir: Path,
        run_dir: Path,
        experiment_dir: Path,
        test_name: str,
        test_code: str,
        stderr: str,
        iteration: int,
        max_iterations: int,
        previous_fixes: list[dict[str, Any]] | None = None,
        codebases_dir: str = "",
    ) -> tuple[bool, dict[str, str], str]:
        """Use Aider to fix a sanity check failure.

        Returns (success, {filename: patched_content}, aider_log).
        The caller is responsible for writing patched files back to experiment_dir.
        """
        if not self.check_available():
            return False, {}, "Aider CLI not available"

        api_key = self._resolve_api_key()

        ws = self._prepare_fix_workspace(
            stage_dir=stage_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
            codebases_dir=codebases_dir,
        )

        repeat_hint = ""
        if previous_fixes:
            prev_details = []
            for entry in previous_fixes[-3:]:
                err_tail = str(entry.get("error_tail", ""))[-500:]
                last_line = err_tail.strip().splitlines()[-1] if err_tail.strip() else "?"
                diff_stats = entry.get("patch_diff_stats", {})
                diff_summary = ""
                if diff_stats:
                    diff_parts = [
                        f"{fname}: {s.get('changed_lines', '?')} lines changed ({s.get('change_pct', '?')}%)"
                        for fname, s in diff_stats.items()
                    ]
                    diff_summary = f"\n  Changes made: {', '.join(diff_parts)}"
                prev_details.append(
                    f"- **Iteration {entry.get('iteration', '?')}**: "
                    f"patched {entry.get('patches_applied', [])}, "
                    f"failed test `{entry.get('failed_test', '?')}`{diff_summary}\n"
                    f"  Error: `{last_line}`"
                )

            last_err = str(previous_fixes[-1].get("error_tail", ""))[-300:].strip()
            current_err = stderr[-300:].strip()
            if len(previous_fixes) >= 2:
                prev_prev_err = str(previous_fixes[-2].get("error_tail", ""))[-300:].strip()
            else:
                prev_prev_err = ""

            same_as_last = (
                last_err and current_err
                and last_err.splitlines()[-1:] == current_err.splitlines()[-1:]
            )
            is_cycle = (
                prev_prev_err and current_err
                and len(previous_fixes) >= 2
                and prev_prev_err.splitlines()[-1:] == current_err.splitlines()[-1:]
            )

            if is_cycle:
                escalation = (
                    "**ESCALATION — CYCLE DETECTED**: The same error appeared before, was 'fixed', "
                    "then a different fix broke it again the same way. Your previous approach is "
                    "FUNDAMENTALLY WRONG. You must try a COMPLETELY DIFFERENT strategy:\n"
                    "- For path errors: instead of manipulating the path string, use `os.path.basename()` "
                    "to extract just the filename and rebuild the path from known constants.\n"
                    "- For NoneType errors: read the reference implementation (inference.py) to find "
                    "the EXACT correct values, don't guess.\n"
                    "- For missing keys: read the ACTUAL config file or grep ALL attribute accesses "
                    "in the codebase source.\n"
                )
            elif same_as_last:
                escalation = (
                    "**WARNING — SAME ERROR**: The error is IDENTICAL to the previous iteration. "
                    "Your last fix had NO EFFECT on this error. The previous change was either "
                    "wrong or insufficient. Do NOT repeat the same approach — try something different.\n"
                    "Read the actual data/config files to understand what values are really there.\n"
                )
            else:
                escalation = (
                    "**NOTE**: The error changed from the previous iteration — your fix partially worked "
                    "but exposed a new issue. Fix this new error while keeping the previous fix intact.\n"
                )

            repeat_hint = (
                "\n## Previous fix attempts (all FAILED)\n"
                + "\n".join(prev_details) + "\n\n"
                + escalation
            )

        fix_prompt = _FIX_SANITY_PROMPT.replace(
            "{test_name}", test_name
        ).replace(
            "{test_code}", test_code
        ).replace(
            "{stderr}", stderr[-3000:]
        ).replace(
            "{repeat_hint}", repeat_hint
        )

        logger.info(
            "Aider sanity fix: invoking for test=%s iteration=%d workspace=%s",
            test_name, iteration, ws,
        )

        ok, log, elapsed = self._invoke_aider(
            workspace=ws,
            message=fix_prompt,
            api_key=api_key,
            step_timeout=max(180, self.timeout_sec // 3),
            edit_format="diff",
        )

        logger.info(
            "Aider sanity fix: done (ok=%s, %.1fs)", ok, elapsed,
        )

        patched_files: dict[str, str] = {}
        for py_file in sorted(ws.glob("*.py")):
            orig = experiment_dir / py_file.name
            if not orig.exists():
                continue
            try:
                ws_content = py_file.read_text(encoding="utf-8")
                orig_content = orig.read_text(encoding="utf-8")
                if ws_content != orig_content and len(ws_content.strip()) > 30:
                    patched_files[py_file.name] = ws_content
            except OSError:
                pass

        return bool(patched_files), patched_files, log
