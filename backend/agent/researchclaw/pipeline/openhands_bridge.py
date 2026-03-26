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
CRITICAL: You MUST read EXPERIMENT_PLAN.yaml FIRST before writing any code.
The plan defines the exact method names, algorithm steps, baselines, and metrics.

Context files in this workspace:
- EXPERIMENT_PLAN.yaml — REQUIRED READING: defines methods, baselines, ablations, metrics, dataset protocol
- GUIDANCE.md — topic, environment, ABSOLUTE paths for data/checkpoints/codebases
- Source code .py files — the codebase you MUST build upon (read them to learn the API)
- data/ and checkpoints/ — symlinks to local datasets and model weights

RULES:
- Read EXPERIMENT_PLAN.yaml to find: (a) baseline class_name and algorithm_steps, (b) condition names and differentiators, (c) metric name/definition/direction, (d) dataset protocol
- Build ON TOP of the existing codebase — read .py files first to learn the API
- Use ABSOLUTE paths from GUIDANCE.md for datasets, checkpoints, codebases
- NEVER use synthetic data (torch.randn, np.random) as metric placeholders
- NEVER invent module names — only import modules that ACTUALLY EXIST in the workspace
- NEVER hardcode fake metric values like 0.5 or 0.75
- ABSOLUTELY NO try/except blocks — let errors crash with full traceback
- Do NOT use argparse — hardcode all configuration
- Each condition must implement a genuinely DIFFERENT algorithm (not just different hyperparameters)
- Do NOT create useless wrapper classes with empty methods — use plain functions
- Before loading models, LIST files in CHECKPOINTS_DIR to find correct filenames
- NEVER download models from the internet. No HuggingFace IDs, no `pretrained=True`, no URLs. ALL models/weights are pre-downloaded in CHECKPOINTS_DIR. Always load from LOCAL ABSOLUTE PATH: `from_pretrained(os.path.join(CHECKPOINTS_DIR, 'model-name'))`. Check the CHECKPOINTS directory tree in GUIDANCE.md to find the exact folder names available
- The condition_name parameter MUST control which code path runs
- When calling codebase functions, READ the function body (not just signature) to understand what it expects. If a function accesses `model.unet` internally, pass the PIPELINE object, not `pipeline.unet`

- When adding codebase paths via sys.path.insert, add the REPOSITORY ROOT (e.g. `/path/to/FreeCustom`), NOT subdirectories like `/path/to/FreeCustom/utils`. Adding subdirectories breaks internal relative imports within the codebase
- NEVER create files named `utils.py`, `models.py`, `config.py`, or any name that shadows modules inside the codebase — this causes import conflicts. If you need helper code, put it directly in main.py

ANTI-PATTERNS (your code MUST NOT contain these):
  BAD: `primary_metric = 0.75 if hasattr(output, 'videos') else 0.5`
  BAD: `except Exception as e: primary_metric = 0.0`
  BAD: `def run_condition(... condition_name ...):` that ignores condition_name
  BAD: `next(Path(dir).glob('*.safetensors'))` without checking if files exist
  BAD: `def report_metric(self, ...): pass`
  BAD: loading a directory path as if it were a file
  BAD: `sys.path.insert(0, '/path/to/repo/utils')` — add the repo root instead
  BAD: creating `utils.py` in workspace when codebase already has a `utils/` package
  BAD: `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` — downloads from internet!
  BAD: `some_model(pretrained=True)` — downloads from internet!
  BAD: `torch.hub.load(...)` — downloads from internet!
  GOOD: `CLIPModel.from_pretrained(os.path.join(CHECKPOINTS_DIR, "clip-vit-base-patch32"))` — loads local
  BAD: `except Exception: return DummyPipeline()` — silent fallback hides real errors!
  BAD: `_ = pipe(...)` then compute metrics from formulas — pipeline output MUST be used for metrics
  BAD: `class DummyPipeline` / `class FallbackPipeline` — NEVER create fake pipeline substitutes
  BAD: computing metrics from `_condition_profile` dicts or math formulas instead of real model outputs

FAIL-FAST RULES (critical):
- If pipeline loading fails, the program MUST crash. Do NOT catch the exception and substitute a dummy.
- If model inference fails, the program MUST crash. Do NOT return fake outputs.
- Metric values MUST be computed from ACTUAL model outputs (generated images/videos). NEVER compute metrics from hardcoded profiles, math formulas, prompt distances, or hash-based noise.
- The pipeline output MUST be used in compute_metric(). If you assign it to `_` or ignore it, the experiment is INVALID.
- NEVER wrap load_pipeline() or pipe(...) in try/except with a fallback. Let errors propagate so the sanity check system can diagnose and fix them.
"""

_SKELETON_PROMPT = _RULES + """
TASK: Generate a SHORT main.py SKELETON. This is ONLY a skeleton — do NOT implement any logic.

WARNING: Your output token budget is LIMITED. Keep the skeleton under 120 lines.
Do NOT write implementation code. Use `pass` for ALL function bodies that need implementation.

1. Read EXPERIMENT_PLAN.yaml. Pick 1 baseline + 2 proposed methods (3 total).

2. Create main.py with:
   - Imports + sys.path.insert (add REPO ROOT only, not subdirectories)
   - Constants: DATASETS_DIR, CHECKPOINTS_DIR, TIME_BUDGET={time_budget_sec}, SEEDS=[42,123,456]
   - set_seed(seed) — IMPLEMENT this (3 lines)
   - should_stop() — IMPLEMENT this (2 lines)
   - load_pipeline() — write `# TODO: <what to load and how>` then `pass`
   - load_data() — write `# TODO: <what data to load>` then `pass`
   - compute_metric(generated, reference) — write `# TODO: <exact metric name and formula>` then `pass`
   - 3 condition functions (use REAL names from plan) — each gets `# TODO: <algorithm steps>` then `pass`
   - run_condition() — IMPLEMENT this (dispatch dict + call compute_metric)
   - main() — IMPLEMENT this (loop over conditions/seeds, print results)
   - if __name__ == '__main__': main()

CRITICAL RULES:
- Functions with TODO MUST have `pass` as body. Do NOT write any implementation.
- Do NOT use classes. Use plain functions only.
- Do NOT write more than 120 lines total. This is a SKELETON.
- The TODO comment must describe WHAT to implement (refer to EXPERIMENT_PLAN.yaml specifics).
"""

_FILL_TODO_PROMPT = _RULES + """
TASK: Implement ONE specific TODO in main.py. Here is the TODO to implement:

{todo_line}

Instructions:
- Read the workspace source code (.py files) to learn the codebase API before implementing
- Read EXPERIMENT_PLAN.yaml for the algorithm steps and metric definitions
- Implement ONLY this one function — do NOT modify other functions
- Remove the `# TODO:` comment line when the implementation is complete
- Replace `pass` with the actual implementation
- Keep the output SHORT — implement just this one function
- Use ABSOLUTE paths from the constants defined at the top of main.py
"""

_FIX_PROMPT = """\
The file main.py has a syntax error or import error:

{error_output}

Fix the error. Make the MINIMAL change needed. Do NOT rewrite the entire file.
"""

# Single-shot fallback
_FALLBACK_PROMPT = _RULES + """
Read EXPERIMENT_PLAN.yaml and GUIDANCE.md, then create a COMPLETE main.py.
Use the EXACT method names, algorithm steps, and metric definitions from EXPERIMENT_PLAN.yaml.
- All imports, constants, def main(), if __name__ == "__main__": main()
- 3 conditions (1 baseline + 2 proposed) with genuinely different algorithms
- compute_metric must compute the REAL metric, not hardcoded values
- Print: {metric}: <value> for each condition and seed
- Time budget: {time_budget_sec} seconds
- NO try/except, NO fake metrics, NO identical conditions
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
        """Locate the aider binary."""
        for candidate in (
            shutil.which("aider"),
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
            _example_budget = 120  # max lines for examples
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
                hint_lines.append("These examples show the CORRECT way to load models and call the pipeline.\n")
                hint_lines.append("Copy the loading pattern — do NOT guess the API.\n")
                hint_lines.append(
                    "\n**IMPORTANT**: If the examples use HuggingFace model IDs (e.g. `model_id='org/model'`), "
                    "you MUST adapt them to load from LOCAL paths instead. Check the model config class "
                    "for a `path=` parameter that accepts local file paths directly. "
                    "Use the file paths from the CHECKPOINTS directory tree above.\n\n"
                )
                hint_lines.extend(example_lines)

            # API signatures AFTER examples (lower priority, budget-limited)
            api_lines: list[str] = ["\n## Key API signatures from codebase\n"]
            _api_line_budget = 60
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
        _MAX_GUIDANCE_LINES = 400
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

        # Editable files
        add_files: list[str] = []
        main_py = workspace / "main.py"
        if main_py.exists():
            add_files.append(str(main_py))
        for ctx in ("EXPERIMENT_PLAN.yaml", "GUIDANCE.md"):
            ctx_path = workspace / ctx
            if ctx_path.exists():
                add_files.append(str(ctx_path))

        # Read-only codebase files: aider can see full source but won't edit them
        read_files: list[str] = []
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
        api_key = ""
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env, "")
        if not api_key and self.api_key:
            api_key = self.api_key
        return api_key

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

        # --- Phase 0: Analyze workspace and write concise reference into GUIDANCE.md ---
        _has_codebases = (workspace / "codebases").is_dir() and any((workspace / "codebases").iterdir())
        _has_data = (workspace / "datasets").is_dir() or (workspace / "checkpoints").is_dir()
        if _has_codebases or _has_data:
            analyze_parts = [
                "Read GUIDANCE.md and ALL read-only source files. "
                "Then REPLACE the first sections of GUIDANCE.md (everything before '## Topic') with a single concise section "
                "titled '## Workspace Quick Reference (LLM-generated)'. This section must contain:\n\n"
            ]
            if _has_data:
                analyze_parts.append(
                    "### Data & Checkpoints\n"
                    "- Summarize the dataset structure in 3-5 lines (what directories exist, file types, how to iterate episodes/samples)\n"
                    "- Summarize checkpoint structure in 3-5 lines (what model files exist, exact filenames, how to load them)\n"
                    "- Give the EXACT Python code (3-5 lines) to load data and the EXACT code to load checkpoints from local paths\n\n"
                )
            if _has_codebases:
                analyze_parts.append(
                    "### Codebase API\n"
                    "- The EXACT code to load the pipeline/model (copy from example scripts, adapt to use local CHECKPOINTS_DIR paths)\n"
                    "- The EXACT code to prepare input data (images, masks, latents, prompts) with correct shapes noted in comments\n"
                    "- The EXACT constructor call for key classes with ALL required parameters\n"
                    "- The EXACT function call pattern for model inference / attention hacking\n"
                    "- Important gotchas (e.g. pass pipeline not pipeline.unet, required non-None params, tensor shape requirements)\n"
                    "- If conditions need to modify internal model behavior: show HOW to subclass or override the key method\n\n"
                )
            analyze_parts.append(
                "Keep the ENTIRE section under 80 lines. Use real code snippets, not descriptions. "
                "This reference will be used by another LLM to write experiment code.\n"
                "IMPORTANT: Preserve '## Topic', '## Primary Metric', '## Time Budget', '## Environment' sections unchanged."
            )
            analyze_prompt = "".join(analyze_parts)

            logger.info("Aider TODO loop: analyzing codebase...")
            ok, log, elapsed = self._invoke_aider(
                workspace, analyze_prompt, api_key,
                step_timeout=per_call_timeout,
                edit_format="diff",
            )
            total_elapsed += elapsed
            all_logs.append(f"=== Codebase analysis (ok={ok}, {elapsed:.1f}s) ===\n{log}")

        # --- Phase 1: Generate skeleton ---
        skeleton_prompt = _SKELETON_PROMPT.replace(
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
            fill_prompt = _FILL_TODO_PROMPT.replace(
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
        time_budget_sec: int = 300,
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
            fallback_prompt = _FALLBACK_PROMPT.replace(
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
            logger.info(
                "Aider beast mode: attempt %d failed (%.1fs, files=%s): %s",
                attempt + 1, elapsed, list(files.keys()), last_error[:200],
            )

        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            elapsed_sec=0.0,
            error=f"Aider failed after {1 + self.max_retries} attempt(s)",
        )
