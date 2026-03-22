"""OpenHands Beast Mode bridge for experiment code generation.

Replaces OpenCode with OpenHands CLI (headless mode) for generating
experiment code. Uses litellm streaming internally, avoiding the 60s
proxy timeout issue that affects OpenCode with certain API providers.

Re-exports complexity scoring and result types from opencode_bridge
so executor.py can import from either module interchangeably.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
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

_TASK_TEMPLATE = """\
You are implementing a complete, runnable ML/science experiment.

STEP 1 — READ THE WORKSPACE:
The workspace contains the COMPLETE source code of a local codebase plus context files:
- EXPERIMENT_PLAN.yaml — the experiment design
- GUIDANCE.md — topic, metric, environment, constraints
- Source code files — the actual codebase you MUST build upon
- `data/` and `checkpoints/` — symlinks to local datasets and model weights
- GUIDANCE.md lists the ABSOLUTE paths for datasets, checkpoints, and codebases

Before writing ANY code, read the existing source files to understand the codebase:
1. Find and read any example/demo scripts (e.g. *_demo.py, run_*.py, example_*.py) to see how the pipeline works end-to-end.
2. Read the core modules to understand the API (class signatures, function arguments).
3. Read the dataset configs or sample data in data/ to understand the data format.
4. Check what checkpoints/weights are available in checkpoints/.

STEP 2 — IMPLEMENTATION RULES:
- Build ON TOP of the existing codebase. Import and extend existing modules.
- Load pretrained models using the ABSOLUTE paths from GUIDANCE.md, NOT from the internet.
- Load real data using the ABSOLUTE paths from GUIDANCE.md, NOT synthetic torch.randn().
- Compute REAL metrics from actual model outputs. NEVER use np.random or random.uniform as a metric placeholder.
- Each experimental condition must produce genuinely different behavior.
- Do NOT rewrite modules that already exist — import and extend them.
- NEVER invent module names — only use modules visible in the workspace.

STEP 3 — CREATE main.py:
1. main.py MUST have a `def main():` function AND `if __name__ == "__main__": main()` at the bottom.
2. main() must ACTUALLY CALL the pipeline to generate images and compute metrics — not just define functions.
3. It must print the primary metric as: {metric}: <value>
4. Use multi-seed evaluation (seeds 0, 1, 2) and report mean +/- std.
5. Implement a time guard: stop gracefully at 80% of the time budget ({time_budget_sec} seconds).
6. Print per-condition results: condition=<name> seed=<s> {metric}: <value>

CRITICAL REQUIREMENTS:
- main.py MUST be a runnable script, NOT a library of functions. `python main.py` must produce output.
- ABSOLUTELY NO try/except blocks anywhere in the code. If ANY operation fails (import, model loading, data loading, pipeline call, metric computation), the program MUST crash with a full traceback. Do NOT catch exceptions to print error messages and continue — this hides bugs and produces empty metrics. The sanity check system will detect and fix crashes automatically, but it CANNOT fix silently swallowed errors.
- Do NOT use argparse or CLI arguments — hardcode all configuration.
- All output must go to stdout (print statements).
- Keep the experiment feasible within {time_budget_sec} seconds total.
"""


@dataclass
class OpenHandsBridge:
    """Manages OpenHands CLI invocations for beast mode code generation."""

    model: str = "openai/claude-opus-4-6"
    llm_base_url: str = ""
    api_key_env: str = ""
    api_key: str = ""
    timeout_sec: int = 1200
    max_retries: int = 0

    @staticmethod
    def _find_binary() -> str:
        """Locate the openhands binary."""
        for candidate in (
            shutil.which("openhands"),
            os.path.expanduser("~/.local/bin/openhands"),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate
        return "openhands"

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
    ) -> Path:
        ws = stage_dir / f"openhands_beast_{int(time.time())}_{os.getpid()}"
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
            cb_path = Path(codebases_dir)
            if cb_path.is_dir():
                for repo in sorted(cb_path.iterdir()):
                    if repo.is_dir() and not repo.name.startswith("."):
                        shutil.copytree(
                            repo, ws,
                            ignore=shutil.ignore_patterns(
                                ".git", "__pycache__", "*.pyc",
                                "node_modules", ".eggs", "_manifest.json",
                            ),
                            dirs_exist_ok=True,
                        )

        if datasets_dir and Path(datasets_dir).is_dir():
            link = ws / "data"
            if not link.exists():
                link.symlink_to(Path(datasets_dir).resolve())

        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            link = ws / "checkpoints"
            if not link.exists():
                link.symlink_to(Path(checkpoints_dir).resolve())

        # Append data/checkpoint layout hints to GUIDANCE.md
        usage_hints: list[str] = []
        usage_hints.append(
            "## IMPORTANT: Use ABSOLUTE paths in your code\n"
            "The code will be copied to a different directory for execution. "
            "Do NOT rely on relative paths or `__file__`-based resolution.\n"
        )
        if datasets_dir and Path(datasets_dir).is_dir():
            try:
                ds_abs = str(Path(datasets_dir).resolve())
                ds_items = sorted(
                    d.name for d in Path(ds_abs).iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                )
                usage_hints.append(
                    f"## Local Data\n"
                    f"DATASETS_DIR = \"{ds_abs}\"\n"
                    f"Available datasets: {', '.join(ds_items)}\n"
                )
                for ds_name in ds_items[:3]:
                    ds_sub = Path(ds_abs) / ds_name
                    sub_items = sorted(d.name for d in ds_sub.rglob("*") if d.is_file())[:10]
                    if sub_items:
                        usage_hints.append(f"- `{ds_abs}/{ds_name}/`: {', '.join(sub_items)}")
            except OSError:
                pass
        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            try:
                ck_abs = str(Path(checkpoints_dir).resolve())
                ck_items = sorted(
                    d.name for d in Path(ck_abs).iterdir() if not d.name.startswith(".")
                )
                ck_hint = (
                    f"\n## Local Checkpoints\n"
                    f"CHECKPOINTS_DIR = \"{ck_abs}\"\n"
                    f"Available: {', '.join(ck_items)}\n"
                    "Load pretrained models from this absolute path.\n"
                )
                if ck_items:
                    ck_hint += f"Example: `model.from_pretrained('{ck_abs}/{ck_items[0]}')`"
                usage_hints.append(ck_hint)
            except OSError:
                pass
        if codebases_dir and Path(codebases_dir).is_dir():
            cb_abs = str(Path(codebases_dir).resolve())
            usage_hints.append(
                f"\n## Local Codebases\n"
                f"CODEBASES_DIR = \"{cb_abs}\"\n"
                "The codebase source is already in the workspace. "
                f"In main.py, add: `sys.path.insert(0, '{cb_abs}/<repo_name>')` "
                "so imports work regardless of working directory.\n"
            )
        if usage_hints:
            with open(ws / "GUIDANCE.md", "a", encoding="utf-8") as f:
                f.write("\n\n" + "\n".join(usage_hints) + "\n")

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
            if any(p.startswith("__pycache__") or p.startswith(".") for p in parts):
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
                    logger.warning("OpenHands: failed to read %s: %s", py_file, exc)

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

    def _invoke_openhands(
        self,
        workspace: Path,
        task: str,
    ) -> tuple[bool, str, float]:
        """Run openhands --headless in the workspace."""
        workspace = workspace.resolve()
        env = os.environ.copy()

        api_key = env.get(self.api_key_env, "") if self.api_key_env else ""
        if not api_key and self.api_key:
            api_key = self.api_key

        env["LLM_MODEL"] = self.model
        if api_key:
            env["LLM_API_KEY"] = api_key
        if self.llm_base_url:
            env["LLM_BASE_URL"] = self.llm_base_url

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(var, None)
        env["no_proxy"] = "*"
        env["NO_PROXY"] = "*"

        task_final = (
            task
            + "\n\nProceed immediately. Do NOT ask for confirmation — implement everything now."
            "\nDo NOT run or execute the code. Just write all the files and finish."
        )

        cmd = [
            self._find_binary(),
            "--headless",
            "--always-approve",
            "--override-with-envs",
            "-t", task_final,
        ]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
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
            return False, "openhands CLI not found", elapsed
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return False, f"Unexpected error: {exc}", elapsed

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
    ) -> OpenCodeResult:
        """Run OpenHands to generate experiment code."""
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="OpenHands CLI not installed or not callable",
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
                )
            except OSError as exc:
                last_error = f"Failed to prepare workspace: {exc}"
                logger.warning("OpenHands beast mode: %s", last_error)
                continue

            task = _TASK_TEMPLATE.replace("{metric}", metric).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )

            logger.info(
                "OpenHands beast mode: attempt %d/%d (timeout=%ds)",
                attempt + 1, 1 + self.max_retries, self.timeout_sec,
            )

            success, log, elapsed = self._invoke_openhands(workspace, task)

            files = self._collect_files(workspace) if workspace.exists() else {}

            if "main.py" in files:
                if not success:
                    logger.info(
                        "OpenHands: exited non-zero but main.py found — treating as success"
                    )

                (stage_dir / "openhands_log.txt").write_text(
                    log, encoding="utf-8",
                )

                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = log[:500] if log else "No main.py produced"
            logger.info(
                "OpenHands beast mode: attempt %d failed (%.1fs, files=%s): %s",
                attempt + 1, elapsed, list(files.keys()), last_error[:200],
            )

        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            elapsed_sec=0.0,
            error=f"OpenHands failed after {1 + self.max_retries} attempt(s)",
        )
