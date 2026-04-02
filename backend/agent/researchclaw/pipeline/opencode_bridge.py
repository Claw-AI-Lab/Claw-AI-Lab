"""OpenCode 'Beast Mode' bridge — routes complex code generation to OpenCode CLI.

OpenCode (https://github.com/anomalyco/opencode) is an external AI coding agent
invoked via ``opencode run --format json "prompt"``.  This module provides:

1. **ComplexityScore / score_complexity()** — analyses an experiment plan to
   decide whether beast mode is warranted.
2. **OpenCodeBridge** — manages workspace creation, OpenCode invocation, file
   collection, and cleanup.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

# Keywords that indicate multi-component architectures
_COMPONENT_KEYWORDS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "discriminator",
    "generator",
    "critic",
    "actor",
    "teacher",
    "student",
    "backbone",
    "head",
    "neck",
    "classifier",
    "embedder",
    "attention",
    "transformer",
    "tokenizer",
    "vae",
    "autoencoder",
)

# Indicators that multi-file generation is needed
_FILE_HINT_KEYWORDS: tuple[str, ...] = (
    "model.py",
    "trainer.py",
    "dataset.py",
    "utils.py",
    "config.py",
    "multiple files",
    "modular",
    "separate module",
    "multi-file",
)

# Domain-complexity keywords
_DOMAIN_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "multi-modal",
    "multimodal",
    "distributed",
    "gan",
    "diffusion",
    "nerf",
    "mixture of experts",
    "moe",
    "meta-learning",
    "meta learning",
    "maml",
    "neural ode",
    "neural sde",
    "physics-informed",
    "pinn",
    "graph neural",
    "gnn",
    "reinforcement learning",
    "multi-agent",
    "world model",
    "vision-language",
    "text-to-image",
    "image-to-text",
)

# Patterns suggesting deep dependency chains
_DEPENDENCY_KEYWORDS: tuple[str, ...] = (
    "custom layer",
    "custom loss",
    "wrapper",
    "registry",
    "hook",
    "callback",
    "scheduler",
    "custom optimizer",
    "custom dataset",
    "custom sampler",
    "custom transform",
)


@dataclass
class ComplexityScore:
    """Result of complexity analysis on an experiment plan."""

    score: float  # 0.0-1.0
    signals: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""  # "beast_mode" | "code_agent" | "legacy"
    reason: str = ""


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_complexity(
    exp_plan: str,
    topic: str = "",
    *,
    historical_failures: int = 0,
    threshold: float = 0.6,
) -> ComplexityScore:
    """Score the complexity of an experiment to determine if beast mode is warranted.

    Returns a ComplexityScore with score in [0.0, 1.0].
    """
    if not exp_plan and not topic:
        return ComplexityScore(
            score=0.0,
            signals={},
            recommendation="legacy",
            reason="Empty plan",
        )

    combined = f"{topic}\n{exp_plan}"

    # Signal 1: Component count (weight 0.25)
    comp_hits = _count_keyword_hits(combined, _COMPONENT_KEYWORDS)
    component_score = min(comp_hits / 5.0, 1.0)

    # Signal 2: File count hint (weight 0.20)
    file_hits = _count_keyword_hits(combined, _FILE_HINT_KEYWORDS)
    file_score = min(file_hits / 3.0, 1.0)

    # Signal 3: Domain complexity (weight 0.20)
    domain_hits = _count_keyword_hits(combined, _DOMAIN_COMPLEX_KEYWORDS)
    domain_score = min(domain_hits / 3.0, 1.0)

    # Signal 4: Condition count (weight 0.15)
    # Look for numbered conditions, ablation mentions, variant mentions
    condition_pattern = re.compile(
        r"(?:condition|ablation|variant|experiment)\s*[\-_:]?\s*\d+",
        re.IGNORECASE,
    )
    condition_matches = len(condition_pattern.findall(combined))
    # Also count bullet points in conditions/ablations sections
    condition_matches += combined.lower().count("baseline")
    condition_score = min(condition_matches / 8.0, 1.0)

    # Signal 5: Historical failures (weight 0.10)
    failure_score = min(historical_failures / 3.0, 1.0)

    # Signal 6: Dependency depth (weight 0.10)
    dep_hits = _count_keyword_hits(combined, _DEPENDENCY_KEYWORDS)
    dep_score = min(dep_hits / 3.0, 1.0)

    # Weighted sum
    weighted = (
        0.25 * component_score
        + 0.20 * file_score
        + 0.20 * domain_score
        + 0.15 * condition_score
        + 0.10 * failure_score
        + 0.10 * dep_score
    )
    final_score = min(max(weighted, 0.0), 1.0)

    signals = {
        "component_count": round(component_score, 3),
        "file_count_hint": round(file_score, 3),
        "domain_complexity": round(domain_score, 3),
        "condition_count": round(condition_score, 3),
        "historical_failure": round(failure_score, 3),
        "dependency_depth": round(dep_score, 3),
    }

    if final_score >= threshold:
        recommendation = "beast_mode"
        reason = (
            f"Complexity {final_score:.2f} >= threshold {threshold:.2f}: "
            f"top signals: "
            + ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(signals.items(), key=lambda x: -x[1])[:3]
            )
        )
    else:
        recommendation = "code_agent"
        reason = f"Complexity {final_score:.2f} < threshold {threshold:.2f}"

    return ComplexityScore(
        score=round(final_score, 4),
        signals=signals,
        recommendation=recommendation,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# OpenCode bridge
# ---------------------------------------------------------------------------

@dataclass
class OpenCodeResult:
    """Result from an OpenCode invocation."""

    success: bool
    files: dict[str, str] = field(default_factory=dict)
    opencode_log: str = ""
    elapsed_sec: float = 0.0
    error: str = ""


_MEGA_PROMPT_TEMPLATE = """\
You are implementing a complete, runnable ML/science experiment.

STEP 1 — READ THE WORKSPACE:
The workspace contains the COMPLETE source code of a local codebase plus context files:
- EXPERIMENT_PLAN.yaml — the experiment design
- GUIDANCE.md — topic, metric, environment, constraints
- Source code files — the actual codebase you MUST build upon
- `data/` — symlink to local datasets (reference images, masks, configs)
- `checkpoints/` — symlink to pretrained model weights

Before writing ANY code, read the existing source files to understand the codebase:
1. Find and read any example/demo scripts (e.g. *_demo.py, run_*.py, example_*.py) to see how the pipeline works end-to-end.
2. Read the core modules to understand the API (class signatures, function arguments).
3. Read the dataset configs or sample data in data/ to understand the data format.
4. Check what checkpoints/weights are available in checkpoints/.

STEP 2 — IMPLEMENTATION RULES:
- Build ON TOP of the existing codebase. Import and extend existing modules.
- Load pretrained models from `checkpoints/` using the codebase's loading API, NOT from the internet.
- Load real data from `data/` using the codebase's data loading utilities, NOT synthetic torch.randn().
- Compute REAL metrics from actual model outputs. NEVER use np.random or random.uniform as a metric placeholder.
- Each experimental condition must produce genuinely different behavior.
- Do NOT rewrite modules that already exist — import and extend them.
- NEVER invent module names — only use modules visible in the workspace.

STEP 3 — CREATE main.py:
1. main.py is the NEW entry point that runs ALL experimental conditions.
2. It must print the primary metric as: {metric}: <value>
3. Use multi-seed evaluation (seeds 0, 1, 2) and report mean +/- std.
4. Implement a time guard: stop gracefully at 80% of the time budget ({time_budget_sec} seconds).
5. Each condition must be wrapped in try/except for crash resilience.
6. Print per-condition results: condition=<name> seed=<s> {metric}: <value>

IMPORTANT CONSTRAINTS:
- Do NOT use argparse or CLI arguments — hardcode all configuration.
- All output must go to stdout (print statements).
- Keep the experiment feasible within {time_budget_sec} seconds total.
"""


class OpenCodeBridge:
    """Manages OpenCode CLI invocations for beast mode code generation."""

    def __init__(
        self,
        *,
        model: str = "",
        llm_base_url: str = "",
        api_key_env: str = "",
        llm_provider: str = "openai-compatible",
        timeout_sec: int = 600,
        max_retries: int = 1,
        workspace_cleanup: bool = True,
    ) -> None:
        self._model = model
        self._llm_base_url = llm_base_url
        self._api_key_env = api_key_env
        self._llm_provider = llm_provider
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._workspace_cleanup = workspace_cleanup

    # -- availability check ---------------------------------------------------

    @staticmethod
    def check_available() -> bool:
        """Return True if the ``opencode`` CLI is installed and callable."""
        try:
            result = subprocess.run(
                ["opencode", "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:  # noqa: BLE001
            return False

    # -- workspace preparation ------------------------------------------------

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
        """Create a temporary workspace directory with context files."""
        import shutil
        import hashlib

        ws = stage_dir / f"opencode_beast_{int(time.time())}_{time.monotonic_ns() % 100000}"
        ws.mkdir(parents=True, exist_ok=True)

        # Write experiment plan
        (ws / "EXPERIMENT_PLAN.yaml").write_text(
            exp_plan or "# No experiment plan provided\n",
            encoding="utf-8",
        )

        # Write guidance document
        guidance_parts = [
            f"# Experiment Guidance\n",
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

        # Write opencode.json config
        opencode_cfg = self._build_opencode_config()
        (ws / "opencode.json").write_text(
            json.dumps(opencode_cfg, indent=2), encoding="utf-8",
        )

        # Copy local codebases into workspace root so OpenCode can
        # directly import and modify them.
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

        # Symlink datasets and checkpoints into workspace
        if datasets_dir and Path(datasets_dir).is_dir():
            link = ws / "data"
            if not link.exists():
                link.symlink_to(Path(datasets_dir).resolve())

        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            link = ws / "checkpoints"
            if not link.exists():
                link.symlink_to(Path(checkpoints_dir).resolve())

        # Append concrete usage hints to GUIDANCE.md so LLM knows
        # how to use the local data/checkpoints/codebase in practice.
        usage_hints: list[str] = []
        if (ws / "data").exists():
            try:
                ds_root = Path(datasets_dir).resolve()
                ds_items = sorted(
                    d.name for d in ds_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                )
                usage_hints.append(
                    "## Local Data Layout\n"
                    f"Datasets available in `data/`: {', '.join(ds_items)}\n"
                )
                for ds_name in ds_items[:3]:
                    ds_sub = ds_root / ds_name
                    sub_items = sorted(d.name for d in ds_sub.rglob("*") if d.is_file())[:10]
                    if sub_items:
                        usage_hints.append(f"- `data/{ds_name}/`: {', '.join(sub_items)}")
            except OSError:
                pass
        if (ws / "checkpoints").exists():
            try:
                ck_root = Path(checkpoints_dir).resolve()
                ck_items = sorted(
                    d.name for d in ck_root.iterdir() if not d.name.startswith(".")
                )
                ck_hint = (
                    "\n## Local Checkpoints\n"
                    f"Available in `checkpoints/`: {', '.join(ck_items)}\n"
                    "Load pretrained models from this path instead of downloading.\n"
                )
                if ck_items:
                    ck_hint += f"Example: `model.from_pretrained('checkpoints/{ck_items[0]}')`"
                usage_hints.append(ck_hint)
            except OSError:
                pass
        if usage_hints:
            with open(ws / "GUIDANCE.md", "a", encoding="utf-8") as f:
                f.write("\n\n" + "\n".join(usage_hints) + "\n")

        # Snapshot codebase file hashes so _collect_files can distinguish
        # OpenCode's new/modified files from the original codebase.
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

    def _is_azure(self) -> bool:
        """Detect Azure OpenAI from base URL or provider string."""
        return (
            "azure" in (self._llm_base_url or "").lower()
            or "azure" in (self._llm_provider or "").lower()
        )

    def _build_opencode_config(self) -> dict[str, Any]:
        """Build the opencode.json configuration."""
        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
        }

        if self._is_azure():
            # Azure OpenAI provider — uses "azure" provider type in OpenCode
            # Extract resource name from URL like:
            #   https://myresource-eastus2.services.ai.azure.com/openai/v1
            #   https://myresource.openai.azure.com/openai
            resource_name = ""
            if self._llm_base_url:
                m = re.match(r"https?://([^.]+)", self._llm_base_url)
                if m:
                    resource_name = m.group(1)

            # Normalize base URL: Azure provider wants the /openai path
            base_url = self._llm_base_url.rstrip("/")
            if not base_url.endswith("/openai"):
                # Strip /v1 suffix if present, add /openai if needed
                base_url = base_url.removesuffix("/v1")
                if not base_url.endswith("/openai"):
                    base_url += "/openai"

            if self._model:
                # If model already has a provider prefix (e.g. "anthropic/..."), use as-is
                cfg["model"] = self._model if "/" in self._model else f"azure/{self._model}"
            cfg["provider"] = {
                "azure": {
                    "options": {
                        "apiKey": f"{{env:{self._api_key_env}}}"
                        if self._api_key_env
                        else "",
                        "baseURL": base_url,
                        "resourceName": resource_name,
                    },
                    "models": {},
                }
            }
            # Register the model so OpenCode knows it exists
            if self._model:
                cfg["provider"]["azure"]["models"] = {
                    self._model: {
                        "name": self._model,
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"],
                        },
                    }
                }
        elif self._llm_base_url:
            resolved = self._model if "/" in self._model else f"openai/{self._model}"
            if self._model:
                cfg["model"] = resolved
            provider_cfg: dict[str, Any] = {
                "options": {
                    "baseURL": self._llm_base_url,
                    "apiKey": f"{{env:{self._api_key_env}}}"
                    if self._api_key_env
                    else "",
                },
            }
            if self._model:
                bare_name = self._model.split("/")[-1] if "/" in self._model else self._model
                provider_cfg["models"] = {
                    bare_name: {
                        "name": bare_name,
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"],
                        },
                    }
                }
            cfg["provider"] = {"openai": provider_cfg}
        elif self._model:
            cfg["model"] = self._model if "/" in self._model else f"openai/{self._model}"

        return cfg

    # -- model resolution -------------------------------------------------------

    def _resolve_opencode_model(self) -> str:
        """Resolve the model identifier for OpenCode CLI's ``-m`` flag.

        Azure OpenAI endpoints use the Responses API which many Azure deployments
        don't support.  When the configured provider is Azure, we fall back to
        using Anthropic models directly (which OpenCode supports natively) rather
        than trying to proxy through Azure.

        Resolution order:
        1. If model already contains "/" (e.g. "anthropic/claude-sonnet-4-6") → use as-is
        2. If NOT Azure → "openai/{model}"
        3. If Azure → fall back to "anthropic/claude-sonnet-4-6" (reliable default)
        """
        if not self._model:
            return "anthropic/claude-sonnet-4-6"
        if "/" in self._model:
            return self._model
        if self._is_azure():
            # Azure AI Services endpoints don't support OpenCode's Responses API.
            # Fall back to Anthropic which OpenCode supports natively.
            logger.info(
                "Beast mode: Azure endpoint detected — using Anthropic model "
                "for OpenCode (Azure doesn't support Responses API)"
            )
            return "anthropic/claude-sonnet-4-6"
        return f"openai/{self._model}"

    # -- invocation ------------------------------------------------------------

    def _invoke_opencode(
        self,
        workspace: Path,
        prompt: str,
    ) -> tuple[bool, str, float]:
        """Run ``opencode run`` in the workspace. Returns (success, log, elapsed)."""
        workspace = workspace.resolve()
        env = os.environ.copy()
        # Pass API key via environment if configured
        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env, "")
            if api_key:
                if self._is_azure():
                    env["AZURE_API_KEY"] = api_key
                else:
                    env["OPENAI_API_KEY"] = api_key
        # OpenCode v1.2+ needs stdin closed to avoid hanging on TTY detection
        env["DO_NOT_TRACK"] = "1"

        resolved_model = self._resolve_opencode_model()
        cmd = [
            "opencode", "run",
            "-m", resolved_model,
            "--format", "json",
            "--dir", str(workspace),
            prompt,
        ]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            elapsed = time.monotonic() - t0
            log = result.stdout + "\n" + result.stderr
            return result.returncode == 0, log, elapsed
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            log = f"TIMEOUT after {elapsed:.1f}s"
            if exc.stdout:
                log += f"\nstdout: {exc.stdout[:2000] if isinstance(exc.stdout, str) else exc.stdout.decode(errors='replace')[:2000]}"
            return False, log, elapsed
        except FileNotFoundError:
            return False, "opencode CLI not found", 0.0
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            return False, f"Unexpected error: {exc}", elapsed

    # -- file collection -------------------------------------------------------

    @staticmethod
    def _collect_files(workspace: Path) -> dict[str, str]:
        """Collect generated Python files, requirements.txt, and setup.py.

        Only collects files that are **new or modified** relative to the
        codebase snapshot taken during workspace preparation.  Unchanged
        codebase files are skipped so downstream stages only see
        OpenCode's actual output.

        File names are flattened to basenames (e.g. ``src/main.py`` →
        ``main.py``) because the downstream executor expects a flat file
        dict.  If two files share the same basename, the one closer to the
        workspace root wins.
        """
        import hashlib as _hl

        # Load the original codebase snapshot (if any)
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
                    current_hash = _hl.md5(py_file.read_bytes()).hexdigest()
                except OSError:
                    continue
                if current_hash == _original_hashes[rel_str]:
                    continue  # unchanged codebase file

            basename = rel.name
            if basename not in files:
                try:
                    files[basename] = py_file.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as exc:
                    logger.warning(
                        "Beast mode: failed to read %s: %s", py_file, exc
                    )

        for extra in ("requirements.txt", "setup.py"):
            p = workspace / extra
            if p.exists() and extra not in files:
                # Check snapshot for these too
                if extra in _original_hashes:
                    try:
                        cur = _hl.md5(p.read_bytes()).hexdigest()
                    except OSError:
                        continue
                    if cur == _original_hashes.get(extra):
                        continue
                files[extra] = p.read_text(encoding="utf-8", errors="replace")

        return files

    # -- main entry point ------------------------------------------------------

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
    ) -> OpenCodeResult:
        """Run OpenCode to generate experiment code.

        Returns an OpenCodeResult with success status and generated files.
        """
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="OpenCode CLI not installed or not callable",
            )

        workspace: Path | None = None
        last_error = ""

        for attempt in range(1 + self._max_retries):
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
                logger.warning("Beast mode: %s", last_error)
                continue

            # Build the mega-prompt (use replace instead of .format() to
            # avoid KeyError when metric contains curly braces like "F{1}")
            prompt = _MEGA_PROMPT_TEMPLATE.replace(
                "{metric}", metric
            ).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )

            logger.info(
                "Beast mode: invoking OpenCode (attempt %d/%d, timeout=%ds)",
                attempt + 1,
                1 + self._max_retries,
                self._timeout_sec,
            )

            success, log, elapsed = self._invoke_opencode(workspace, prompt)

            # Collect files regardless of exit status — OpenCode may have
            # written valid code before a timeout or non-zero exit.
            files = self._collect_files(workspace) if workspace.exists() else {}

            if "main.py" in files:
                if not success:
                    logger.info(
                        "Beast mode: OpenCode exited non-zero / timed out but "
                        "main.py found (%d files) — treating as success",
                        len(files),
                    )

                (stage_dir / "opencode_log.txt").write_text(
                    log, encoding="utf-8",
                )

                if self._workspace_cleanup and workspace.exists():
                    shutil.rmtree(workspace, ignore_errors=True)

                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = log if not success else "No main.py in OpenCode output"
            logger.warning(
                "Beast mode: OpenCode attempt %d failed (%.1fs, files=%s): %s",
                attempt + 1,
                elapsed,
                list(files.keys()),
                last_error[:500],
            )
            if self._workspace_cleanup and workspace and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

        # All attempts failed
        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            error=f"OpenCode failed after {1 + self._max_retries} attempt(s)",
        )


# ---------------------------------------------------------------------------
# Helper: count historical failures
# ---------------------------------------------------------------------------

def count_historical_failures(run_dir: Path, stage_name: str = "stage-10") -> int:
    """Count past Stage 10 failures from stage directories and logs.

    Each stage directory is counted at most once, even if multiple failure
    indicators are present.
    """
    failures = 0
    for d in run_dir.glob(f"{stage_name}*"):
        failed = False
        # Check for beast_mode_log.json
        bm_log = d / "beast_mode_log.json"
        if bm_log.exists():
            try:
                data = json.loads(bm_log.read_text(encoding="utf-8"))
                if not data.get("success", True):
                    failed = True
            except Exception:  # noqa: BLE001
                pass
        # Check for stage health failures
        if not failed:
            health = d / "stage_health.json"
            if health.exists():
                try:
                    data = json.loads(health.read_text(encoding="utf-8"))
                    if data.get("status") == "FAILED":
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        # Check for validation report with FAILED status
        if not failed:
            vr = d / "validation_report.md"
            if vr.exists():
                try:
                    content = vr.read_text(encoding="utf-8")
                    if "BLOCKED" in content or "FAILED" in content:
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        if failed:
            failures += 1
    return failures
