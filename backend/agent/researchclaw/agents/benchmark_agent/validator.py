"""Validator Agent — validates generated code for correctness.

Performs three levels of validation:
1. **Syntax check** — ``ast.parse()`` on generated Python code.
2. **Import check** — verifies that referenced modules are importable
   or listed in requirements.
3. **LLM review** — asks the LLM to review code for common pitfalls
   (wrong API usage, missing transforms, incorrect splits).
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from researchclaw.agents.base import AgentStepResult, BaseAgent

logger = logging.getLogger(__name__)

# Packages available in Docker image (no pip install needed)
_BUILTIN_MODULES = {
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "sklearn",
    "pandas", "matplotlib", "seaborn", "tqdm", "gymnasium", "networkx",
    "timm", "einops", "torchmetrics", "transformers", "datasets",
    "accelerate", "peft", "trl", "bitsandbytes", "tokenizers",
    "safetensors", "h5py", "tensorboard", "PIL", "yaml", "kornia",
    "albumentations", "cv2", "mujoco", "os", "sys", "json", "re",
    "pathlib", "typing", "collections", "functools", "itertools",
    "math", "random", "copy", "dataclasses", "abc", "io", "csv",
    "glob", "shutil", "time", "datetime", "logging", "warnings",
    "argparse", "pickle", "struct", "hashlib",
}


class ValidatorAgent(BaseAgent):
    """Validates generated code artifacts for syntax and API correctness."""

    name = "validator"

    def _check_syntax(self, code: str, label: str) -> list[str]:
        """Check Python syntax via ast.parse.  Returns list of errors."""
        if not code.strip():
            return []
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [f"{label}: SyntaxError at line {e.lineno}: {e.msg}"]

    def _check_imports(
        self,
        code: str,
        label: str,
        extra_requirements: list[str],
    ) -> list[str]:
        """Check that imported modules are available or declared."""
        if not code.strip():
            return []

        warnings: list[str] = []
        # Extract import statements
        import_pattern = re.compile(
            r"^\s*(?:import|from)\s+(\w+)", re.MULTILINE,
        )
        imports = set(import_pattern.findall(code))

        # Build allowed set
        allowed = set(_BUILTIN_MODULES)
        # Map pip package names to import names
        pip_to_import = {
            "torch-geometric": "torch_geometric",
            "ogb": "ogb",
            "stable-baselines3": "stable_baselines3",
            "xgboost": "xgboost",
            "opencv-python": "cv2",
            "scikit-learn": "sklearn",
            "gymnasium[mujoco]": "gymnasium",
            "huggingface_hub": "huggingface_hub",
        }
        for pkg in extra_requirements:
            import_name = pip_to_import.get(pkg, pkg.replace("-", "_"))
            allowed.add(import_name)

        for mod in imports:
            if mod not in allowed:
                warnings.append(
                    f"{label}: import '{mod}' not in Docker image or requirements"
                )

        return warnings

    def _llm_review(
        self,
        data_code: str,
        baseline_code: str,
        setup_code: str,
        benchmark_names: list[str],
        baseline_names: list[str],
    ) -> dict[str, Any]:
        """Ask LLM to review generated code for common pitfalls."""
        system = (
            "You are a code reviewer specializing in ML experiment code. "
            "Review the following generated code for correctness.\n\n"
            "Check for:\n"
            "1. Correct API usage (torchvision, HuggingFace datasets, PyG, etc.)\n"
            "2. Proper data transforms and normalization\n"
            "3. Correct train/val/test split handling\n"
            "4. Compatible input/output dimensions between data and models\n"
            "5. Missing error handling for optional dependencies\n"
            "6. Hardcoded paths that should use variables\n"
            "7. Missing download=True in setup.py for tier 2 datasets\n\n"
            "Return JSON:\n"
            "{\n"
            '  "passed": true/false,\n'
            '  "issues": ["issue 1", "issue 2"],\n'
            '  "suggestions": ["suggestion 1"],\n'
            '  "severity": "none" | "warning" | "error"\n'
            "}"
        )

        code_sections = []
        if data_code:
            code_sections.append(f"## Data Loading Code\n```python\n{data_code}\n```")
        if baseline_code:
            code_sections.append(f"## Baseline Code\n```python\n{baseline_code}\n```")
        if setup_code:
            code_sections.append(f"## Setup Script\n```python\n{setup_code}\n```")

        user = (
            f"Benchmarks: {', '.join(benchmark_names)}\n"
            f"Baselines: {', '.join(baseline_names)}\n\n"
            + "\n\n".join(code_sections)
        )

        return self._chat_json(system, user, max_tokens=2048)

    # -- Main entry point --------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Validate all generated code artifacts.

        Context keys:
            acquisition (dict): Output from AcquirerAgent
        """
        acq = context.get("acquisition", {})

        data_code = acq.get("data_loader_code", "")
        baseline_code = acq.get("baseline_code", "")
        setup_code = acq.get("setup_code", "")
        requirements = acq.get("requirements", "")
        benchmark_names = acq.get("benchmark_names", [])
        baseline_names = acq.get("baseline_names", [])

        extra_pip = [r.strip() for r in requirements.split("\n") if r.strip()]

        all_errors: list[str] = []
        all_warnings: list[str] = []

        # 1. Syntax checks
        for code, label in [
            (data_code, "data_loader"),
            (baseline_code, "baseline"),
            (setup_code, "setup"),
        ]:
            errors = self._check_syntax(code, label)
            all_errors.extend(errors)

        # 2. Import checks
        for code, label in [
            (data_code, "data_loader"),
            (baseline_code, "baseline"),
        ]:
            warnings = self._check_imports(code, label, extra_pip)
            all_warnings.extend(warnings)

        # 3. LLM review (only if no syntax errors)
        llm_review: dict[str, Any] = {}
        if not all_errors:
            llm_review = self._llm_review(
                data_code, baseline_code, setup_code,
                benchmark_names, baseline_names,
            )
            if llm_review.get("severity") == "error":
                all_errors.extend(llm_review.get("issues", []))
            elif llm_review.get("issues"):
                all_warnings.extend(llm_review.get("issues", []))

        passed = len(all_errors) == 0
        severity = "error" if all_errors else ("warning" if all_warnings else "none")

        result = {
            "passed": passed,
            "errors": all_errors,
            "warnings": all_warnings,
            "severity": severity,
            "llm_review": llm_review,
            "suggestions": llm_review.get("suggestions", []),
        }

        self.logger.info(
            "Validation %s: %d errors, %d warnings",
            "PASSED" if passed else "FAILED",
            len(all_errors), len(all_warnings),
        )

        return self._make_result(passed, data=result)
