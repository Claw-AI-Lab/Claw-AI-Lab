"""AST-based hard validation gates for generated code.

Delegates to ``researchclaw.experiment.validator`` for the heavy lifting.
This module wraps the calls and classifies issues as critical vs warning.
"""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.pipeline.codegen.types import GeneratedFiles

logger = logging.getLogger(__name__)


def run_hard_validation(files: GeneratedFiles) -> tuple[list[str], list[str]]:
    """Run AST and quality checks on generated files.

    Returns (complexity_warnings, deep_warnings) where each entry is a
    human-readable string.
    """
    from researchclaw.experiment.validator import (
        auto_fix_unbound_locals,
        check_code_complexity,
        check_main_entry_point,
        deep_validate_files,
    )

    complexity_warnings: list[str] = []

    if "main.py" in files:
        entry_warnings = check_main_entry_point(files["main.py"])
        for w in entry_warnings:
            complexity_warnings.append(f"[main.py entry] {w}")

    for fname, code in files.items():
        if fname.endswith(".py"):
            cw = check_code_complexity(code)
            for w in cw:
                complexity_warnings.append(f"[{fname}] {w}")

    deep_warnings = deep_validate_files(files)
    for w in deep_warnings:
        complexity_warnings.append(w)

    return complexity_warnings, deep_warnings


def auto_fix_unbound(files: GeneratedFiles) -> tuple[GeneratedFiles, int]:
    """Programmatic fix for UnboundLocalError patterns.

    Returns (fixed_files, total_fixes_applied).
    """
    from researchclaw.experiment.validator import auto_fix_unbound_locals

    total_fixes = 0
    result = dict(files)
    for fname, code in list(result.items()):
        if fname.endswith(".py"):
            fixed_code, n_fixes = auto_fix_unbound_locals(code)
            if n_fixes > 0:
                result[fname] = fixed_code
                total_fixes += n_fixes
                logger.info(
                    "auto-fixed %d UnboundLocalError risk(s) in %s",
                    n_fixes, fname,
                )
    return result, total_fixes


def classify_critical_issues(deep_warnings: list[str]) -> list[str]:
    """Extract critical issues from deep validation warnings.

    Critical issues are those that would cause runtime failures or
    produce scientifically invalid results.
    """
    critical_keywords = (
        "UnboundLocalError", "unregistered", "does not exist",
        "empty or trivial subclass", "does NOT override",
        "Import-usage mismatch", "NameError",
        "was removed", "ptp()",
        "copy-paste", "identical method signatures",
        "identical AST", "NOT a real ablation",
    )
    return [w for w in deep_warnings if any(kw in w for kw in critical_keywords)]
