"""LLM-driven repair for critical code issues.

Sends identified critical issues back to the LLM along with the current
code and asks for targeted fixes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import CodegenPhase, GeneratedFiles

logger = logging.getLogger(__name__)


def repair_critical_issues(
    files: GeneratedFiles,
    critical_issues: list[str],
    llm: Any,
    prompts: Any | None = None,
    max_tokens: int = 8192,
) -> GeneratedFiles:
    """Ask the LLM to fix critical validation issues.

    Returns the repaired files (merged with originals for unchanged files).
    """
    from researchclaw.pipeline.executor import (
        _chat_with_prompt,
        _extract_multi_file_blocks,
    )

    repair_issues = "\n".join(f"- {w}" for w in critical_issues)
    all_code_ctx = "\n\n".join(
        f"```filename:{f}\n{c}\n```" for f, c in files.items()
    )

    system_prompt = ""
    if prompts is not None:
        try:
            system_prompt = prompts.prompts["code_generation"]["system"]
        except (KeyError, AttributeError, TypeError):
            pass

    repair_prompt = (
        f"CRITICAL CODE QUALITY ISSUES FOUND:\n{repair_issues}\n\n"
        f"Fix ALL these issues in the code below. Return the complete "
        f"corrected files using ```filename:xxx.py format.\n\n"
        f"RULES:\n"
        f"- nn.Linear/nn.Conv must be created in __init__(), not forward()\n"
        f"- Variables used after if/else must be defined before the branch\n"
        f"- Use scipy.special.erf, not np.erf\n"
        f"- Ablation/variant classes must have genuinely different logic\n"
        f"- Every class must have a real implementation, not just `pass`\n"
        f"- Ablation classes MUST override the parent method that implements "
        f"the component being ablated (e.g., if ablating attention, override "
        f"the attention method with a simpler alternative like mean pooling)\n"
        f"- IMPORT CONSISTENCY: if you write `from X import Y`, call `Y()` "
        f"directly — NOT `X.Y()`. Mixing styles causes NameError.\n"
        f"- NumPy 2.0: ndarray.ptp() was removed — use arr.max()-arr.min()\n"
        f"- NumPy 2.0: np.bool/np.int/np.float removed — use builtins\n"
        f"- Pretrained models (EfficientNet, ResNet, ViT) expect 224x224 input "
        f"— add `transforms.Resize(224)` when using CIFAR (32x32) or similar\n"
        f"- Copy-paste ablation: if two classes have identical bodies, REWRITE "
        f"the ablation to genuinely remove/reduce a component\n"
        f"- NO try/except blocks — fix the root cause of errors instead of "
        f"catching them. All errors must crash with a full traceback.\n\n"
        f"Current code:\n{all_code_ctx}\n"
    )

    try:
        resp = _chat_with_prompt(
            llm, system_prompt, repair_prompt, max_tokens=max_tokens
        )
        repaired = _extract_multi_file_blocks(resp.content)
        if repaired and "main.py" in repaired:
            from researchclaw.experiment.validator import deep_validate_files
            deep_after = deep_validate_files(repaired)
            from researchclaw.pipeline.codegen.validation.gates import classify_critical_issues
            remaining = classify_critical_issues(deep_after)
            fixed_count = len(critical_issues) - len(remaining)
            logger.info(
                "Deep repair fixed %d/%d critical issues",
                fixed_count, len(critical_issues),
            )
            return repaired
    except Exception as exc:
        logger.debug("Deep repair failed: %s", exc)

    return files
