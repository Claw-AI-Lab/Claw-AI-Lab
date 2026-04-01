"""LLM code review — senior researcher review prompt.

Sends the generated code to the LLM for a structured review, then
applies fixes for critical issues found.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from researchclaw.pipeline.codegen.session import CodegenSession
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenPhase, GeneratedFiles

logger = logging.getLogger(__name__)


def run_code_review(
    files: GeneratedFiles,
    ctx: CodegenContext,
    config: Any,
    llm: Any,
    session: CodegenSession,
    prompts: Any | None = None,
    max_tokens: int = 8192,
) -> GeneratedFiles:
    """Run LLM code review and fix critical issues.

    Returns the (potentially repaired) files.
    """
    from researchclaw.pipeline.executor import (
        _chat_with_prompt,
        _extract_multi_file_blocks,
        _safe_json_loads,
    )

    session.log(CodegenPhase.REVIEW, "LLM code review started")

    all_code = "\n\n".join(
        f"# --- {fname} ---\n{code}" for fname, code in files.items()
    )
    if len(all_code) > 12000:
        all_code = all_code[:12000] + "\n... [truncated]"

    review_prompt = (
        f"You are a senior researcher reviewing experiment code for a "
        f"research submission.\n\n"
        f"TOPIC: {ctx.topic}\n"
        f"EXPERIMENT PLAN:\n{ctx.exp_plan[:3000]}\n\n"
        f"CODE:\n```python\n{all_code}\n```\n\n"
        f"Review the code and return JSON with this EXACT structure:\n"
        f'{{"score": <1-10>, "issues": ['
        f'{{"severity": "critical|major|minor", '
        f'"description": "...", "fix": "..."}}], '
        f'"verdict": "pass|needs_fix"}}\n\n'
        f"Check specifically:\n"
        f"1. Does each algorithm/method have a DISTINCT implementation? "
        f"(Not just renamed copies)\n"
        f"2. Are ablation conditions genuinely different from the main method?\n"
        f"3. Are loss functions / training loops mathematically correct?\n"
        f"4. Will the code actually run without errors? Check variable scoping, "
        f"API usage, tensor shape compatibility.\n"
        f"5. Is the code complex enough for a research paper? (Not trivial)\n"
        f"6. Are experimental conditions fairly compared (same seeds, data)?\n"
        f"7. If using pretrained models (EfficientNet, ResNet, ViT), are input "
        f"images resized to the model's expected size (e.g., 224x224)?\n"
        f"8. Are imports consistent? `from X import Y` must use `Y()`, not `X.Y()`.\n"
    )

    try:
        review_resp = llm.chat(
            [{"role": "user", "content": review_prompt}],
            system="You are a meticulous ML code reviewer. Be strict.",
            max_tokens=2048,
        )
        session.llm_calls += 1

        review_text = review_resp.content if hasattr(review_resp, "content") else str(review_resp)
        review_text = review_text.strip()
        if review_text.startswith("```"):
            lines = review_text.splitlines()
            start = 1 if lines[0].strip().startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            review_text = "\n".join(lines[start:end])

        review_data = _safe_json_loads(review_text, {})
        if not isinstance(review_data, dict):
            session.log(CodegenPhase.REVIEW, "Could not parse review JSON")
            return files

        review_score = review_data.get("score", 0)
        review_verdict = review_data.get("verdict", "unknown")
        review_issues = review_data.get("issues", [])

        if ctx.stage_dir is not None:
            (ctx.stage_dir / "code_review.json").write_text(
                json.dumps({
                    "score": review_score,
                    "verdict": review_verdict,
                    "issues": review_issues,
                }, indent=2),
                encoding="utf-8",
            )

        session.log(
            CodegenPhase.REVIEW,
            f"Review: score={review_score}/10, verdict={review_verdict}, "
            f"issues={len(review_issues)}",
        )

        critical_issues = [
            i for i in review_issues
            if isinstance(i, dict) and i.get("severity") == "critical"
        ]

        if critical_issues and review_score <= 4:
            files = _fix_review_issues(
                files, critical_issues, review_score, llm, session,
                prompts=prompts, max_tokens=max_tokens,
            )

    except Exception as exc:
        logger.debug("Code review failed: %s", exc)
        session.log(CodegenPhase.REVIEW, f"Review failed: {exc}")

    return files


def _fix_review_issues(
    files: GeneratedFiles,
    critical_issues: list[dict[str, Any]],
    review_score: int,
    llm: Any,
    session: CodegenSession,
    prompts: Any | None = None,
    max_tokens: int = 8192,
) -> GeneratedFiles:
    """Fix critical issues identified during code review."""
    from researchclaw.pipeline.executor import (
        _chat_with_prompt,
        _extract_multi_file_blocks,
    )

    logger.warning(
        "Code review: score=%d, %d critical issues — attempting fix",
        review_score, len(critical_issues),
    )

    fix_descriptions = "\n".join(
        f"- [{i.get('severity', '?')}] {i.get('description', '?')}: "
        f"{i.get('fix', 'no fix suggested')}"
        for i in critical_issues
    )

    system_prompt = ""
    if prompts is not None:
        try:
            system_prompt = prompts.prompts["code_generation"]["system"]
        except (KeyError, AttributeError, TypeError):
            pass

    fix_prompt = (
        f"Code review found {len(critical_issues)} CRITICAL issues "
        f"(score: {review_score}/10):\n{fix_descriptions}\n\n"
        f"Fix ALL critical issues. Return complete corrected files "
        f"using ```filename:xxx.py format.\n"
        f"Do NOT add try/except blocks — fix the root cause instead.\n\n"
        f"Current code:\n"
        + "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
    )

    try:
        fix_resp = _chat_with_prompt(
            llm, system_prompt, fix_prompt, max_tokens=max_tokens,
        )
        session.llm_calls += 1
        fixed_files = _extract_multi_file_blocks(fix_resp.content)
        if fixed_files and "main.py" in fixed_files:
            logger.info(
                "Code fixed after review (was %d/10, %d critical issues)",
                review_score, len(critical_issues),
            )
            return fixed_files
    except Exception as exc:
        logger.debug("Review-fix failed: %s", exc)

    return files
