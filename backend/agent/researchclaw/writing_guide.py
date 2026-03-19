"""Conference-grade writing knowledge base.

Structured tips from NeurIPS/ICML/ICLR best practices, reviewer feedback
analysis, and accepted paper patterns. Can be loaded and injected into
prompts at runtime, allowing updates without modifying prompt YAML.
"""

from __future__ import annotations

CONFERENCE_WRITING_TIPS: dict[str, list[str]] = {
    "title": [
        "Signal novelty — title should hint at what is new",
        "Be specific and concrete, under 15 words",
        "No abbreviations unless universally known",
        "Pattern: '[Finding]: [Evidence]' or '[Method]: [What it does]'",
        "Memeability test: would a reader enjoy telling a colleague about this?",
    ],
    "abstract": [
        "5-sentence structure: (1) problem, (2) prior approaches + limitations, "
        "(3) your approach + novelty, (4) key results with numbers, (5) implication",
        "150-250 words for ML conferences",
        "Include at least 2 specific quantitative results",
    ],
    "figure_1": [
        "Most important figure in the paper — many readers look at Figure 1 first",
        "Should convey the key idea or main result at a glance",
        "Invest significant time in this figure",
    ],
    "introduction": [
        "State contributions clearly as bullet points",
        "Many reviewers stop reading carefully after the intro",
        "Include paper organization paragraph at the end",
    ],
    "experiments": [
        "Strong baselines: tune baselines with the same effort as your method",
        "Ablations: remove one component at a time and measure the effect",
        "Reproducibility: include hyperparameters, seeds, hardware specs",
        "Statistical rigor: report variance, run multiple seeds",
    ],
    "common_rejections": [
        "Weak baselines (79% of rejected papers)",
        "Missing ablations",
        "Overclaiming beyond evidence",
        "Poor reproducibility details",
        "Ignoring limitations",
    ],
    "rebuttal": [
        "Start with positives reviewers identified",
        "Quote reviewers directly, then respond",
        "Provide new data/experiments rather than arguing",
        "Do not promise — deliver",
    ],
}


def format_writing_tips(categories: list[str] | None = None) -> str:
    """Format writing tips as a prompt-injectable string.

    Parameters
    ----------
    categories:
        Subset of tip categories to include. If *None*, include all.

    Returns
    -------
    str
        Formatted markdown-style tips block.
    """
    lines: list[str] = ["## Conference Writing Best Practices"]
    cats = categories or list(CONFERENCE_WRITING_TIPS.keys())
    for cat in cats:
        tips = CONFERENCE_WRITING_TIPS.get(cat, [])
        if not tips:
            continue
        lines.append(f"\n### {cat.replace('_', ' ').title()}")
        for tip in tips:
            lines.append(f"- {tip}")
    return "\n".join(lines)
