"""Content quality assessment — template detection and metrics.

Detects placeholder/template content in LLM-generated text and provides
quality metrics for pipeline outputs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    (
        r"(?i)template\s+(abstract|introduction|method|methodology|conclusion|discussion|results|related\s+work)",
        "Template section header",
    ),
    (r"(?i)\[INSERT\s+.*?\]", "Insert placeholder"),
    (r"(?i)\[TODO\s*:?\s*.*?\]", "TODO placeholder"),
    (r"(?i)\[PLACEHOLDER\s*:?\s*.*?\]", "Explicit placeholder"),
    (r"(?i)lorem\s+ipsum", "Lorem ipsum filler"),
    (
        r"(?i)this\s+section\s+will\s+(describe|discuss|present|outline|explain)",
        "Future-tense placeholder",
    ),
    (
        r"(?i)we\s+will\s+(describe|discuss|present|outline|explain)\s+in\s+this\s+section",
        "Future-tense placeholder",
    ),
    (
        r"(?i)add\s+(your|the)\s+(content|text|description)\s+here",
        "Add content placeholder",
    ),
    (r"(?i)replace\s+this\s+(text|content|section)", "Replace placeholder"),
    (r"(?i)^#+\s*section\s+\d+\s*$", "Generic section header"),
    (
        r"(?i)your\s+(abstract|introduction|method|results)\s+goes?\s+here",
        "Content placeholder",
    ),
    (r"(?i)sample\s+(abstract|introduction|text|content)", "Sample content marker"),
]


@dataclass(frozen=True)
class TemplateMatch:
    """A single template/placeholder detection."""

    pattern_desc: str
    line_number: int
    excerpt: str


@dataclass(frozen=True)
class QualityReport:
    """Quality assessment for a text document."""

    total_lines: int
    total_chars: int
    template_matches: tuple[TemplateMatch, ...] = ()
    template_ratio: float = 0.0

    @property
    def has_template_content(self) -> bool:
        return len(self.template_matches) > 0

    @property
    def match_count(self) -> int:
        return len(self.template_matches)

    def to_dict(self) -> dict[str, object]:
        match_rows: list[dict[str, object]] = [
            {
                "pattern": m.pattern_desc,
                "line": m.line_number,
                "excerpt": m.excerpt,
            }
            for m in self.template_matches
        ]
        return {
            "total_lines": self.total_lines,
            "total_chars": self.total_chars,
            "template_matches": match_rows,
            "template_ratio": round(self.template_ratio, 4),
            "has_template_content": self.has_template_content,
            "match_count": self.match_count,
        }


def detect_template_content(text: str) -> list[TemplateMatch]:
    """Scan text for template/placeholder patterns.

    Returns list of TemplateMatch objects for each detected pattern.
    """

    matches: list[TemplateMatch] = []
    lines = text.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, desc in _TEMPLATE_PATTERNS:
            for m in re.finditer(pattern, stripped):
                excerpt = m.group(0)[:100]
                matches.append(
                    TemplateMatch(
                        pattern_desc=desc,
                        line_number=line_num,
                        excerpt=excerpt,
                    )
                )

    return matches


def compute_template_ratio(text: str) -> float:
    """Estimate what fraction of the text is template/placeholder content.

    Returns 0.0 (fully original) to 1.0 (fully template).
    Simple heuristic: count characters in matched lines vs total.
    """

    if not text.strip():
        return 0.0

    lines = text.split("\n")
    total_chars = sum(len(line.strip()) for line in lines if line.strip())
    if total_chars == 0:
        return 0.0

    template_chars = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, _ in _TEMPLATE_PATTERNS:
            if re.search(pattern, stripped):
                template_chars += len(stripped)
                break

    return min(template_chars / total_chars, 1.0)


def assess_quality(text: str) -> QualityReport:
    """Full quality assessment of a text document."""

    lines = text.split("\n")
    matches = detect_template_content(text)
    ratio = compute_template_ratio(text)

    report = QualityReport(
        total_lines=len(lines),
        total_chars=len(text),
        template_matches=tuple(matches),
        template_ratio=ratio,
    )
    logger.debug(
        "quality assessed lines=%d chars=%d matches=%d ratio=%.4f",
        report.total_lines,
        report.total_chars,
        report.match_count,
        report.template_ratio,
    )
    return report


def check_strict_quality(text: str, *, threshold: float = 0.05) -> tuple[bool, str]:
    """Check if text passes strict quality gate.

    Returns (passed, message).
    Fails if template_ratio > threshold.
    """

    report = assess_quality(text)

    if report.template_ratio > threshold:
        details = "; ".join(
            f"L{m.line_number}: {m.excerpt}" for m in report.template_matches[:5]
        )
        return False, (
            f"Template content detected: ratio={report.template_ratio:.2%}, "
            f"{report.match_count} matches. Examples: {details}"
        )

    return True, f"Quality check passed: template_ratio={report.template_ratio:.2%}"
