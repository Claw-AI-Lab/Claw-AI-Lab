"""Conference-aware LaTeX template system.

Supports automatic template switching for NeurIPS, ICLR, and ICML.
Given a target conference name, generates a complete ``.tex`` file from
Markdown paper content + BibTeX references.

Usage::

    from researchclaw.templates import get_template, markdown_to_latex

    tpl = get_template("neurips_2025")
    tex = markdown_to_latex(paper_md, tpl, title=..., authors=..., bib_file="references.bib")
"""

from researchclaw.templates.conference import (
    CONFERENCE_REGISTRY,
    ConferenceTemplate,
    get_template,
    list_conferences,
)
from researchclaw.templates.converter import markdown_to_latex

__all__ = [
    "CONFERENCE_REGISTRY",
    "ConferenceTemplate",
    "get_template",
    "list_conferences",
    "markdown_to_latex",
]
