"""Integrator Agent — determines figure placement in the paper.

Maps each rendered figure to the correct paper section, generates
markdown image references with captions, and produces a
``figure_manifest.json`` that downstream stages use for paper embedding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult

logger = logging.getLogger(__name__)

# Mapping from figure section → paper section heading
_SECTION_MAP = {
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "architecture": "Method",
    "results": "Results",
    "experiment": "Results",
    "experiments": "Results",
    "analysis": "Analysis",
    "discussion": "Discussion",
    "ablation": "Results",
    "introduction": "Introduction",
}


class IntegratorAgent(BaseAgent):
    """Determines figure placement and generates paper references."""

    name = "figure_integrator"

    def __init__(self, llm: Any) -> None:
        super().__init__(llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Generate figure manifest and markdown references.

        Context keys:
            rendered (list[dict]): Successfully rendered figures with
                'figure_id', 'output_path', 'title', 'caption', 'section'
            topic (str): Research topic
            output_dir (str|Path): Charts directory
        """
        try:
            rendered = context.get("rendered", [])
            topic = context.get("topic", "")
            output_dir = Path(context.get("output_dir", "charts"))

            # Filter to successfully rendered figures only
            successful = [r for r in rendered if r.get("success")]

            if not successful:
                return self._make_result(
                    True,
                    data={"manifest": [], "markdown_refs": "", "figure_count": 0},
                )

            # Build manifest
            manifest = self._build_manifest(successful, output_dir)

            # Generate markdown references for paper embedding
            markdown_refs = self._generate_markdown_refs(manifest)

            # Generate figure descriptions for paper writing prompt
            figure_descriptions = self._generate_descriptions(manifest)

            # Save manifest
            manifest_path = output_dir / "figure_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.logger.info(
                "Generated figure manifest: %d figures", len(manifest)
            )

            return self._make_result(
                True,
                data={
                    "manifest": manifest,
                    "markdown_refs": markdown_refs,
                    "figure_descriptions": figure_descriptions,
                    "figure_count": len(manifest),
                    "manifest_path": str(manifest_path),
                },
            )
        except Exception as exc:
            self.logger.error("Integrator failed: %s", exc)
            return self._make_result(False, error=str(exc))

    # ------------------------------------------------------------------
    # Manifest building
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        rendered: list[dict[str, Any]],
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Build a structured manifest of all figures."""
        manifest: list[dict[str, Any]] = []

        # Sort by priority: priority 1 (must-have) first
        sorted_figs = sorted(
            rendered,
            key=lambda f: (
                self._section_order(f.get("section", "results")),
                f.get("priority", 2),
            ),
        )

        for i, fig in enumerate(sorted_figs, 1):
            figure_id = fig.get("figure_id", f"fig_{i}")
            output_path = fig.get("output_path", "")
            section = fig.get("section", "results")
            paper_section = _SECTION_MAP.get(section.lower(), "Results")

            # Relative path for paper embedding (charts/filename.png)
            if output_path:
                rel_path = f"charts/{Path(output_path).name}"
            else:
                rel_path = f"charts/{figure_id}.png"

            entry = {
                "figure_number": i,
                "figure_id": figure_id,
                "file_path": rel_path,
                "absolute_path": output_path,
                "title": fig.get("title", f"Figure {i}"),
                "caption": fig.get("caption", ""),
                "paper_section": paper_section,
                "width": fig.get("width", "single_column"),
                "label": f"fig:{figure_id}",
                "script_path": fig.get("script_path", ""),
            }
            manifest.append(entry)

        return manifest

    @staticmethod
    def _section_order(section: str) -> int:
        """Order sections for figure numbering."""
        order = {
            "introduction": 0,
            "method": 1,
            "methods": 1,
            "methodology": 1,
            "architecture": 1,
            "results": 2,
            "experiment": 2,
            "experiments": 2,
            "ablation": 3,
            "analysis": 4,
            "discussion": 5,
        }
        return order.get(section.lower(), 3)

    # ------------------------------------------------------------------
    # Markdown reference generation
    # ------------------------------------------------------------------

    def _generate_markdown_refs(
        self, manifest: list[dict[str, Any]]
    ) -> str:
        """Generate markdown image references for paper embedding."""
        refs: list[str] = []

        for entry in manifest:
            fig_num = entry["figure_number"]
            file_path = entry["file_path"]
            caption = entry.get("caption") or entry.get("title", f"Figure {fig_num}")
            refs.append(
                f"![Figure {fig_num}: {caption}]({file_path})"
            )

        return "\n\n".join(refs)

    # ------------------------------------------------------------------
    # Description generation for paper writing prompt
    # ------------------------------------------------------------------

    def _generate_descriptions(
        self, manifest: list[dict[str, Any]]
    ) -> str:
        """Generate figure descriptions for injection into paper writing prompt."""
        parts: list[str] = []
        parts.append("## AVAILABLE FIGURES (embed in the paper)")
        parts.append(
            "The following figures were generated from actual experiment data. "
            "Reference them in the appropriate paper sections using markdown "
            "image syntax: `![Caption](charts/filename.png)`\n"
        )

        for entry in manifest:
            fig_num = entry["figure_number"]
            file_path = entry["file_path"]
            title = entry.get("title", "")
            caption = entry.get("caption", "")
            section = entry.get("paper_section", "Results")

            parts.append(
                f"**Figure {fig_num}** (`{file_path}`) — {title}\n"
                f"  Caption: {caption}\n"
                f"  Place in: **{section}** section\n"
            )

        parts.append(
            "\nFor each figure referenced, write a descriptive caption and "
            "discuss what the figure shows in 2-3 sentences.\n"
        )

        return "\n".join(parts)
