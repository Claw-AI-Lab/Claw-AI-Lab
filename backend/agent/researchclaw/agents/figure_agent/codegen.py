"""CodeGen Agent — generates visualization code for each figure.

Takes the Planner's figure specifications and experiment data, then
generates either:
  - Standalone Python scripts (Matplotlib/Seaborn) — run by Renderer
  - LaTeX code (TikZ/PGFPlots) — embedded directly in the paper

Architecture follows Visual ChatGPT (Wu et al., 2023): the LLM acts
as a *controller* calling deterministic render tools instead of
generating pixels directly.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult
from researchclaw.agents.figure_agent.style_config import get_style_preamble
from researchclaw.utils.sanitize import sanitize_figure_id
from researchclaw.utils.thinking_tags import strip_thinking_tags

logger = logging.getLogger(__name__)


def _esc(s: str) -> str:
    """Escape curly braces in user-provided strings for str.format()."""
    return s.replace("{", "{{").replace("}", "}}")


def _shorten_label(label: str, max_chars: int = 20) -> str:
    """Shorten a condition label for tick marks (e.g. 'spectral_adaptive_manipulation' → 'Spectral Adapt. Manip.')."""
    words = label.replace("_", " ").split()
    result = []
    length = 0
    for w in words:
        if length + len(w) + 1 > max_chars and result:
            if len(w) > 4:
                result.append(w[:4].capitalize() + ".")
            else:
                result.append(w.capitalize())
            break
        result.append(w.capitalize())
        length += len(w) + 1
    return " ".join(result) if result else label[:max_chars]


# ---------------------------------------------------------------------------
# Built-in chart templates
# ---------------------------------------------------------------------------

_TEMPLATE_BAR_COMPARISON = '''
{style_preamble}
import textwrap

def _shorten(label, max_chars=18):
    words = label.replace("_", " ").split()
    result, length = [], 0
    for w in words:
        if length + len(w) + 1 > max_chars and result:
            result.append(w[:4].capitalize() + "." if len(w) > 4 else w.capitalize())
            break
        result.append(w.capitalize())
        length += len(w) + 1
    return " ".join(result) if result else label[:max_chars]

# Data
conditions = {conditions}
values = {values}
ci_low = {ci_low}
ci_high = {ci_high}

# Plot
n = len(conditions)
fig_w = max({width}, n * 1.1 + 1.0)
fig, ax = plt.subplots(figsize=(fig_w, {height}))
x = np.arange(n)
bar_colors = [COLORS[i % len(COLORS)] for i in range(n)]

yerr_lo = [max(0, v - lo) for v, lo in zip(values, ci_low)]
yerr_hi = [max(0, hi - v) for v, hi in zip(values, ci_high)]
has_error = any(lo > 1e-9 or hi > 1e-9 for lo, hi in zip(yerr_lo, yerr_hi))

bars = ax.bar(x, values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
if has_error:
    ax.errorbar(x, values, yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="#333", capsize=4, capthick=1.2, linewidth=1.2)

# Value labels — position above bar (or error bar)
y_range = max(values) - min(min(values), 0) if values else 1
offset = y_range * 0.03
for i, v in enumerate(values):
    top = v + (yerr_hi[i] if has_error else 0)
    fmt = f"{{v:.2f}}" if abs(v) >= 0.005 or v == 0 else f"{{v:.4f}}"
    ax.text(i, top + offset, fmt, ha="center", va="bottom", fontsize=8)

ax.set_xlabel("{x_label}")
ax.set_ylabel("{y_label}")
ax.set_title("{title}")
ax.set_xticks(x)
short_labels = [_shorten(c) for c in conditions]
ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.margins(y=0.15)
fig.tight_layout()
fig.savefig("{output_path}")
plt.close(fig)
print(f"Saved: {output_path}")
'''

_TEMPLATE_GROUPED_BAR = '''
{style_preamble}

def _shorten(label, max_chars=18):
    words = label.replace("_", " ").split()
    result, length = [], 0
    for w in words:
        if length + len(w) + 1 > max_chars and result:
            result.append(w[:4].capitalize() + "." if len(w) > 4 else w.capitalize())
            break
        result.append(w.capitalize())
        length += len(w) + 1
    return " ".join(result) if result else label[:max_chars]

# Data: conditions x metrics
conditions = {conditions}
metric_names = {metric_names}
# data_matrix[i][j] = value for condition i, metric j
data_matrix = {data_matrix}

# Plot
n_groups = len(conditions)
n_bars = len(metric_names)
fig_w = max({width}, n_groups * (n_bars * 0.4 + 0.6) + 1.5)
fig, ax = plt.subplots(figsize=(fig_w, {height}))
x = np.arange(n_groups)
bar_width = 0.8 / max(n_bars, 1)

for j, metric in enumerate(metric_names):
    offset = (j - n_bars / 2 + 0.5) * bar_width
    vals = [data_matrix[i][j] for i in range(n_groups)]
    ax.bar(x + offset, vals, bar_width, label=metric.replace("_", " ").title(),
           color=COLORS[j % len(COLORS)], alpha=0.85, edgecolor="white", linewidth=0.5)

ax.set_xlabel("{x_label}")
ax.set_ylabel("{y_label}")
ax.set_title("{title}")
ax.set_xticks(x)
short_labels = [_shorten(c) for c in conditions]
ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
ax.legend(framealpha=0.9, edgecolor="gray", fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.margins(y=0.10)
fig.tight_layout()
fig.savefig("{output_path}")
plt.close(fig)
print(f"Saved: {output_path}")
'''

_TEMPLATE_TRAINING_CURVE = '''
{style_preamble}

# Data: each series is (label, epochs, values, [optional std])
series_data = {series_data}

fig, ax = plt.subplots(figsize=({width}, {height}))

for idx, series in enumerate(series_data):
    label = series["label"]
    epochs = series["epochs"]
    values = series["values"]
    color = COLORS[idx % len(COLORS)]
    ls = LINE_STYLES[idx % len(LINE_STYLES)]
    marker = MARKERS[idx % len(MARKERS)]

    ax.plot(epochs, values, linestyle=ls, color=color, linewidth=1.5,
            marker=marker, markersize=4, markevery=max(1, len(epochs)//10),
            label=label.replace("_", " "))

    if "std" in series and series["std"]:
        std = series["std"]
        lower = [v - s for v, s in zip(values, std)]
        upper = [v + s for v, s in zip(values, std)]
        ax.fill_between(epochs, lower, upper, alpha=0.15, color=color)

ax.set_xlabel("{x_label}")
ax.set_ylabel("{y_label}")
ax.set_title("{title}")
ax.legend(framealpha=0.9, edgecolor="gray")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("{output_path}")
plt.close(fig)
print(f"Saved: {output_path}")
'''

_TEMPLATE_HEATMAP = '''
{style_preamble}

def _shorten(label, max_chars=16):
    words = label.replace("_", " ").split()
    result, length = [], 0
    for w in words:
        if length + len(w) + 1 > max_chars and result:
            result.append(w[:4].capitalize() + "." if len(w) > 4 else w.capitalize())
            break
        result.append(w.capitalize())
        length += len(w) + 1
    return " ".join(result) if result else label[:max_chars]

# Data
row_labels = {row_labels}
col_labels = {col_labels}
data = np.array({data_matrix})

n_rows, n_cols = data.shape
fig_w = max({width}, n_cols * 1.0 + 2.0)
fig_h = max({height}, n_rows * 0.7 + 1.5)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
im = ax.imshow(data, cmap="cividis", aspect="auto")

ax.set_xticks(np.arange(n_cols))
ax.set_yticks(np.arange(n_rows))
ax.set_xticklabels([_shorten(c) for c in col_labels], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([_shorten(r) for r in row_labels], fontsize=8)

# Annotate cells
for i in range(n_rows):
    for j in range(n_cols):
        val = data[i, j]
        color = "white" if val > (data.max() + data.min()) / 2 else "black"
        fmt = f"{{val:.2f}}" if abs(val) >= 0.005 or val == 0 else f"{{val:.4f}}"
        ax.text(j, i, fmt, ha="center", va="center", color=color, fontsize=8)

ax.set_xlabel("{x_label}")
ax.set_ylabel("{y_label}")
ax.set_title("{title}")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("{output_path}")
plt.close(fig)
print(f"Saved: {output_path}")
'''

_TEMPLATE_LINE_MULTI = '''
{style_preamble}

# Data: list of series dicts with label, x, y, [std]
series_data = {series_data}

fig, ax = plt.subplots(figsize=({width}, {height}))

for idx, series in enumerate(series_data):
    label = series["label"]
    x = series["x"]
    y = series["y"]
    color = COLORS[idx % len(COLORS)]
    ls = LINE_STYLES[idx % len(LINE_STYLES)]
    marker = MARKERS[idx % len(MARKERS)]

    ax.plot(x, y, linestyle=ls, color=color, linewidth=1.5,
            marker=marker, markersize=4, markevery=max(1, len(x)//8),
            label=label.replace("_", " "))

    if "std" in series and series["std"]:
        std = series["std"]
        lower = [v - s for v, s in zip(y, std)]
        upper = [v + s for v, s in zip(y, std)]
        ax.fill_between(x, lower, upper, alpha=0.15, color=color)

ax.set_xlabel("{x_label}")
ax.set_ylabel("{y_label}")
ax.set_title("{title}")
ax.legend(framealpha=0.9, edgecolor="gray")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("{output_path}")
plt.close(fig)
print(f"Saved: {output_path}")
'''

_TEMPLATE_SCATTER = '''
{style_preamble}

# Data: list of groups with label, x, y
groups = {groups}

fig, ax = plt.subplots(figsize=({width}, {height}))

for idx, group in enumerate(groups):
    label = group["label"]
    x = group["x"]
    y = group["y"]
    color = COLORS[idx % len(COLORS)]
    marker = MARKERS[idx % len(MARKERS)]
    ax.scatter(x, y, c=color, marker=marker, s=40, alpha=0.7, label=label.replace("_", " "))

ax.set_xlabel("{x_label}")
ax.set_ylabel("{y_label}")
ax.set_title("{title}")
ax.legend(framealpha=0.9, edgecolor="gray")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("{output_path}")
plt.close(fig)
print(f"Saved: {output_path}")
'''

_TEMPLATES: dict[str, str] = {
    "bar_comparison": _TEMPLATE_BAR_COMPARISON,
    "ablation_grouped": _TEMPLATE_BAR_COMPARISON,  # Same template, different data
    "grouped_bar": _TEMPLATE_GROUPED_BAR,
    "training_curve": _TEMPLATE_TRAINING_CURVE,
    "loss_curve": _TEMPLATE_TRAINING_CURVE,
    "heatmap": _TEMPLATE_HEATMAP,
    "confusion_matrix": _TEMPLATE_HEATMAP,
    "line_multi": _TEMPLATE_LINE_MULTI,
    "scatter_plot": _TEMPLATE_SCATTER,
}

# ---------------------------------------------------------------------------
# LaTeX / PGFPlots templates — for direct LaTeX embedding
# ---------------------------------------------------------------------------

_LATEX_TEMPLATE_BAR_COMPARISON = r'''
\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    ybar,
    bar width=15pt,
    width={width}cm,
    height={height}cm,
    xlabel={{{x_label}}},
    ylabel={{{y_label}}},
    title={{{title}}},
    symbolic x coords={{{x_coords}}},
    xtick=data,
    x tick label style={{rotate=25, anchor=east, font=\small}},
    ymin=0,
    nodes near coords,
    nodes near coords align={{vertical}},
    every node near coord/.append style={{font=\tiny}},
    grid=major,
    grid style={{dashed, gray!30}},
]
\addplot[fill=blue!60, draw=blue!80] coordinates {{{coords}}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{{caption}}}
\label{{fig:{figure_id}}}
\end{{figure}}
'''

_LATEX_TEMPLATE_LINE = r'''
\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width={width}cm,
    height={height}cm,
    xlabel={{{x_label}}},
    ylabel={{{y_label}}},
    title={{{title}}},
    legend pos=north west,
    grid=major,
    grid style={{dashed, gray!30}},
    cycle list name=color list,
]
{plot_commands}
\end{{axis}}
\end{{tikzpicture}}
\caption{{{caption}}}
\label{{fig:{figure_id}}}
\end{{figure}}
'''

_LATEX_TEMPLATE_HEATMAP = r'''
\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    colormap/viridis,
    colorbar,
    width={width}cm,
    height={height}cm,
    xlabel={{{x_label}}},
    ylabel={{{y_label}}},
    title={{{title}}},
    point meta min={meta_min},
    point meta max={meta_max},
    xtick={{{xtick}}},
    ytick={{{ytick}}},
    xticklabels={{{xticklabels}}},
    yticklabels={{{yticklabels}}},
    x tick label style={{rotate=45, anchor=east, font=\small}},
]
\addplot[matrix plot*, mesh/cols={cols}, mesh/rows={rows},
    point meta=explicit] coordinates {{
{matrix_coords}
}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{{caption}}}
\label{{fig:{figure_id}}}
\end{{figure}}
'''

_LATEX_TEMPLATES: dict[str, str] = {
    "bar_comparison": _LATEX_TEMPLATE_BAR_COMPARISON,
    "ablation_grouped": _LATEX_TEMPLATE_BAR_COMPARISON,
    "training_curve": _LATEX_TEMPLATE_LINE,
    "loss_curve": _LATEX_TEMPLATE_LINE,
    "line_multi": _LATEX_TEMPLATE_LINE,
    "heatmap": _LATEX_TEMPLATE_HEATMAP,
    "confusion_matrix": _LATEX_TEMPLATE_HEATMAP,
}


class CodeGenAgent(BaseAgent):
    """Generates visualization code (Python or LaTeX) for each planned figure.

    Supports two output formats:
      - ``"python"`` (default): Matplotlib/Seaborn scripts executed by Renderer
      - ``"latex"``: TikZ/PGFPlots code embedded directly in the paper
    """

    name = "figure_codegen"

    def __init__(self, llm: Any, *, output_format: str = "python") -> None:
        super().__init__(llm)
        self._output_format = output_format  # "python" or "latex"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Generate plotting scripts for all planned figures.

        Context keys:
            figures (list[dict]): Figure plan from Planner
            experiment_results (dict): Raw experiment data
            condition_summaries (dict): Per-condition aggregated stats
            metrics_summary (dict): Per-metric aggregated stats
            metric_key (str): Primary metric name
            output_dir (str): Directory for output scripts
            critic_feedback (list[dict], optional): Previous Critic feedback
        """
        try:
            figures = context.get("figures", [])
            experiment_results = context.get("experiment_results", {})
            condition_summaries = context.get("condition_summaries", {})
            metrics_summary = context.get("metrics_summary", {})
            metric_key = context.get("metric_key", "primary_metric")
            output_dir = context.get("output_dir", "charts")
            critic_feedback = context.get("critic_feedback", [])

            scripts: list[dict[str, Any]] = []

            for fig_spec in figures:
                # BUG-36: skip non-dict entries (LLM may return strings)
                if not isinstance(fig_spec, dict):
                    self.logger.warning("Skipping non-dict fig_spec: %s", type(fig_spec))
                    continue
                figure_id = fig_spec.get("figure_id", "unknown")
                chart_type = fig_spec.get("chart_type", "bar_comparison")

                # Check for critic feedback on this specific figure
                fig_feedback = None
                for fb in critic_feedback:
                    # BUG-FIX: guard against non-dict entries in feedback
                    if isinstance(fb, dict) and fb.get("figure_id") == figure_id:
                        fig_feedback = fb
                        break

                script = self._generate_script(
                    fig_spec=fig_spec,
                    chart_type=chart_type,
                    condition_summaries=condition_summaries,
                    metrics_summary=metrics_summary,
                    experiment_results=experiment_results,
                    metric_key=metric_key,
                    output_dir=output_dir,
                    critic_feedback=fig_feedback,
                )

                scripts.append({
                    "figure_id": figure_id,
                    "chart_type": chart_type,
                    "script": script,
                    "output_filename": f"{figure_id}.png",
                    "title": fig_spec.get("title", ""),
                    "caption": fig_spec.get("caption", ""),
                    "section": fig_spec.get("section", "results"),
                    "width": fig_spec.get("width", "single_column"),
                })

            return self._make_result(True, data={"scripts": scripts})
        except Exception as exc:
            self.logger.error("CodeGen failed: %s", exc)
            return self._make_result(False, error=str(exc))

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def _generate_script(
        self,
        *,
        fig_spec: dict[str, Any],
        chart_type: str,
        condition_summaries: dict[str, Any],
        metrics_summary: dict[str, Any],
        experiment_results: dict[str, Any],
        metric_key: str,
        output_dir: str,
        critic_feedback: dict[str, Any] | None,
    ) -> str:
        """Generate a plotting script for a single figure."""
        figure_id = sanitize_figure_id(fig_spec.get("figure_id", "figure"))
        # BUG-20: Use absolute path to avoid CWD-relative savefig errors
        output_path = str((Path(output_dir) / f"{figure_id}.png").resolve())
        title = fig_spec.get("title", "")
        x_label = fig_spec.get("x_label", "")
        y_label = fig_spec.get("y_label", "")
        width_key = fig_spec.get("width", "single_column")
        # BUG-FIX: LLM may return data_source as a plain string (e.g.
        # "condition_comparison") instead of a dict.  Normalize to dict.
        _raw_ds = fig_spec.get("data_source", {})
        if isinstance(_raw_ds, str):
            data_source = {"type": _raw_ds}
        elif isinstance(_raw_ds, dict):
            data_source = _raw_ds
        else:
            data_source = {}

        from researchclaw.agents.figure_agent.style_config import FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT
        width = FIGURE_WIDTH.get(width_key, FIGURE_WIDTH["single_column"])
        height = DEFAULT_FIGURE_HEIGHT

        # Try template-based generation first
        template = _TEMPLATES.get(chart_type)
        if template and not critic_feedback:
            script = self._fill_template(
                template=template,
                chart_type=chart_type,
                data_source=data_source,
                condition_summaries=condition_summaries,
                metrics_summary=metrics_summary,
                experiment_results=experiment_results,
                metric_key=metric_key,
                output_path=output_path,
                title=title,
                x_label=x_label,
                y_label=y_label,
                width=width,
                height=height,
            )
            if script:
                return script

        # Fall back to LLM-generated script
        return self._llm_generate_script(
            fig_spec=fig_spec,
            chart_type=chart_type,
            condition_summaries=condition_summaries,
            metrics_summary=metrics_summary,
            experiment_results=experiment_results,
            metric_key=metric_key,
            output_path=output_path,
            width=width,
            height=height,
            critic_feedback=critic_feedback,
        )

    def _fill_template(
        self,
        *,
        template: str,
        chart_type: str,
        data_source: dict[str, Any],
        condition_summaries: dict[str, Any],
        metrics_summary: dict[str, Any],
        experiment_results: dict[str, Any],
        metric_key: str,
        output_path: str,
        title: str,
        x_label: str,
        y_label: str,
        width: float,
        height: float,
    ) -> str:
        """Fill a template with actual data values."""
        style_preamble = get_style_preamble()
        source_type = data_source.get("type", "condition_comparison")

        if chart_type in ("bar_comparison", "ablation_grouped"):
            return self._fill_bar_template(
                template=template,
                condition_summaries=condition_summaries,
                metric_key=data_source.get("metric", metric_key),
                output_path=output_path,
                title=title,
                x_label=x_label,
                y_label=y_label,
                width=width,
                height=height,
                style_preamble=style_preamble,
            )

        if chart_type == "grouped_bar" and source_type == "multi_metric":
            # BUG-37: LLM may return nested lists in metrics — flatten to list[str]
            _raw_metrics = data_source.get("metrics", [])
            _flat_metrics: list[str] = []
            for _mi in (_raw_metrics if isinstance(_raw_metrics, list) else []):
                if isinstance(_mi, str):
                    _flat_metrics.append(_mi)
                elif isinstance(_mi, list):
                    _flat_metrics.extend(str(x) for x in _mi)
                else:
                    _flat_metrics.append(str(_mi))
            return self._fill_grouped_bar_template(
                template=template,
                condition_summaries=condition_summaries,
                metrics=_flat_metrics,
                output_path=output_path,
                title=title,
                x_label=x_label,
                y_label=y_label,
                width=width,
                height=height,
                style_preamble=style_preamble,
            )

        if chart_type in ("heatmap", "confusion_matrix"):
            return self._fill_heatmap_template(
                template=template,
                condition_summaries=condition_summaries,
                metrics_summary=metrics_summary,
                output_path=output_path,
                title=title,
                x_label=x_label,
                y_label=y_label,
                width=width,
                height=height,
                style_preamble=style_preamble,
            )

        # For other types, fall through to LLM generation
        return ""

    def _fill_bar_template(
        self,
        *,
        template: str,
        condition_summaries: dict[str, Any],
        metric_key: str,
        output_path: str,
        title: str,
        x_label: str,
        y_label: str,
        width: float,
        height: float,
        style_preamble: str,
    ) -> str:
        """Fill bar comparison template with condition data."""
        conditions: list[str] = []
        values: list[float] = []
        ci_low: list[float] = []
        ci_high: list[float] = []

        for cond, cdata in condition_summaries.items():
            if not isinstance(cdata, dict):
                continue
            metrics = cdata.get("metrics", {})
            val = metrics.get(f"{metric_key}_mean") or metrics.get(metric_key)
            if val is None:
                continue
            try:
                fval = float(val)
            except (ValueError, TypeError):
                continue

            conditions.append(cond)
            values.append(fval)
            raw_lo = cdata.get("ci95_low")
            raw_hi = cdata.get("ci95_high")
            ci_low.append(float(raw_lo) if raw_lo is not None else fval)
            ci_high.append(float(raw_hi) if raw_hi is not None else fval)

        if not conditions:
            return ""

        return template.format(
            style_preamble=style_preamble,
            conditions=repr(conditions),
            values=repr(values),
            ci_low=repr(ci_low),
            ci_high=repr(ci_high),
            output_path=output_path,
            title=_esc(title),
            x_label=_esc(x_label),
            y_label=_esc(y_label),
            width=width,
            height=height,
        )

    def _fill_grouped_bar_template(
        self,
        *,
        template: str,
        condition_summaries: dict[str, Any],
        metrics: list[str],
        output_path: str,
        title: str,
        x_label: str,
        y_label: str,
        width: float,
        height: float,
        style_preamble: str,
    ) -> str:
        """Fill grouped bar template with multi-metric data."""
        conditions: list[str] = list(condition_summaries.keys())
        if not conditions or not metrics:
            return ""

        data_matrix: list[list[float]] = []
        for cond in conditions:
            cdata = condition_summaries.get(cond, {})
            cmetrics = cdata.get("metrics", {}) if isinstance(cdata, dict) else {}
            row = []
            for m in metrics:
                val = cmetrics.get(f"{m}_mean") or cmetrics.get(m, 0)
                try:
                    row.append(float(val))
                except (ValueError, TypeError):
                    row.append(0.0)
            data_matrix.append(row)

        return template.format(
            style_preamble=style_preamble,
            conditions=repr(conditions),
            metric_names=repr(metrics),
            data_matrix=repr(data_matrix),
            output_path=output_path,
            title=_esc(title),
            x_label=_esc(x_label),
            y_label=_esc(y_label),
            width=width,
            height=height,
        )

    def _fill_heatmap_template(
        self,
        *,
        template: str,
        condition_summaries: dict[str, Any],
        metrics_summary: dict[str, Any],
        output_path: str,
        title: str,
        x_label: str,
        y_label: str,
        width: float,
        height: float,
        style_preamble: str,
    ) -> str:
        """Fill heatmap template — rows=conditions, cols=metrics."""
        conditions = list(condition_summaries.keys())
        # Select non-timing metrics
        metric_names = [
            m for m in metrics_summary
            if not any(t in m.lower() for t in ["time", "elapsed", "seed", "runtime"])
        ][:8]

        if not conditions or not metric_names:
            return ""

        data_matrix: list[list[float]] = []
        for cond in conditions:
            cdata = condition_summaries.get(cond, {})
            cmetrics = cdata.get("metrics", {}) if isinstance(cdata, dict) else {}
            row = []
            for m in metric_names:
                val = cmetrics.get(f"{m}_mean") or cmetrics.get(m, 0)
                try:
                    row.append(round(float(val), 4))
                except (ValueError, TypeError):
                    row.append(0.0)
            data_matrix.append(row)

        return template.format(
            style_preamble=style_preamble,
            row_labels=repr(conditions),
            col_labels=repr(metric_names),
            data_matrix=repr(data_matrix),
            output_path=output_path,
            title=_esc(title),
            x_label=_esc(x_label or "Metric"),
            y_label=_esc(y_label or "Method"),
            width=max(width, len(metric_names) * 0.8),
            height=max(height, len(conditions) * 0.6),
        )

    # ------------------------------------------------------------------
    # LLM-based script generation
    # ------------------------------------------------------------------

    def _llm_generate_script(
        self,
        *,
        fig_spec: dict[str, Any],
        chart_type: str,
        condition_summaries: dict[str, Any],
        metrics_summary: dict[str, Any],
        experiment_results: dict[str, Any],
        metric_key: str,
        output_path: str,
        width: float,
        height: float,
        critic_feedback: dict[str, Any] | None,
    ) -> str:
        """Generate a plotting script using LLM."""
        if self._output_format == "latex":
            return self._llm_generate_latex(
                fig_spec=fig_spec,
                chart_type=chart_type,
                condition_summaries=condition_summaries,
                metrics_summary=metrics_summary,
                metric_key=metric_key,
                width=width,
                height=height,
                critic_feedback=critic_feedback,
            )

        style_preamble = get_style_preamble()

        system_prompt = (
            "You are an expert scientific visualization programmer. "
            "Generate a standalone Python script that creates a publication-quality "
            "matplotlib chart.\n\n"
            "RULES:\n"
            "- The script must be completely self-contained (no external imports "
            "beyond matplotlib, numpy, seaborn)\n"
            "- Use ONLY the exact data values provided below. NEVER generate "
            "synthetic/random data (no np.random, no fake distributions). "
            "If only mean±std are available, plot those directly as bar+errorbar.\n"
            "- Use the provided style preamble at the top of the script\n"
            "- Output format: PNG at 300 DPI\n"
            "- Use colorblind-safe colors from the COLORS list\n"
            "- Include descriptive axis labels and title\n"
            "- Call fig.savefig() and plt.close(fig) at the end\n"
            "- Print 'Saved: <path>' after saving\n"
            "- Do NOT include any <think> or </think> tags\n\n"
            "LAYOUT RULES (prevent text overlap):\n"
            "- figsize width: at least (number_of_conditions × 1.1 + 1.0) inches, "
            "minimum 3.5 inches\n"
            "- figsize height: at least 3.5 inches\n"
            "- X-tick labels: rotation=40, ha='right', fontsize=8. "
            "Shorten long names (>18 chars) by abbreviating words.\n"
            "- Value annotations: fontsize=8, avoid overlapping bars\n"
            "- Always call fig.tight_layout() before savefig\n\n"
            "Return ONLY the Python script, no explanation."
        )

        # Build data context (truncated to avoid token overflow)
        data_context = {
            "conditions": list(condition_summaries.keys())[:10],
            "metric_key": metric_key,
        }
        # Add condition values
        for cond, cdata in list(condition_summaries.items())[:10]:
            if isinstance(cdata, dict):
                data_context[cond] = {
                    "metrics": {k: v for k, v in (cdata.get("metrics") or {}).items()
                                if not any(t in k.lower()
                                           for t in ["time", "elapsed", "runtime"])},
                    "ci95_low": cdata.get("ci95_low"),
                    "ci95_high": cdata.get("ci95_high"),
                }

        user_prompt = (
            f"Style preamble (paste at top of script):\n```python\n{style_preamble}\n```\n\n"
            f"Figure specification:\n{json.dumps(fig_spec, indent=2)}\n\n"
            f"Experiment data:\n{json.dumps(data_context, indent=2, default=str)}\n\n"
            f"Output path: {output_path}\n"
            f"Figure size: ({width}, {height})\n"
        )

        if critic_feedback:
            user_prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED REVIEW. Fix these issues:\n"
                f"{json.dumps(critic_feedback.get('issues', []), indent=2)}\n"
            )

        raw = self._chat(system_prompt, user_prompt, max_tokens=4096, temperature=0.3)

        # Strip reasoning model thinking tags before parsing
        raw = strip_thinking_tags(raw)

        # Strip markdown fences
        script = self._strip_fences(raw)

        # Ensure style preamble is present
        if "matplotlib" not in script:
            script = style_preamble + "\n\n" + script

        return script

    def _llm_generate_latex(
        self,
        *,
        fig_spec: dict[str, Any],
        chart_type: str,
        condition_summaries: dict[str, Any],
        metrics_summary: dict[str, Any],
        metric_key: str,
        width: float,
        height: float,
        critic_feedback: dict[str, Any] | None,
    ) -> str:
        """Generate LaTeX TikZ/PGFPlots code for a figure.

        This produces code that compiles directly in a LaTeX document that
        includes ``\\usepackage{pgfplots}`` and ``\\usepackage{tikz}``.
        """
        system_prompt = (
            "You are an expert scientific visualization programmer specializing "
            "in LaTeX/TikZ/PGFPlots.\n\n"
            "Generate LaTeX code using PGFPlots that creates a publication-quality "
            "chart suitable for a top-tier AI conference paper.\n\n"
            "RULES:\n"
            "- Use pgfplots (version ≥ 1.18) with \\pgfplotsset{compat=1.18}\n"
            "- All data values must be hardcoded in the LaTeX source\n"
            "- Use the colorbrewer palette or viridis colormap\n"
            "- Include descriptive axis labels and title\n"
            "- Wrap in a figure environment with \\caption and \\label\n"
            "- Font sizes should match: title 12pt, labels 10pt, ticks 9pt\n"
            "- Width should be \\columnwidth or 0.48\\textwidth for single column\n"
            "- Do NOT include any <think> or </think> tags\n\n"
            "Return ONLY the LaTeX code, no explanation."
        )

        # Build data context
        data_context = {
            "conditions": list(condition_summaries.keys())[:10],
            "metric_key": metric_key,
        }
        for cond, cdata in list(condition_summaries.items())[:10]:
            if isinstance(cdata, dict):
                data_context[cond] = {
                    "metrics": {k: v for k, v in (cdata.get("metrics") or {}).items()
                                if not any(t in k.lower()
                                           for t in ["time", "elapsed", "runtime"])},
                }

        user_prompt = (
            f"Chart type: {chart_type}\n"
            f"Figure specification:\n{json.dumps(fig_spec, indent=2)}\n\n"
            f"Experiment data:\n{json.dumps(data_context, indent=2, default=str)}\n\n"
            f"Figure dimensions: width={width}in, height={height}in\n"
        )

        if critic_feedback:
            user_prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED REVIEW. Fix these issues:\n"
                f"{json.dumps(critic_feedback.get('issues', []), indent=2)}\n"
            )

        raw = self._chat(system_prompt, user_prompt, max_tokens=4096, temperature=0.3)

        # Strip reasoning model thinking tags before parsing
        raw = strip_thinking_tags(raw)

        # Strip markdown fences (```latex ... ```)
        return self._strip_latex_fences(raw)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences from LLM output."""
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _strip_latex_fences(text: str) -> str:
        """Remove markdown code fences from LaTeX LLM output."""
        m = re.search(r"```(?:latex|tex)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()
