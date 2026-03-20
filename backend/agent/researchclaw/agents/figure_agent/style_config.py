"""Academic chart styling configuration for FigureAgent.

Defines global constants for chart styling that conform to AI conference
publication standards (IEEE, NeurIPS, ICML, ICLR).  Used by CodeGen Agent
when generating matplotlib plotting scripts.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

# SciencePlots style list — CodeGen Agent inserts this into generated scripts.
# Fallback: seaborn-v0_8-whitegrid if SciencePlots is not installed.
MATPLOTLIB_STYLES = ["science", "ieee"]
MATPLOTLIB_STYLES_FALLBACK = ["seaborn-v0_8-whitegrid"]

# Output resolution (DPI) — 300+ for publication, 150 for draft
DPI_PUBLICATION = 300
DPI_DRAFT = 150

# ---------------------------------------------------------------------------
# Font sizes (points)
# ---------------------------------------------------------------------------

FONT_SIZE = {
    "title": 12,
    "axis_label": 10,
    "tick": 9,
    "legend": 9,
    "annotation": 9,
}

# ---------------------------------------------------------------------------
# Figure dimensions (inches) — column-width aware
# ---------------------------------------------------------------------------

FIGURE_WIDTH = {
    "single_column": 3.5,   # IEEE / NeurIPS single column
    "double_column": 7.0,   # IEEE / NeurIPS double column
    "full_page": 7.0,       # Full width
}

DEFAULT_FIGURE_HEIGHT = 3.0  # reasonable default height

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Paul Tol's "bright" scheme)
# ---------------------------------------------------------------------------

COLORS_BRIGHT = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

# Extended palette for > 7 categories
COLORS_EXTENDED = COLORS_BRIGHT + [
    "#332288",  # indigo
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # dark green
    "#999933",  # olive
    "#CC6677",  # rose
    "#882255",  # wine
]

# ---------------------------------------------------------------------------
# Line and marker styles (for B&W printing compatibility)
# ---------------------------------------------------------------------------

LINE_STYLES = ["-", "--", "-.", ":"]
MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "*", "X"]

# ---------------------------------------------------------------------------
# Output format preferences
# ---------------------------------------------------------------------------

OUTPUT_FORMAT_PRIMARY = "pdf"      # Vector — preferred for publication
OUTPUT_FORMAT_FALLBACK = "png"     # Raster — for markdown embedding
OUTPUT_FORMATS = ["pdf", "png"]    # Generate both

# ---------------------------------------------------------------------------
# Chart type constants
# ---------------------------------------------------------------------------

CHART_TYPES = {
    "bar_comparison",
    "grouped_bar",
    "training_curve",
    "loss_curve",
    "heatmap",
    "confusion_matrix",
    "scatter_plot",
    "violin_box",
    "ablation_grouped",
    "line_multi",
    "radar_chart",
    "architecture_diagram",  # Placeholder — generated via description
}

# ---------------------------------------------------------------------------
# Style snippet for injection into generated scripts
# ---------------------------------------------------------------------------

STYLE_PREAMBLE = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Academic styling
try:
    plt.style.use({styles})
except Exception:
    try:
        plt.style.use({fallback})
    except Exception:
        pass  # Use default matplotlib style

# Colorblind-safe palette
COLORS = {colors}
LINE_STYLES = {line_styles}
MARKERS = {markers}

# Publication settings
plt.rcParams.update({{
    "font.size": {font_axis},
    "axes.titlesize": {font_title},
    "axes.labelsize": {font_axis},
    "xtick.labelsize": {font_tick},
    "ytick.labelsize": {font_tick},
    "legend.fontsize": {font_legend},
    "figure.dpi": {dpi},
    "savefig.dpi": {dpi},
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}})
'''.strip()


def get_style_preamble(*, dpi: int = DPI_PUBLICATION) -> str:
    """Return the style preamble string for injection into chart scripts."""
    return STYLE_PREAMBLE.format(
        styles=repr(MATPLOTLIB_STYLES),
        fallback=repr(MATPLOTLIB_STYLES_FALLBACK),
        colors=repr(COLORS_BRIGHT),
        line_styles=repr(LINE_STYLES),
        markers=repr(MARKER_STYLES),
        font_title=FONT_SIZE["title"],
        font_axis=FONT_SIZE["axis_label"],
        font_tick=FONT_SIZE["tick"],
        font_legend=FONT_SIZE["legend"],
        dpi=dpi,
    )
