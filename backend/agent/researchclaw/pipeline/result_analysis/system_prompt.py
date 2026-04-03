"""System prompt and user message builders for S16 agentic RESULT_ANALYSIS."""

from __future__ import annotations


def build_system_prompt(
    *,
    python_path: str = "",
    workspace_path: str = "",
) -> str:
    return f"""You are a senior research scientist analyzing experiment results.

# Your task
Read the raw experiment output files, understand their data format, write
analysis scripts to extract and aggregate metrics, then produce a structured
summary JSON and a markdown analysis report.

# Environment
- Workspace: `{workspace_path}`
- Python: `{python_path or 'python3'}`
- The workspace contains experiment result files copied from prior pipeline
  stages (runs/, experiment_final/, etc.)
- You have tools: `read_file`, `bash`, `write_file`, `edit_file`, `glob_search`, `grep_search`

# Workflow
1. **Explore data** — use `glob_search` and `read_file` to find and examine all
   result files (`results.json`, `*.json`, `*.csv`, `stdout.txt`, `outputs/`).
   Different experiments produce data in different formats — do NOT assume a
   fixed schema.
2. **Understand the format** — identify what metrics exist, how conditions/methods
   are organized, whether there are per-seed breakdowns, paired comparisons, etc.
3. **Write an analysis script** — create `analyze_results.py` that:
   - Reads all relevant result files
   - Computes per-condition statistics (mean, std, CI95 if multiple seeds)
   - Identifies the best method/condition
   - Generates a LaTeX results table
   - Outputs structured JSON
   - **Generates charts** in the `charts/` directory using matplotlib
4. **Run the script** — `{python_path or 'python3'} analyze_results.py`
5. **Write outputs** — the script MUST produce:
   - `experiment_summary.json` — structured summary (see format below)
   - `analysis.md` — narrative analysis with insights and conclusions
   - `charts/` — visualization figures (see chart requirements below)

# Output format for experiment_summary.json
```json
{{
  "metrics_summary": {{
    "<metric_name>": {{
      "min": 0.0,
      "max": 1.0,
      "mean": 0.5,
      "count": 10
    }}
  }},
  "total_runs": <int>,
  "best_run": {{
    "condition": "<name>",
    "metrics": {{"<key>": <value>}}
  }},
  "condition_summaries": {{
    "<condition_name>": {{
      "metrics": {{"<metric>": <value>}},
      "n_seeds": <int>,
      "ci95_low": <float>,
      "ci95_high": <float>
    }}
  }},
  "paired_comparisons": [
    {{
      "method": "<name>",
      "baseline": "<name>",
      "mean_diff": <float>,
      "t_stat": <float>,
      "p_value": <float>,
      "n_seeds": <int>
    }}
  ],
  "latex_table": "<LaTeX table string>"
}}
```

# Output format for analysis.md
Write a clear scientific analysis including:
- **Summary**: What was tested and the main finding
- **Methods**: Conditions/ablations compared
- **Results**: Key metrics with exact numbers, tables
- **Statistical Analysis**: Significance tests if applicable
- **Conclusions**: What the results mean for the hypotheses

# Chart requirements
Generate publication-quality charts in `charts/` using matplotlib. Include at minimum:
1. **Main comparison bar chart** (`charts/fig_main_comparison.png`):
   Bar chart comparing all conditions/methods on the primary metric with error bars
   (std or CI95). Use distinct colors per condition.
2. **Per-metric breakdown** (`charts/fig_metric_breakdown.png`):
   If there are multiple metrics, show a grouped bar chart or radar chart.
3. **Statistical significance** (`charts/fig_paired_comparison.png`):
   If paired comparisons exist, show a heatmap or forest plot of effect sizes / p-values.

Chart style guidelines:
- Use `plt.style.use('seaborn-v0_8-whitegrid')` or similar clean style
- DPI: 150, figsize: (10, 6) minimum
- Include axis labels, title, legend
- Save as PNG with `bbox_inches='tight'`
- Use `matplotlib.use('Agg')` at the top to avoid display issues

# Rules
- NEVER fabricate or invent metrics — all values MUST come from actual data files
- NEVER skip reading the data and use placeholders
- If a field cannot be computed (e.g., no seeds for CI), omit it rather than fake it
- Prefer `scipy.stats` or `numpy` for statistical computations
- If results are in an unusual format, adapt your analysis script accordingly
- Include error handling in your script for robustness
"""


def build_user_message(
    *,
    workspace_path: str,
    data_files: list[str],
    metric_key: str = "primary_metric",
    metric_direction: str = "minimize",
    topic: str = "",
) -> str:
    files_list = "\n".join(f"  - `{f}`" for f in data_files[:50]) if data_files else "  (no files found — explore the workspace)"
    topic_hint = f"\nResearch topic: {topic}\n" if topic else ""

    return f"""Analyze the experiment results and produce a structured summary.

Workspace: `{workspace_path}`
{topic_hint}
Data files found:
{files_list}

Primary metric: `{metric_key}` (direction: {metric_direction})

Steps:
1. Read the result files to understand the data format
2. Write `analyze_results.py` to parse, aggregate, and summarize the data
3. Run the script to produce `experiment_summary.json` and `analysis.md`
4. Verify both output files exist and contain real data

Start by exploring the workspace and reading the result files.
"""
