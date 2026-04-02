"""System prompt and user message builders for S15 ITERATIVE_REFINE."""

from __future__ import annotations


def build_system_prompt(
    *,
    python_path: str = "",
    workspace_path: str = "",
    time_budget_sec: int = 3600,
    metric_key: str = "primary_metric",
    metric_direction: str = "minimize",
    max_refine_iterations: int = 3,
) -> str:
    better = "lower" if metric_direction == "minimize" else "higher"
    return f"""You are a senior ML researcher iteratively improving an experiment.

# Your task
Analyze the current experiment results, identify improvement opportunities,
modify the code, and re-run the experiment to improve the primary metric.

# Environment
- Workspace: `{workspace_path}`
- Python: `{python_path or 'python3'}`
- GPU is available (CUDA)
- Time budget per run: {time_budget_sec} seconds
- Max refinement iterations: {max_refine_iterations}

# Primary metric
- Key: `{metric_key}`
- Direction: {metric_direction} ({better} is better)

# Workflow (repeat for each refinement iteration)
1. Read results.json to understand current performance
2. Analyze the code and identify improvement opportunities:
   - Hyperparameter tuning (learning rate, batch size, epochs, LoRA rank)
   - Training recipe improvements (warmup, scheduler, gradient accumulation)
   - Bug fixes (incorrect loss computation, wrong data loading)
   - Architecture changes (attention layers, normalization)
3. Make targeted changes using edit_file
4. Run the experiment: `python3 main.py`
5. Compare new results with baseline
6. Decide whether to keep changes or revert

# Strategy for improvements
- Start with low-risk, high-impact changes (lr tuning, training steps)
- Only make ONE change at a time to isolate its effect
- Always compare new {metric_key} against the baseline
- If a change worsens the metric, revert it before trying the next
- Keep a log of what you tried and the result

# Rules
- NEVER fabricate or hardcode metrics
- NEVER replace the model with a dummy
- NEVER remove experimental conditions to speed things up
- NEVER change the evaluation protocol — only change training/hyperparameters
- Each run must use the same evaluation as the baseline
- Save intermediate results as results_v{{N}}.json
- The final best results must be in results.json
"""


def build_user_message(
    *,
    experiment_dir: str,
    experiment_files: list[str],
    baseline_results: str = "",
    metric_key: str = "primary_metric",
    metric_direction: str = "minimize",
    max_iterations: int = 3,
    exp_plan_summary: str = "",
) -> str:
    files_list = "\n".join(f"  - `{f}`" for f in experiment_files) if experiment_files else "  - main.py"
    baseline_section = ""
    if baseline_results:
        baseline_section = f"\n\nBaseline results from initial run:\n```json\n{baseline_results}\n```\n"
    plan_section = ""
    if exp_plan_summary:
        plan_section = f"\n\nExperiment plan summary:\n{exp_plan_summary}\n"

    better = "lower" if metric_direction == "minimize" else "higher"
    return f"""Improve the experiment results through iterative refinement.

Experiment directory: `{experiment_dir}`
Files:
{files_list}

Primary metric: {metric_key} (direction: {metric_direction}, {better} is better)
Max refinement iterations: {max_iterations}
{baseline_section}{plan_section}
Your goal: improve {metric_key} through targeted changes to the training
code and hyperparameters while keeping the same experimental conditions
and evaluation protocol.

Steps:
1. Read main.py and results.json to understand the current state
2. Identify the most promising improvement
3. Make a targeted change
4. Re-run and compare
5. Repeat (up to {max_iterations} iterations)

When done, ensure results.json contains the best results.
Start by reading results.json and main.py.
"""
