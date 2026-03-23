"""FigureAgent — multi-agent intelligent chart generation system.

Architecture
------------
1. **Planner** — analyzes experiment results and determines which charts
   to generate, their types, layouts, and captions.
2. **CodeGen** — generates Python matplotlib plotting scripts using
   academic styling (SciencePlots, 300 DPI, colorblind-safe palettes).
3. **Renderer** — executes plotting scripts and verifies output files.
4. **Critic** — tri-modal review: numerical accuracy, text correctness,
   and visual quality assessment.
5. **Integrator** — determines figure placement in the paper and
   generates markdown references with captions.

The ``FigureOrchestrator`` coordinates all agents and produces a
``FigurePlan`` consumed by downstream pipeline stages (paper draft,
paper export).
"""

from researchclaw.agents.figure_agent.orchestrator import (
    FigureOrchestrator,
    FigurePlan,
)

__all__ = ["FigureOrchestrator", "FigurePlan"]
