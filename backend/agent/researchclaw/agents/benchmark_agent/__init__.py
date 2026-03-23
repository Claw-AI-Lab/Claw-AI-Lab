"""BenchmarkAgent — multi-agent benchmark, dataset, and baseline selection.

Architecture
------------
1. **Surveyor** — searches HuggingFace Hub + local knowledge base for
   domain-relevant benchmarks, datasets, and baseline methods.
2. **Selector** — filters and ranks candidates based on hardware constraints,
   time budget, network policy, and tier availability.
3. **Acquirer** — generates data-loading code snippets, ``setup.py`` download
   scripts, baseline boilerplate, and ``requirements.txt`` entries.
4. **Validator** — validates generated code for syntax correctness and
   API compatibility.

The ``BenchmarkOrchestrator`` coordinates the four agents and produces a
``BenchmarkPlan`` consumed by downstream pipeline stages (experiment design,
code generation).
"""

from researchclaw.agents.benchmark_agent.orchestrator import (
    BenchmarkOrchestrator,
    BenchmarkPlan,
)

__all__ = ["BenchmarkOrchestrator", "BenchmarkPlan"]
