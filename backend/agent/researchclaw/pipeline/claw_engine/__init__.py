"""Shared agentic engine for claw-code style turn loops.

Provides a generic ``AgentTurnLoop`` and sandboxed tool infrastructure
that can be used by any pipeline stage (S11 CODE_GENERATION,
S12 SANITY_CHECK, S14 EXPERIMENT_RUN, S15 ITERATIVE_REFINE, etc.).

Architecture ported from claw-code's ``ConversationRuntime``:
    user_message → (LLM call → tool execution →)* → done
"""

from researchclaw.pipeline.claw_engine.turn_loop import AgentTurnLoop, TurnResult
from researchclaw.pipeline.claw_engine.session import StageSession

__all__ = ["AgentTurnLoop", "TurnResult", "StageSession"]
