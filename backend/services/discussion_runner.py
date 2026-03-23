#!/usr/bin/env python3
"""
L1 Discussion Runner — Multi-round LLM debate between agent perspectives.

Collects synthesis.md from multiple L1 agents who independently researched the
same topic, runs a structured multi-round debate via the LLM, and produces a
consensus synthesis that enriches all agents' context for hypothesis generation.

Usage:
    python discussion_runner.py \
        --config /path/to/config.arc.yaml \
        --synthesis-dirs /path/to/agent-a/stage-07 /path/to/agent-b/stage-07 \
        --output /path/to/discussion_output \
        --rounds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AGENT_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent / "agent")
if AGENT_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, AGENT_PACKAGE_DIR)

from researchclaw.config import RCConfig
from researchclaw.llm import create_llm_client


def _read_synthesis(stage_dir: Path) -> str | None:
    for name in ("synthesis.md", "synthesis.txt"):
        f = stage_dir / name
        if f.is_file():
            return f.read_text(encoding="utf-8")
    return None


def _label(index: int) -> str:
    return chr(ord("A") + index)


SYSTEM_PRESENT = """\
You are a senior research coordinator facilitating a multi-agent research discussion.
Multiple research agents have independently studied the same topic and produced their
own literature syntheses. Your task is to analyze each agent's perspective and identify:
1. Key findings unique to each perspective
2. Common themes across perspectives
3. Knowledge gaps that no perspective addressed
4. Contradictions or disagreements between perspectives

Be specific and cite which perspective (Agent A, Agent B, etc.) each point comes from.
Respond in the same language as the syntheses (Chinese if they are in Chinese)."""

SYSTEM_CRITIQUE = """\
You are a critical research reviewer. Given the initial analysis of multiple research
perspectives, your task is to:
1. Evaluate the strength of evidence for each key finding
2. Identify potential biases in individual perspectives
3. Find complementary findings that could be combined for stronger conclusions
4. Highlight the most promising research directions that emerge from combining perspectives
5. Note any methodological differences that explain contradictions

Be rigorous and constructive. Respond in the same language as the input."""

SYSTEM_CONSENSUS = """\
You are a research synthesis expert. Based on the multi-round discussion of independent
research perspectives, produce a unified consensus synthesis that:
1. Integrates the strongest findings from all perspectives
2. Resolves contradictions with reasoned explanations
3. Preserves novel insights that appeared in only one perspective
4. Identifies the most promising hypotheses suggested by the combined evidence
5. Notes remaining uncertainties and open questions

Format the output as a well-structured markdown document with clear sections.
This consensus will be used by all agents to generate research hypotheses.
Respond in the same language as the input."""


def run_discussion(
    llm,
    topic: str,
    syntheses: dict[str, str],
    num_rounds: int,
    output_dir: Path,
) -> tuple[str, str]:
    """Run the multi-round discussion and return (transcript, consensus)."""
    transcript_parts: list[str] = []
    agent_labels = list(syntheses.keys())

    perspectives_block = "\n\n".join(
        f"## {label}\n\n{text}" for label, text in syntheses.items()
    )

    # ── Round 1: Present and analyze perspectives ──
    print(f"  [Round 1/{num_rounds}] Presenting perspectives...")
    user_r1 = (
        f"Research topic: {topic}\n\n"
        f"The following are independent literature syntheses from {len(syntheses)} "
        f"research agents who studied this topic independently:\n\n"
        f"{perspectives_block}\n\n"
        f"Please analyze these perspectives following your instructions."
    )
    resp_r1 = llm.chat(
        [{"role": "user", "content": user_r1}],
        system=SYSTEM_PRESENT,
        max_tokens=4096,
    )
    analysis = resp_r1.content
    transcript_parts.append(f"# Discussion Transcript\n\nTopic: {topic}\n")
    transcript_parts.append(f"Participants: {', '.join(agent_labels)}\n")
    transcript_parts.append(f"## Round 1: Perspective Analysis\n\n{analysis}\n")
    print(f"    Done ({resp_r1.total_tokens} tokens)")

    # ── Round 2: Critical review ──
    if num_rounds >= 2:
        print(f"  [Round 2/{num_rounds}] Critical review...")
        user_r2 = (
            f"Original perspectives:\n\n{perspectives_block}\n\n"
            f"---\n\nInitial analysis:\n\n{analysis}\n\n"
            f"Please provide your critical review following your instructions."
        )
        resp_r2 = llm.chat(
            [{"role": "user", "content": user_r2}],
            system=SYSTEM_CRITIQUE,
            max_tokens=4096,
        )
        critique = resp_r2.content
        transcript_parts.append(f"## Round 2: Critical Review\n\n{critique}\n")
        print(f"    Done ({resp_r2.total_tokens} tokens)")
    else:
        critique = analysis

    # ── Round 3+: Consensus synthesis ──
    if num_rounds >= 3:
        print(f"  [Round 3/{num_rounds}] Building consensus...")
        user_r3 = (
            f"Research topic: {topic}\n\n"
            f"Original perspectives:\n\n{perspectives_block}\n\n"
            f"---\n\nPerspective analysis:\n\n{analysis}\n\n"
            f"---\n\nCritical review:\n\n{critique}\n\n"
            f"Please produce the consensus synthesis following your instructions."
        )
        resp_r3 = llm.chat(
            [{"role": "user", "content": user_r3}],
            system=SYSTEM_CONSENSUS,
            max_tokens=8192,
        )
        consensus = resp_r3.content
        transcript_parts.append(f"## Round 3: Consensus Synthesis\n\n{consensus}\n")
        print(f"    Done ({resp_r3.total_tokens} tokens)")
    else:
        consensus_prompt = (
            f"Research topic: {topic}\n\n"
            f"Perspectives:\n\n{perspectives_block}\n\n"
            f"Discussion so far:\n\n{critique}\n\n"
            f"Please produce a unified consensus synthesis."
        )
        resp_c = llm.chat(
            [{"role": "user", "content": consensus_prompt}],
            system=SYSTEM_CONSENSUS,
            max_tokens=8192,
        )
        consensus = resp_c.content
        transcript_parts.append(f"## Consensus\n\n{consensus}\n")

    transcript = "\n---\n\n".join(transcript_parts)
    return transcript, consensus


def main():
    parser = argparse.ArgumentParser(description="L1 Discussion Runner")
    parser.add_argument("--config", required=True, help="Path to config.arc.yaml")
    parser.add_argument(
        "--synthesis-dirs", nargs="+", required=True,
        help="Paths to stage-07 dirs containing synthesis.md",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--rounds", type=int, default=3, help="Number of discussion rounds")
    parser.add_argument("--topic", default="", help="Research topic override")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write heartbeat so the bridge can track progress
    heartbeat = {"status": "starting", "started_at": int(time.time() * 1000)}
    (output_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, ensure_ascii=False), encoding="utf-8",
    )

    print(f"🦞 Discussion Runner starting")
    print(f"   Config:     {args.config}")
    print(f"   Dirs:       {args.synthesis_dirs}")
    print(f"   Rounds:     {args.rounds}")
    print(f"   Output:     {args.output}")

    # Load config and create LLM client
    config = RCConfig.load(args.config, check_paths=False)
    llm = create_llm_client(config)

    topic = args.topic or config.research.topic
    print(f"   Topic:      {topic}")

    # Collect syntheses
    syntheses: dict[str, str] = {}
    for i, sd in enumerate(args.synthesis_dirs):
        label = f"Agent {_label(i)}"
        text = _read_synthesis(Path(sd))
        if text:
            syntheses[label] = text
            print(f"   ✓ {label}: {len(text)} chars from {sd}")
        else:
            print(f"   ✗ {label}: no synthesis.md found in {sd}")

    if len(syntheses) < 2:
        print("ERROR: Need at least 2 syntheses for discussion")
        heartbeat["status"] = "failed"
        heartbeat["error"] = "fewer than 2 syntheses available"
        (output_dir / "heartbeat.json").write_text(
            json.dumps(heartbeat, ensure_ascii=False), encoding="utf-8",
        )
        sys.exit(1)

    heartbeat["status"] = "discussing"
    heartbeat["participants"] = len(syntheses)
    (output_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, ensure_ascii=False), encoding="utf-8",
    )

    # Run discussion
    print(f"\n🗣️  Starting {args.rounds}-round discussion with {len(syntheses)} perspectives...\n")
    transcript, consensus = run_discussion(llm, topic, syntheses, args.rounds, output_dir)

    # Write outputs
    (output_dir / "discussion_transcript.md").write_text(transcript, encoding="utf-8")
    (output_dir / "consensus_synthesis.md").write_text(consensus, encoding="utf-8")

    heartbeat["status"] = "completed"
    heartbeat["completed_at"] = int(time.time() * 1000)
    (output_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n✅ Discussion complete")
    print(f"   Transcript: {output_dir / 'discussion_transcript.md'}")
    print(f"   Consensus:  {output_dir / 'consensus_synthesis.md'}")


if __name__ == "__main__":
    main()
