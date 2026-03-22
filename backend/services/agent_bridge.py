#!/usr/bin/env python3
"""
Agent Bridge v2 — project isolation, inter-layer task queues, idle-pull scheduling.

Architecture:
  runs_base/
  ├── projects/
  │   ├── proj-xxx/          # Each project has its own run_dir
  │   │   ├── stage-01/ ... stage-15/
  │   │   ├── checkpoint.json
  │   │   └── heartbeat.json
  │   └── proj-yyy/ ...
  └── queues/
      ├── idea_to_experiment.json
      ├── experiment_to_coding.json
      ├── coding_to_execution.json
      └── execution_feedback.json

Usage:
    python agent_bridge.py [--port 8766] [--agent-dir /path/to/agent]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import websockets

# ── Constants ───────────────────────────────────────────────────────────────

STAGE_TO_LAYER: dict[int, str] = {
    1: "idea", 2: "idea", 3: "idea", 4: "idea",
    5: "idea", 6: "idea", 7: "idea", 8: "idea",
    9: "experiment",
    10: "coding", 11: "coding", 12: "coding", 13: "coding",
    14: "execution", 15: "execution", 16: "execution", 17: "execution", 18: "execution",
    19: "writing", 20: "writing", 21: "writing", 22: "writing",
}

LAYER_STAGES: dict[str, list[int]] = {
    "idea": [1, 2, 3, 4, 5, 6, 7, 8],
    "experiment": [9],
    "coding": [10, 11, 12, 13],
    "execution": [14, 15, 16, 17, 18],
    "writing": [19, 20, 21, 22],
}

LAYER_RANGE: dict[str, tuple[int, int]] = {
    "idea": (1, 8),
    "experiment": (9, 9),
    "coding": (10, 13),
    "execution": (14, 18),
    "writing": (19, 22),
}

LAYER_RANGE_PHASE1: dict[str, tuple[int, int]] = {"idea": (1, 7)}
LAYER_RANGE_PHASE2: dict[str, tuple[int, int]] = {"idea": (8, 8)}

DISCUSSION_STAGE = 100

PASSTHROUGH_LAYERS: set[str] = set()

STAGE_NAMES: dict[int, str] = {
    1: "TOPIC_INIT", 2: "PROBLEM_DECOMPOSE", 3: "SEARCH_STRATEGY",
    4: "LITERATURE_COLLECT", 5: "LITERATURE_SCREEN", 6: "KNOWLEDGE_EXTRACT",
    7: "SYNTHESIS", 8: "HYPOTHESIS_GEN", 9: "EXPERIMENT_DESIGN",
    10: "CODEBASE_SEARCH", 11: "CODE_GENERATION", 12: "SANITY_CHECK",
    13: "RESOURCE_PLANNING", 14: "EXPERIMENT_RUN", 15: "ITERATIVE_REFINE",
    16: "RESULT_ANALYSIS", 17: "RESEARCH_DECISION", 18: "KNOWLEDGE_SUMMARY",
    19: "PAPER_OUTLINE", 20: "PAPER_DRAFT", 21: "PEER_REVIEW", 22: "PAPER_REVISION",
}

STAGE_OUTPUTS: dict[int, list[str]] = {
    1: ["goal.md", "hardware_profile.json"], 2: ["problem_tree.md"],
    3: ["search_plan.yaml", "sources.json", "queries.json"], 4: ["candidates.jsonl"],
    5: ["shortlist.jsonl"], 6: ["cards/"], 7: ["synthesis.md"], 8: ["hypotheses.md"],
    9: ["exp_plan.yaml"], 10: ["codebase_candidates.json"],
    11: ["experiment/", "experiment_spec.md"], 12: ["sanity_report.json"],
    13: ["schedule.json"], 14: ["runs/"],
    15: ["refinement_log.json", "experiment_final/"],
    16: ["analysis.md", "experiment_summary.json", "charts/"], 17: ["decision.md"], 18: ["knowledge_entry.json"],
    19: ["outline.md"], 20: ["paper_draft.md"], 21: ["reviews.md"], 22: ["paper_revised.md"],
}

REPO_FOR_STAGE: dict[int, str] = {
    1: "knowledge", 2: "knowledge", 3: "knowledge", 4: "knowledge",
    5: "knowledge", 6: "knowledge", 7: "knowledge", 8: "knowledge",
    9: "exp_design",
    10: "codebase", 11: "codebase", 12: "codebase", 13: "codebase",
    14: "results", 15: "results", 16: "results", 17: "results", 18: "insights",
    19: "papers", 20: "papers", 21: "papers", 22: "papers",
}

# Queue names between layers
QUEUE_NAMES: dict[str, tuple[str, str]] = {
    "idea_to_experiment":     ("idea",       "experiment"),
    "experiment_to_coding":   ("experiment", "coding"),
    "coding_to_execution":    ("coding",     "execution"),
    "execution_to_writing":   ("execution",  "writing"),
    "execution_feedback":     ("execution",  "idea"),
}

# Which queue a completing layer feeds into
LAYER_OUTPUT_QUEUE: dict[str, str] = {
    "idea":       "idea_to_experiment",
    "experiment": "experiment_to_coding",
    "coding":     "coding_to_execution",
    "execution":  "execution_to_writing",
    "writing":    "execution_feedback",
}

# Which queue a layer pulls tasks from
LAYER_INPUT_QUEUE: dict[str, str] = {
    "experiment": "idea_to_experiment",
    "coding":     "experiment_to_coding",
    "execution":  "coding_to_execution",
    "writing":    "execution_to_writing",
    "idea":       "execution_feedback",
}

# ── Utilities ───────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())[:8]

def _now_ms() -> int:
    return int(time.time() * 1000)

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    project_id: str
    run_dir: str
    config_path: str
    source_layer: str
    target_layer: str
    topic: str = ""
    status: str = "pending"          # pending | assigned | completed | failed
    assigned_to: str | None = None
    created_at: int = 0
    assigned_at: int = 0
    completed_at: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "Task":
        t = Task(
            id=d["id"], project_id=d["project_id"], run_dir=d["run_dir"],
            config_path=d.get("config_path", ""),
            source_layer=d["source_layer"], target_layer=d["target_layer"],
            topic=d.get("topic", ""),
        )
        t.status = d.get("status", "pending")
        t.assigned_to = d.get("assigned_to")
        t.created_at = d.get("created_at", 0)
        t.assigned_at = d.get("assigned_at", 0)
        t.completed_at = d.get("completed_at", 0)
        return t


@dataclass
class TaskQueue:
    """File-backed FIFO task queue."""
    name: str
    path: Path
    tasks: list[Task] = field(default_factory=list)

    def load(self):
        data = _read_json(self.path)
        if data and isinstance(data, list):
            self.tasks = [Task.from_dict(d) for d in data]

    def save(self):
        _write_json(self.path, [t.to_dict() for t in self.tasks])

    def push(self, task: Task):
        self.tasks.append(task)
        self.save()

    def peek_pending(self) -> Task | None:
        for t in self.tasks:
            if t.status == "pending":
                return t
        return None

    def assign(self, task_id: str, agent_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id and t.status == "pending":
                t.status = "assigned"
                t.assigned_to = agent_id
                t.assigned_at = _now_ms()
                self.save()
                return t
        return None

    def complete(self, task_id: str):
        for t in self.tasks:
            if t.id == task_id:
                t.status = "completed"
                t.completed_at = _now_ms()
                self.save()
                return

    def fail(self, task_id: str):
        for t in self.tasks:
            if t.id == task_id:
                t.status = "failed"
                self.save()
                return

    def pending_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == "pending")

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total": len(self.tasks),
            "pending": sum(1 for t in self.tasks if t.status == "pending"),
            "assigned": sum(1 for t in self.tasks if t.status == "assigned"),
            "completed": sum(1 for t in self.tasks if t.status == "completed"),
        }


@dataclass
class DiscussionGroup:
    """Tracks a group of L1 agents discussing the same topic."""
    project_id: str
    topic: str
    config_path: str
    agent_ids: list[str] = field(default_factory=list)
    run_dirs: dict[str, str] = field(default_factory=dict)   # agent_id -> run_dir
    completed_s7: set[str] = field(default_factory=set)       # agent_ids done with S7
    status: str = "gathering"    # gathering | waiting | discussing | done
    discussion_process: subprocess.Popen | None = field(default=None, repr=False)
    discussion_output_dir: str = ""

    def all_ready(self) -> bool:
        return len(self.completed_s7) >= len(self.agent_ids) and len(self.agent_ids) >= 2

    def synthesis_dirs(self) -> list[str]:
        dirs = []
        for aid in self.agent_ids:
            rd = self.run_dirs.get(aid, "")
            if rd:
                dirs.append(str(Path(rd) / "stage-07"))
        return dirs


@dataclass
class LobsterAgent:
    id: str
    name: str
    layer: str
    run_id: str
    run_dir: str
    config_path: str
    project_id: str = ""
    status: str = "idle"
    current_stage: int | None = None
    current_task: str = ""
    assigned_task_id: str | None = None
    stage_progress: dict[int, str] = field(default_factory=dict)
    process: subprocess.Popen | None = field(default=None, repr=False)
    _prev_heartbeat: dict = field(default_factory=dict, repr=False)
    _prev_checkpoint: dict = field(default_factory=dict, repr=False)
    _known_artifacts: set[str] = field(default_factory=set, repr=False)

    def to_frontend(self) -> dict:
        return {
            "id": self.id, "name": self.name, "layer": self.layer,
            "runId": self.run_id, "status": self.status,
            "currentStage": self.current_stage,
            "currentTask": self.current_task,
            "stageProgress": self.stage_progress,
        }


class GpuAllocator:
    """Manages GPU assignment across concurrent projects."""

    def __init__(self, total_gpus: int = 8, gpus_per_project: int = 2):
        self.total_gpus = total_gpus
        self.gpus_per_project = gpus_per_project
        self.assignments: dict[str, list[int]] = {}  # project_id -> [gpu_ids]
        self._occupied: set[int] = set()

    def available_count(self) -> int:
        return self.total_gpus - len(self._occupied)

    def can_allocate(self) -> bool:
        return self.available_count() >= self.gpus_per_project

    def allocate(self, project_id: str) -> list[int] | None:
        if project_id in self.assignments:
            return self.assignments[project_id]
        if not self.can_allocate():
            return None
        free = sorted(set(range(self.total_gpus)) - self._occupied)
        assigned = free[:self.gpus_per_project]
        self.assignments[project_id] = assigned
        self._occupied.update(assigned)
        return assigned

    def release(self, project_id: str) -> list[int]:
        gpus = self.assignments.pop(project_id, [])
        self._occupied -= set(gpus)
        return gpus

    def get(self, project_id: str) -> list[int] | None:
        return self.assignments.get(project_id)

    def summary(self) -> dict:
        return {
            "total": self.total_gpus,
            "per_project": self.gpus_per_project,
            "free": self.available_count(),
            "assignments": {k: v for k, v in self.assignments.items()},
        }


@dataclass
class BridgeState:
    agents: dict[str, LobsterAgent] = field(default_factory=dict)
    queues: dict[str, TaskQueue] = field(default_factory=dict)
    clients: set = field(default_factory=set)
    python_path: str = ""
    agent_package_dir: str = ""
    runs_base_dir: str = ""
    gpu_allocator: GpuAllocator = field(default_factory=GpuAllocator)
    result_registry: "ResultRegistry | None" = None
    auto_loop: bool = False
    # Discussion mode: L1 agents discuss after S7, before S8
    discussion_mode: bool = False
    discussion_groups: dict[str, DiscussionGroup] = field(default_factory=dict)
    discussion_rounds: int = 3
    discussion_models: list[str] = field(default_factory=lambda: ["gpt-5.3-codex-spark", "claude-opus-4-6"])
    # Idea factory: L1 idle → produce ideas via S7+S8
    idea_factory_topic: str = ""
    idea_factory_config: str = ""
    idea_factory_remaining: int = 0  # 0=disabled, -1=infinite, N=count
    idea_factory_produced: int = 0

    def projects_dir(self) -> Path:
        return Path(self.runs_base_dir) / "projects"

    def queues_dir(self) -> Path:
        return Path(self.runs_base_dir) / "queues"


# ── Message builders ────────────────────────────────────────────────────────

def msg_agent_update(agent: LobsterAgent) -> dict:
    return {"type": "agent_update", "payload": agent.to_frontend()}

def msg_stage_update(agent_id: str, stage: int, status: str) -> dict:
    return {"type": "stage_update", "payload": {"agentId": agent_id, "stage": stage, "status": status}}

def msg_artifact(repo_id: str, filename: str, agent_name: str, size: str, project_id: str = "", content: str = "") -> dict:
    payload: dict = {
        "id": _uid(), "repoId": repo_id, "projectId": project_id, "filename": filename,
        "producedBy": agent_name, "timestamp": _now_ms(), "size": size, "status": "fresh",
    }
    if content:
        payload["content"] = content
    return {"type": "artifact_produced", "payload": payload}

def msg_log(agent: LobsterAgent, message: str, level: str = "info", stage: int | None = None) -> dict:
    return {"type": "log", "payload": {
        "id": _uid(), "agentId": agent.id, "agentName": agent.name,
        "layer": agent.layer, "stage": stage or agent.current_stage,
        "message": message, "level": level, "timestamp": _now_ms(),
    }}

def msg_queue_update(queues: dict[str, TaskQueue]) -> dict:
    return {"type": "queue_update", "payload": {name: q.summary() for name, q in queues.items()}}


# ── File monitoring ─────────────────────────────────────────────────────────

def _sync_completed_stages(
    agent: LobsterAgent, run_dir: Path, layer_range: tuple[int, int], done_up_to: int,
) -> list[dict]:
    """Mark all stages from layer_range[0] to done_up_to as completed, emit events for new ones."""
    messages: list[dict] = []
    for s in range(layer_range[0], min(done_up_to, layer_range[1]) + 1):
        if agent.stage_progress.get(s) == "completed":
            continue
        if s not in STAGE_TO_LAYER:
            continue
        agent.stage_progress[s] = "completed"
        messages.append(msg_stage_update(agent.id, s, "completed"))
        messages.append(msg_log(agent, f"{STAGE_NAMES.get(s, f'S{s}')} 完成", "success", s))
        stage_dir = run_dir / f"stage-{s:02d}"
        if stage_dir.is_dir():
            for expected in STAGE_OUTPUTS.get(s, []):
                artifact_path = stage_dir / expected.rstrip("/")
                key = f"{s}:{expected}"
                if key not in agent._known_artifacts and artifact_path.exists():
                    agent._known_artifacts.add(key)
                    size = "dir" if artifact_path.is_dir() else f"{artifact_path.stat().st_size / 1024:.1f} KB"
                    content = ""
                    if expected.endswith(".json") and artifact_path.is_file() and REPO_FOR_STAGE.get(s) == "insights":
                        try:
                            _entry = json.loads(artifact_path.read_text(encoding="utf-8"))
                            _parts = []
                            if _entry.get("topic"): _parts.append(f"Topic: {_entry['topic']}")
                            for c in (_entry.get("conclusions") or [])[:5]: _parts.append(f"  - {c}")
                            if _entry.get("insights"):
                                _parts.append("Insights:")
                                for i in (_entry.get("insights") or [])[:3]: _parts.append(f"  * {i}")
                            if _entry.get("suggested_directions"):
                                _parts.append("Directions:")
                                for d in (_entry.get("suggested_directions") or [])[:3]: _parts.append(f"  > {d}")
                            if _entry.get("results"):
                                _parts.append(f"Results: {json.dumps(_entry['results'])}")
                            content = "\n".join(_parts)
                        except Exception:
                            pass
                    messages.append(msg_artifact(
                        REPO_FOR_STAGE.get(s, "knowledge"), expected, agent.name, size, agent.project_id, content,
                    ))
    return messages


def poll_agent(agent: LobsterAgent) -> list[dict]:
    messages: list[dict] = []
    run_dir = Path(agent.run_dir)
    if not run_dir.exists():
        return messages

    # Only read heartbeat/checkpoint if THIS agent's process is running,
    # to avoid cross-contamination when multiple agents share a run_dir.
    if agent.process is not None and agent.process.poll() is None:
        _s7_only = getattr(agent, '_is_idea_factory_s7_only', False)
        layer_range = (7, 7) if _s7_only else LAYER_RANGE.get(agent.layer, (1, 15))

        hb = _read_json(run_dir / "heartbeat.json")
        if hb and hb != agent._prev_heartbeat:
            new_stage = hb.get("last_stage")
            old_stage = agent.current_stage
            if (
                new_stage and new_stage != old_stage
                and new_stage in STAGE_TO_LAYER
                and layer_range[0] <= new_stage <= layer_range[1]
            ):
                agent.current_stage = new_stage
                agent.current_task = f"Stage {new_stage}: {STAGE_NAMES.get(new_stage, '?')}"
                agent.status = "working"
                if new_stage not in agent.stage_progress or agent.stage_progress[new_stage] != "completed":
                    agent.stage_progress[new_stage] = "running"
                messages.append(msg_agent_update(agent))
                messages.append(msg_stage_update(agent.id, new_stage, "running"))
                messages.append(msg_log(agent, f"开始 {STAGE_NAMES.get(new_stage, f'S{new_stage}')}", "info", new_stage))
            agent._prev_heartbeat = hb

        cp = _read_json(run_dir / "checkpoint.json")
        if cp and cp != agent._prev_checkpoint:
            done_up_to = cp.get("last_completed_stage", 0)
            messages.extend(_sync_completed_stages(agent, run_dir, layer_range, done_up_to))
            agent._prev_checkpoint = cp

            if agent.current_stage and done_up_to >= agent.current_stage and done_up_to < layer_range[1]:
                next_stage = done_up_to + 1
                if next_stage in STAGE_TO_LAYER and layer_range[0] <= next_stage <= layer_range[1]:
                    agent.current_stage = next_stage
                    agent.current_task = f"Stage {next_stage}: {STAGE_NAMES.get(next_stage, '?')}"
                    agent.stage_progress[next_stage] = "running"
                    messages.append(msg_agent_update(agent))
                    messages.append(msg_stage_update(agent.id, next_stage, "running"))
                    messages.append(msg_log(agent, f"开始 {STAGE_NAMES.get(next_stage, f'S{next_stage}')}", "info", next_stage))

    if agent.process is not None:
        retcode = agent.process.poll()
        if retcode is not None:
            # Final read: catch any checkpoint/artifact updates written before exit
            _s7_only_final = getattr(agent, '_is_idea_factory_s7_only', False)
            layer_range = (7, 7) if _s7_only_final else LAYER_RANGE.get(agent.layer, (1, 15))
            cp = _read_json(run_dir / "checkpoint.json")
            if cp:
                done_up_to = cp.get("last_completed_stage", 0)
                messages.extend(_sync_completed_stages(agent, run_dir, layer_range, done_up_to))
                agent._prev_checkpoint = cp

            if retcode == 0:
                agent.status = "done"
                agent.current_task = ""
                agent.current_stage = None
                messages.append(msg_agent_update(agent))
                messages.append(msg_log(agent, f"层任务完成 (project={agent.project_id})", "success"))
            else:
                agent.status = "error"
                agent.current_task = f"exit code={retcode}"
                messages.append(msg_agent_update(agent))
                messages.append(msg_log(agent, f"进程异常 (code={retcode})", "error"))
            agent.process = None

    return messages


# ── Agent lifecycle ─────────────────────────────────────────────────────────

def create_agent(state: BridgeState, name: str, layer: str) -> LobsterAgent:
    agent = LobsterAgent(
        id=f"L-{_uid()}", name=name, layer=layer,
        run_id="", run_dir="", config_path="",
        stage_progress={s: "pending" for s in LAYER_STAGES.get(layer, [])},
    )
    state.agents[agent.id] = agent
    return agent


def _assign_task_to_agent(agent: LobsterAgent, task: Task) -> None:
    """Common setup when assigning a task to an agent."""
    agent.project_id = task.project_id
    agent.run_dir = task.run_dir
    agent.run_id = task.project_id
    agent.config_path = task.config_path
    agent.assigned_task_id = task.id
    agent._topic = task.topic
    agent.status = "working"
    agent.stage_progress = {s: "pending" for s in LAYER_STAGES.get(agent.layer, [])}
    agent._prev_heartbeat = {}
    agent._prev_checkpoint = {}
    agent._known_artifacts = set()


def _passthrough_agent(agent: LobsterAgent) -> list[dict]:
    """For passthrough layers (e.g. coding): read existing artifacts, mark done immediately."""
    messages: list[dict] = []
    run_dir = Path(agent.run_dir)
    layer_range = LAYER_RANGE.get(agent.layer, (1, 15))

    for s in range(layer_range[0], layer_range[1] + 1):
        stage_dir = run_dir / f"stage-{s:02d}"
        if stage_dir.is_dir():
            agent.stage_progress[s] = "completed"
            messages.append(msg_stage_update(agent.id, s, "completed"))
            messages.append(msg_log(agent, f"{STAGE_NAMES.get(s, f'S{s}')} 结果已就绪 (由上层产出)", "success", s))
            for expected in STAGE_OUTPUTS.get(s, []):
                artifact_path = stage_dir / expected.rstrip("/")
                key = f"{s}:{expected}"
                if key not in agent._known_artifacts and artifact_path.exists():
                    agent._known_artifacts.add(key)
                    size = "dir" if artifact_path.is_dir() else f"{artifact_path.stat().st_size / 1024:.1f} KB"
                    messages.append(msg_artifact(
                        REPO_FOR_STAGE.get(s, "codebase"), expected, agent.name, size, agent.project_id,
                    ))
        else:
            agent.stage_progress[s] = "failed"
            messages.append(msg_log(agent, f"S{s} 产物未找到 (stage-{s:02d}/ 不存在)", "warning", s))

    agent.status = "done"
    agent.current_task = ""
    agent.current_stage = None
    messages.append(msg_agent_update(agent))
    messages.append(msg_log(agent, f"验收完成 (project={agent.project_id})", "success"))
    return messages


def launch_agent_for_task(state: BridgeState, agent: LobsterAgent, task: Task) -> list[dict]:
    """Assign a task to an agent. Passthrough layers skip process launch."""
    messages: list[dict] = []
    _assign_task_to_agent(agent, task)

    # Passthrough layers: just verify artifacts and mark done
    if agent.layer in PASSTHROUGH_LAYERS:
        messages.append(msg_agent_update(agent))
        messages.append(msg_log(agent, f"领取任务 [{task.project_id}] 验收 S10 代码产物", "info"))
        messages.extend(_passthrough_agent(agent))
        return messages

    # Normal layers: launch ResearchClaw process
    # Discussion mode: L1 runs S1-S7 only (S8 runs after discussion)
    if state.discussion_mode and agent.layer == "idea":
        layer_range = LAYER_RANGE_PHASE1["idea"]
    else:
        layer_range = LAYER_RANGE.get(agent.layer, (1, 15))
    fs, ts = layer_range

    cmd = [
        state.python_path, "-m", "researchclaw", "run",
        "--config", task.config_path,
        "--output", task.run_dir,
        "--from-stage", STAGE_NAMES.get(fs, str(fs)),
        "--to-stage", STAGE_NAMES.get(ts, str(ts)),
        "--auto-approve",
        "--skip-preflight",
    ]
    if task.topic:
        cmd.extend(["--topic", task.topic])

    try:
        proc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        # L4 execution layer: allocate GPUs
        if agent.layer == "execution":
            gpu_ids = state.gpu_allocator.allocate(task.project_id)
            if gpu_ids is not None:
                proc_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
                messages.append(msg_log(agent, f"GPU 分配: {gpu_ids} → CUDA_VISIBLE_DEVICES={proc_env['CUDA_VISIBLE_DEVICES']}", "info"))
            else:
                messages.append(msg_log(agent, "GPU 不足，使用默认分配", "warning"))

        log_path = Path(task.run_dir) / f"agent_{agent.id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=state.agent_package_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env=proc_env,
        )
        agent.process = proc
        agent.current_task = f"项目 {task.project_id} · PID={proc.pid}"
        messages.append(msg_agent_update(agent))
        messages.append(msg_log(agent, f"领取任务 [{task.project_id}] 启动 S{fs}→S{ts} (PID={proc.pid})", "info"))
    except Exception as e:
        agent.status = "error"
        agent.current_task = f"启动失败: {e}"
        messages.append(msg_agent_update(agent))
        messages.append(msg_log(agent, f"启动失败: {e}", "error"))

    return messages


def stop_agent(agent: LobsterAgent) -> list[dict]:
    messages: list[dict] = []
    if agent.process and agent.process.poll() is None:
        agent.process.terminate()
        try:
            agent.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent.process.kill()
    agent.process = None
    agent.status = "idle"
    agent.current_task = ""
    agent.current_stage = None
    agent.assigned_task_id = None
    messages.append(msg_agent_update(agent))
    messages.append(msg_log(agent, "Agent 已停止", "warning"))
    return messages


# ── Task queue operations ───────────────────────────────────────────────────

def submit_new_project(state: BridgeState, project_id: str, config_path: str, topic: str = "") -> list[dict]:
    """Submit a brand-new project — goes into init_to_idea queue.

    In discussion mode, creates one task per idle L1 agent so they all
    research the same topic independently, then discuss after S7.
    """
    messages: list[dict] = []
    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")

    if state.discussion_mode:
        idea_agents = [a for a in state.agents.values() if a.layer == "idea"]
        if len(idea_agents) < 2:
            messages.append(msg_log(sys_agent, f"沟通讨论模式需要至少 2 个 L1 agent，当前只有 {len(idea_agents)} 个", "warning"))
            # Fall through to single-agent mode

        else:
            group = DiscussionGroup(
                project_id=project_id, topic=topic, config_path=config_path,
            )
            for agent in idea_agents:
                perspective_dir = str(state.projects_dir() / project_id / f"perspective-{agent.id}")
                os.makedirs(perspective_dir, exist_ok=True)
                group.agent_ids.append(agent.id)
                group.run_dirs[agent.id] = perspective_dir

                task = Task(
                    id=f"task-{_uid()}", project_id=project_id, run_dir=perspective_dir,
                    config_path=config_path, topic=topic,
                    source_layer="init", target_layer="idea",
                    created_at=_now_ms(),
                )
                state.queues["init_to_idea"].push(task)

            state.discussion_groups[project_id] = group
            messages.append(msg_log(
                sys_agent,
                f"新项目 [{project_id}] 沟通讨论模式: {len(idea_agents)} 个 agent 将独立调研后讨论",
                "info", DISCUSSION_STAGE,
            ))
            messages.append(msg_queue_update(state.queues))
            return messages

    # Default single-agent mode
    run_dir = str(state.projects_dir() / project_id)
    os.makedirs(run_dir, exist_ok=True)

    task = Task(
        id=f"task-{_uid()}", project_id=project_id, run_dir=run_dir,
        config_path=config_path, topic=topic,
        source_layer="init", target_layer="idea",
        created_at=_now_ms(),
    )
    state.queues["init_to_idea"].push(task)

    messages.append(msg_log(sys_agent, f"新项目 [{project_id}] 已加入调研队列", "info"))
    messages.append(msg_queue_update(state.queues))
    return messages


def on_agent_done(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """When an agent finishes, complete its task and create a follow-up task for the next layer."""
    messages: list[dict] = []

    # Complete assigned task
    if agent.assigned_task_id:
        for q in state.queues.values():
            q.complete(agent.assigned_task_id)

    # Discussion mode: L1 agent completed S7 → wait for peers instead of proceeding
    if state.discussion_mode and agent.layer == "idea" and agent.project_id in state.discussion_groups:
        group = state.discussion_groups[agent.project_id]
        group.completed_s7.add(agent.id)
        agent.status = "waiting_discussion"
        agent.current_stage = DISCUSSION_STAGE
        agent.current_task = f"等待沟通讨论 ({len(group.completed_s7)}/{len(group.agent_ids)})"
        agent.stage_progress[DISCUSSION_STAGE] = "running"
        messages.append(msg_agent_update(agent))
        messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "running"))
        messages.append(msg_log(agent, f"S7 完成，等待沟通讨论 ({len(group.completed_s7)}/{len(group.agent_ids)})", "info", DISCUSSION_STAGE))

        if group.all_ready():
            messages.extend(_trigger_discussion(state, group))
        return messages

    # Create follow-up task in the next queue
    output_queue_name = LAYER_OUTPUT_QUEUE.get(agent.layer)
    if output_queue_name and output_queue_name in state.queues and agent.project_id:
        # L4→L5: only push to writing queue if S17 decision is PROCEED (or forced-PROCEED)
        if agent.layer == "execution" and output_queue_name == "execution_to_writing":
            _decision_file = Path(agent.run_dir) / "stage-17" / "decision.md"
            _warning_file = Path(agent.run_dir) / "quality_warning.txt"
            _summary_file = Path(agent.run_dir) / "pipeline_summary.json"
            _is_proceed = False
            if _decision_file.exists():
                _dec_text = _decision_file.read_text(encoding="utf-8").upper()
                _is_proceed = "PROCEED" in _dec_text and "REFINE" not in _dec_text.split("PROCEED")[0][-50:]
            if not _is_proceed and _warning_file.exists():
                _warn_text = _warning_file.read_text(encoding="utf-8")
                if "max pivots" in _warn_text.lower():
                    _is_proceed = True
                    messages.append(msg_log(agent, "S17 决策为 REFINE 但已达最大迭代次数，强制进入论文写作", "warning"))
            if not _is_proceed and _summary_file.exists():
                try:
                    import json as _json
                    _summary = _json.loads(_summary_file.read_text(encoding="utf-8"))
                    if _summary.get("final_status") == "done" and _summary.get("stages_failed", 1) == 0:
                        _is_proceed = True
                        messages.append(msg_log(agent, "Pipeline 全部完成，进入论文写作", "info"))
                except Exception:
                    pass
            if not _is_proceed:
                messages.append(msg_log(agent, f"S17 决策非 PROCEED，跳过论文写作", "info"))
                output_queue_name = None

        if output_queue_name and output_queue_name in state.queues:
            _, target_layer = QUEUE_NAMES[output_queue_name]
            follow_task = Task(
                id=f"task-{_uid()}",
                project_id=agent.project_id,
                run_dir=agent.run_dir,
                config_path=agent.config_path,
                topic=getattr(agent, '_topic', ''),
                source_layer=agent.layer,
                target_layer=target_layer,
                created_at=_now_ms(),
            )
            state.queues[output_queue_name].push(follow_task)
            messages.append(msg_log(
                agent,
                f"任务完成 → 项目 [{agent.project_id}] 已加入 {output_queue_name} 队列",
                "success",
            ))

    # Release GPU allocation for execution layer
    if agent.layer == "execution" and agent.project_id:
        released = state.gpu_allocator.release(agent.project_id)
        if released:
            messages.append(msg_log(agent, f"GPU {released} 已释放", "info"))

    # Reset agent for next task
    agent.assigned_task_id = None
    agent.project_id = ""
    agent.status = "idle"
    agent.current_task = "等待任务..."
    messages.append(msg_agent_update(agent))
    messages.append(msg_queue_update(state.queues))

    return messages


def _create_model_config(base_config_path: str, model_name: str, output_dir: str) -> str:
    """Create a per-agent config file with a different primary_model."""
    import yaml
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "llm" in cfg:
        cfg["llm"]["primary_model"] = model_name
    agent_config_path = str(Path(output_dir) / f"config_{model_name.replace('/', '_')}.yaml")
    with open(agent_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return agent_config_path


def _launch_idea_factory_run(state: BridgeState, agent: LobsterAgent, s7_only: bool = False, model_override: str = "") -> list[dict]:
    """Launch L1 agent to produce ideas. s7_only=True runs only S7 (for discussion mode)."""
    messages: list[dict] = []

    idea_id = f"idea-{_uid()}"
    run_dir = str(Path(state.runs_base_dir).parent / "shared_results" / "idea_runs" / idea_id)
    os.makedirs(run_dir, exist_ok=True)

    _s6_seed = Path(run_dir) / "stage-06"
    _s6_seed.mkdir(parents=True, exist_ok=True)
    (_s6_seed / "cards").mkdir(exist_ok=True)

    _s7_seed = Path(run_dir) / "stage-07"
    _s7_seed.mkdir(parents=True, exist_ok=True)

    config_path = state.idea_factory_config
    if model_override:
        try:
            config_path = _create_model_config(state.idea_factory_config, model_override, run_dir)
        except Exception as e:
            messages.append(msg_log(agent, f"模型配置创建失败 ({model_override}): {e}，使用默认配置", "warning"))
            config_path = state.idea_factory_config

    task = Task(
        id=f"task-{_uid()}",
        project_id=idea_id,
        run_dir=run_dir,
        config_path=config_path,
        topic=state.idea_factory_topic,
        source_layer="idea_factory",
        target_layer="idea",
        created_at=_now_ms(),
    )

    _assign_task_to_agent(agent, task)
    agent._is_idea_factory = True  # type: ignore[attr-defined]
    agent._is_idea_factory_s7_only = s7_only  # type: ignore[attr-defined]

    if s7_only:
        layer_range = (7, 7)
    else:
        layer_range = (7, 8)
    fs, ts = layer_range

    cmd = [
        state.python_path, "-m", "researchclaw", "run",
        "--config", config_path,
        "--output", task.run_dir,
        "--from-stage", STAGE_NAMES.get(fs, str(fs)),
        "--to-stage", STAGE_NAMES.get(ts, str(ts)),
        "--auto-approve",
        "--skip-preflight",
        "--topic", state.idea_factory_topic,
    ]

    try:
        log_path = Path(run_dir) / f"agent_{agent.id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=state.agent_package_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        agent.process = proc
        n = state.idea_factory_produced + 1
        model_tag = f" [{model_override}]" if model_override else ""
        agent.current_task = f"Idea 工厂 #{n}{model_tag}" + (" (S7 综合)" if s7_only else " (S7→S8)")
        messages.append(msg_agent_update(agent))
        label = f"Idea 工厂 #{n}: 知识综合中 (S7){model_tag}" if s7_only else f"Idea 工厂 #{n}: 生成假设中 (S7→S8){model_tag}"
        messages.append(msg_log(agent, label, "info"))
    except Exception as e:
        agent.status = "error"
        agent.current_task = f"Idea 工厂启动失败: {e}"
        messages.append(msg_agent_update(agent))

    return messages


def _on_idea_factory_done(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """Handle idea factory run completion: extract hypotheses, push to idea pool + L2 queue."""
    messages: list[dict] = []
    run_dir = Path(agent.run_dir)

    # Read hypotheses
    hyp_file = None
    for sd in sorted(run_dir.glob("stage-08*"), reverse=True):
        f = sd / "hypotheses.md"
        if f.exists():
            hyp_file = f
            break

    if hyp_file:
        hyp_text = hyp_file.read_text(encoding="utf-8")

        # Write to idea pool
        pool_dir = Path(state.runs_base_dir).parent / "shared_results" / "idea_pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        pool_file = pool_dir / "ideas.jsonl"

        entry = {
            "id": agent.project_id,
            "topic": state.idea_factory_topic,
            "hypotheses": hyp_text[:2000],
            "timestamp": _now_ms(),
            "status": "pending",
        }
        with open(pool_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Create L2 task from this idea
        idea_run_dir = str(Path(state.runs_base_dir) / "projects" / agent.project_id)
        os.makedirs(idea_run_dir, exist_ok=True)

        # Copy hypotheses to the project run dir so L2 can find it
        s8_dir = Path(idea_run_dir) / "stage-08"
        s8_dir.mkdir(parents=True, exist_ok=True)
        (s8_dir / "hypotheses.md").write_text(hyp_text, encoding="utf-8")

        # Also copy synthesis if available
        for sd in sorted(run_dir.glob("stage-07*"), reverse=True):
            sf = sd / "synthesis.md"
            if sf.exists():
                s7_dir = Path(idea_run_dir) / "stage-07"
                s7_dir.mkdir(parents=True, exist_ok=True)
                (s7_dir / "synthesis.md").write_text(sf.read_text(encoding="utf-8"), encoding="utf-8")
                break

        # Push to idea_to_experiment queue
        queue = state.queues.get("idea_to_experiment")
        if queue:
            follow_task = Task(
                id=f"task-{_uid()}",
                project_id=agent.project_id,
                run_dir=idea_run_dir,
                config_path=state.idea_factory_config,
                topic=state.idea_factory_topic,
                source_layer="idea",
                target_layer="experiment",
                created_at=_now_ms(),
            )
            queue.push(follow_task)
            messages.append(msg_log(agent, f"Idea #{state.idea_factory_produced + 1} → 实验设计队列", "success"))

        state.idea_factory_produced += 1
        if state.idea_factory_remaining > 0:
            state.idea_factory_remaining -= 1

        messages.append(msg_log(
            agent,
            f"Idea 工厂: 已产出 {state.idea_factory_produced} 个, 剩余 {'无限' if state.idea_factory_remaining == -1 else state.idea_factory_remaining}",
            "info",
        ))
    else:
        messages.append(msg_log(agent, "Idea 工厂: 未生成假设", "warning"))

    # Reset agent
    agent.assigned_task_id = None
    agent.project_id = ""
    agent.status = "idle"
    agent.current_task = "等待任务..."
    agent._is_idea_factory = False  # type: ignore[attr-defined]
    agent._is_idea_factory_s7_only = False  # type: ignore[attr-defined]
    agent._is_discussion_s8 = False  # type: ignore[attr-defined]
    agent._idea_factory_batch_id = None  # type: ignore[attr-defined]
    messages.append(msg_agent_update(agent))

    return messages


def _on_idea_factory_s7_done(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """Handle idea factory S7-only completion: enter discussion flow."""
    messages: list[dict] = []
    agent._is_idea_factory_s7_only = False  # type: ignore[attr-defined]

    batch_id = getattr(agent, '_idea_factory_batch_id', None)
    if not batch_id or batch_id not in state.discussion_groups:
        messages.append(msg_log(agent, "S7 完成但无沟通讨论组，回退到非讨论模式", "warning"))
        agent._is_idea_factory = False  # type: ignore[attr-defined]
        agent.assigned_task_id = None
        agent.project_id = ""
        agent.status = "idle"
        agent.current_task = "等待任务..."
        messages.append(msg_agent_update(agent))
        return messages

    group = state.discussion_groups[batch_id]
    group.completed_s7.add(agent.id)
    agent.status = "waiting_discussion"
    agent.current_stage = DISCUSSION_STAGE
    agent.current_task = f"等待沟通讨论 ({len(group.completed_s7)}/{len(group.agent_ids)})"
    agent.stage_progress[DISCUSSION_STAGE] = "running"
    messages.append(msg_agent_update(agent))
    messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "running"))
    messages.append(msg_log(agent, f"S7 完成，等待沟通讨论 ({len(group.completed_s7)}/{len(group.agent_ids)})", "info", DISCUSSION_STAGE))

    if group.all_ready():
        messages.extend(_trigger_discussion(state, group))

    return messages


def _trigger_discussion(state: BridgeState, group: DiscussionGroup) -> list[dict]:
    """Launch the discussion runner when all agents in a group have completed S7."""
    messages: list[dict] = []
    group.status = "discussing"

    disc_dir = str(state.projects_dir() / group.project_id / "discussion")
    os.makedirs(disc_dir, exist_ok=True)
    group.discussion_output_dir = disc_dir

    for aid in group.agent_ids:
        agent = state.agents.get(aid)
        if agent:
            agent.status = "discussing"
            agent.current_stage = DISCUSSION_STAGE
            agent.current_task = "多 Agent 沟通讨论中..."
            messages.append(msg_agent_update(agent))

    synthesis_dirs = group.synthesis_dirs()
    runner_path = str(Path(__file__).resolve().parent / "discussion_runner.py")
    cmd = [
        state.python_path, runner_path,
        "--config", group.config_path,
        "--synthesis-dirs", *synthesis_dirs,
        "--output", disc_dir,
        "--rounds", str(state.discussion_rounds),
    ]
    if group.topic:
        cmd.extend(["--topic", group.topic])

    try:
        log_path = Path(disc_dir) / "discussion.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=state.agent_package_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        group.discussion_process = proc
        sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        messages.append(msg_log(
            sys_agent,
            f"项目 [{group.project_id}] 沟通讨论开始: {len(group.agent_ids)} 个 agent, {state.discussion_rounds} 轮 (PID={proc.pid})",
            "info", DISCUSSION_STAGE,
        ))
    except Exception as e:
        group.status = "done"
        sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        messages.append(msg_log(sys_agent, f"沟通讨论启动失败: {e}", "error", DISCUSSION_STAGE))
        for aid in group.agent_ids:
            agent = state.agents.get(aid)
            if agent:
                agent.status = "error"
                agent.current_task = f"沟通讨论启动失败: {e}"
                messages.append(msg_agent_update(agent))

    return messages


def _poll_discussion(state: BridgeState, group: DiscussionGroup) -> list[dict]:
    """Check if a discussion subprocess has finished and handle completion."""
    messages: list[dict] = []
    if group.status != "discussing" or group.discussion_process is None:
        return messages

    retcode = group.discussion_process.poll()
    if retcode is None:
        return messages

    group.discussion_process = None
    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")

    if retcode != 0:
        group.status = "done"
        messages.append(msg_log(sys_agent, f"项目 [{group.project_id}] 沟通讨论失败 (exit={retcode})", "error", DISCUSSION_STAGE))
        for aid in group.agent_ids:
            agent = state.agents.get(aid)
            if agent:
                agent.status = "error"
                agent.current_task = f"沟通讨论失败 (exit={retcode})"
                agent.stage_progress[DISCUSSION_STAGE] = "failed"
                messages.append(msg_agent_update(agent))
                messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "failed"))
        return messages

    # Discussion succeeded — inject consensus into each agent's run_dir and launch S8
    consensus_file = Path(group.discussion_output_dir) / "consensus_synthesis.md"
    if not consensus_file.exists():
        messages.append(msg_log(sys_agent, f"项目 [{group.project_id}] 沟通讨论完成但未产生共识", "warning", DISCUSSION_STAGE))
        group.status = "done"
        return messages

    consensus_text = consensus_file.read_text(encoding="utf-8")
    messages.append(msg_log(sys_agent, f"项目 [{group.project_id}] 沟通讨论完成，共识已生成，启动假设生成", "success", DISCUSSION_STAGE))
    for aid in group.agent_ids:
        agent = state.agents.get(aid)
        if agent:
            agent.stage_progress[DISCUSSION_STAGE] = "completed"
            messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "completed"))

    # Also record the transcript as an artifact
    transcript_file = Path(group.discussion_output_dir) / "discussion_transcript.md"
    if transcript_file.exists():
        messages.append(msg_artifact(
            "knowledge", "discussion_transcript.md",
            "沟通讨论", f"{transcript_file.stat().st_size / 1024:.1f} KB",
            group.project_id,
        ))

    group.status = "done"

    for aid in group.agent_ids:
        agent = state.agents.get(aid)
        if not agent:
            continue

        # Write consensus into agent's stage-07 dir so S8 can read it via _read_prior_artifact
        s7_dir = Path(agent.run_dir) / "stage-07"
        s7_dir.mkdir(parents=True, exist_ok=True)
        # Append consensus to the agent's own synthesis so S8 sees the combined context
        existing_synthesis = s7_dir / "synthesis.md"
        if existing_synthesis.exists():
            original = existing_synthesis.read_text(encoding="utf-8")
            enriched = (
                f"{original}\n\n"
                f"---\n\n"
                f"# Multi-Agent Discussion Consensus\n\n"
                f"{consensus_text}"
            )
            existing_synthesis.write_text(enriched, encoding="utf-8")
        else:
            (s7_dir / "synthesis.md").write_text(consensus_text, encoding="utf-8")

        # Launch S8 for this agent
        messages.extend(_launch_s8_for_agent(state, agent, group))

    return messages


def _launch_s8_for_agent(state: BridgeState, agent: LobsterAgent, group: DiscussionGroup) -> list[dict]:
    """Launch S8 (HYPOTHESIS_GEN) for a single agent after discussion."""
    messages: list[dict] = []
    fs, ts = LAYER_RANGE_PHASE2["idea"]

    agent.status = "working"
    agent.current_task = f"项目 {group.project_id} · S8 假设生成"
    agent.stage_progress[8] = "running"
    messages.append(msg_agent_update(agent))
    messages.append(msg_stage_update(agent.id, 8, "running"))
    messages.append(msg_log(agent, "沟通讨论完成 → 开始假设生成", "info", 8))

    cmd = [
        state.python_path, "-m", "researchclaw", "run",
        "--config", group.config_path,
        "--output", agent.run_dir,
        "--from-stage", STAGE_NAMES.get(fs, str(fs)),
        "--to-stage", STAGE_NAMES.get(ts, str(ts)),
        "--auto-approve",
        "--skip-preflight",
    ]
    if group.topic:
        cmd.extend(["--topic", group.topic])

    try:
        log_path = Path(agent.run_dir) / f"agent_{agent.id}_s8.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=state.agent_package_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        agent.process = proc
        agent._is_discussion_s8 = True  # type: ignore[attr-defined]
        messages.append(msg_log(agent, f"S8 启动 (PID={proc.pid})", "info", 8))
    except Exception as e:
        agent.status = "error"
        agent.current_task = f"S8 启动失败: {e}"
        messages.append(msg_agent_update(agent))
        messages.append(msg_log(agent, f"S8 启动失败: {e}", "error"))

    return messages


def _on_discussion_s8_done(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """Handle S8 completion for an agent that went through the discussion flow."""
    messages: list[dict] = []
    agent._is_discussion_s8 = False  # type: ignore[attr-defined]

    # Push to idea_to_experiment queue
    output_queue_name = LAYER_OUTPUT_QUEUE.get("idea")
    if output_queue_name and output_queue_name in state.queues and agent.project_id:
        _, target_layer = QUEUE_NAMES[output_queue_name]
        follow_task = Task(
            id=f"task-{_uid()}",
            project_id=agent.project_id,
            run_dir=agent.run_dir,
            config_path=agent.config_path,
            topic=getattr(agent, '_topic', ''),
            source_layer="idea",
            target_layer=target_layer,
            created_at=_now_ms(),
        )
        state.queues[output_queue_name].push(follow_task)
        messages.append(msg_log(
            agent, f"假设生成完成 (沟通讨论模式) → 项目 [{agent.project_id}] 已加入 {output_queue_name} 队列", "success",
        ))

    # Reset agent
    agent.assigned_task_id = None
    agent.project_id = ""
    agent.status = "idle"
    agent.current_task = "等待任务..."
    messages.append(msg_agent_update(agent))
    messages.append(msg_queue_update(state.queues))

    return messages


def schedule_idle_agents(state: BridgeState) -> list[dict]:
    """Assign pending tasks to idle agents (FIFO pull)."""
    messages: list[dict] = []

    for agent in state.agents.values():
        if agent.status not in ("idle",) or agent.process is not None:
            continue
        if agent.status in ("waiting_discussion", "discussing"):
            continue
        if agent.assigned_task_id:
            continue

        # L4 execution layer: skip if no GPU available
        if agent.layer == "execution" and not state.gpu_allocator.can_allocate():
            continue

        # Idea agents pull from init_to_idea, and optionally execution_feedback (auto-loop)
        if agent.layer == "idea":
            candidate_queues = ["init_to_idea"]
            if state.auto_loop:
                candidate_queues.append("execution_feedback")
        else:
            q_name = LAYER_INPUT_QUEUE.get(agent.layer, "")
            candidate_queues = [q_name] if q_name else []

        assigned = False
        for queue_name in candidate_queues:
            queue = state.queues.get(queue_name)
            if not queue:
                continue
            task = queue.peek_pending()
            if not task or task.target_layer != agent.layer:
                continue

            queue.assign(task.id, agent.id)
            messages.extend(launch_agent_for_task(state, agent, task))
            messages.append(msg_queue_update(state.queues))
            assigned = True
            break

        # Idea factory: L1 idle with no queued tasks → produce ideas
        if not assigned and agent.layer == "idea" and state.idea_factory_remaining != 0:
            if state.idea_factory_topic and state.idea_factory_config:
                if not state.discussion_mode:
                    messages.extend(_launch_idea_factory_run(state, agent))
                continue

    # Discussion-mode idea factory: batch-launch when 2+ L1 agents are idle
    if state.discussion_mode and state.idea_factory_remaining != 0 and state.idea_factory_topic and state.idea_factory_config:
        idle_idea_agents = [
            a for a in state.agents.values()
            if a.layer == "idea" and a.status == "idle" and a.process is None
            and not a.assigned_task_id
            and a.status not in ("waiting_discussion", "discussing")
        ]
        if len(idle_idea_agents) >= 2:
            batch_id = f"idea-batch-{_uid()}"
            group = DiscussionGroup(
                project_id=batch_id,
                topic=state.idea_factory_topic,
                config_path=state.idea_factory_config,
            )
            models = state.discussion_models
            for i, agent in enumerate(idle_idea_agents[:2]):
                model = models[i % len(models)] if models else ""
                messages.extend(_launch_idea_factory_run(state, agent, s7_only=True, model_override=model))
                group.agent_ids.append(agent.id)
                group.run_dirs[agent.id] = agent.run_dir
                agent._idea_factory_batch_id = batch_id  # type: ignore[attr-defined]
            state.discussion_groups[batch_id] = group
            model_list = ", ".join(models[:2]) if models else "默认"
            sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
            messages.append(msg_log(
                sys_agent,
                f"Idea 工厂沟通讨论模式: {len(group.agent_ids)} 个 agent 开始独立综合 (S7) — 模型: {model_list}",
                "info", DISCUSSION_STAGE,
            ))

    return messages


# ── WebSocket server ────────────────────────────────────────────────────────

async def broadcast(state: BridgeState, messages: list[dict]):
    if not messages or not state.clients:
        return
    dead = set()
    for msg in messages:
        data = json.dumps(msg, ensure_ascii=False)
        for ws in state.clients:
            try:
                await ws.send(data)
            except websockets.ConnectionClosed:
                dead.add(ws)
    state.clients -= dead


async def handle_command(state: BridgeState, data: dict) -> list[dict]:
    cmd = data.get("command")
    messages: list[dict] = []

    if cmd == "list_agents":
        for a in state.agents.values():
            messages.append(msg_agent_update(a))
        messages.append(msg_queue_update(state.queues))

    elif cmd == "add_lobster":
        name = data.get("name", f"🦞 龙虾-{_uid()}")
        layer = data.get("layer", "idea")
        agent = create_agent(state, name, layer)
        messages.append(msg_agent_update(agent))
        messages.append(msg_log(agent, f"龙虾已加入 {layer} 层", "info"))

    elif cmd == "remove_lobster":
        agent_id = data.get("agentId")
        agent = state.agents.pop(agent_id, None)
        if agent:
            messages.extend(stop_agent(agent))

    elif cmd == "submit_project":
        project_id = data.get("projectId") or f"proj-{_uid()}"
        config_path = data.get("configPath", "")
        topic = data.get("topic", "")
        messages.extend(submit_new_project(state, project_id, config_path, topic))
        messages.extend(schedule_idle_agents(state))

    elif cmd == "stop_agent":
        agent_id = data.get("agentId")
        agent = state.agents.get(agent_id)
        if agent:
            messages.extend(stop_agent(agent))

    elif cmd == "get_queues":
        messages.append(msg_queue_update(state.queues))

    elif cmd == "get_shared_results":
        if state.result_registry:
            messages.append({
                "type": "system",
                "payload": {"message": json.dumps(state.result_registry.summary(), ensure_ascii=False)},
            })

    elif cmd == "start_idea_factory":
        state.idea_factory_topic = data.get("topic", "")
        state.idea_factory_config = data.get("configPath", "")
        state.idea_factory_remaining = int(data.get("ideaCount", 0))
        state.idea_factory_produced = 0
        _sys_a = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        label = "无限" if state.idea_factory_remaining == -1 else str(state.idea_factory_remaining)
        messages.append(msg_log(_sys_a, f"Idea 工厂已启动: topic={state.idea_factory_topic[:50]}... count={label}", "info"))

    elif cmd == "stop_idea_factory":
        state.idea_factory_remaining = 0
        _sys_a = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        messages.append(msg_log(_sys_a, f"Idea 工厂已停止 (已产出 {state.idea_factory_produced} 个)", "info"))

    elif cmd == "set_discussion_mode":
        enabled = bool(data.get("enabled", False))
        state.discussion_mode = enabled
        rounds = data.get("rounds")
        if rounds is not None:
            state.discussion_rounds = int(rounds)
        _sys_a = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        label = "开启" if enabled else "关闭"
        messages.append(msg_log(_sys_a, f"沟通讨论模式已{label} (轮数={state.discussion_rounds})", "info", DISCUSSION_STAGE))

    return messages


async def ws_handler(state: BridgeState, websocket: websockets.ServerConnection):
    state.clients.add(websocket)
    print(f"[+] Client connected (total: {len(state.clients)})")
    for agent in state.agents.values():
        try:
            await websocket.send(json.dumps(msg_agent_update(agent), ensure_ascii=False))
        except websockets.ConnectionClosed:
            break
    try:
        await websocket.send(json.dumps(msg_queue_update(state.queues), ensure_ascii=False))
    except websockets.ConnectionClosed:
        pass

    try:
        async for raw in websocket:
            try:
                responses = await handle_command(state, json.loads(raw))
                await broadcast(state, responses)
            except json.JSONDecodeError:
                pass
    except websockets.ConnectionClosed:
        pass
    finally:
        state.clients.discard(websocket)
        print(f"[-] Client disconnected (total: {len(state.clients)})")


async def poll_loop(state: BridgeState, interval: float):
    while True:
        await asyncio.sleep(interval)
        all_messages: list[dict] = []

        for agent in list(state.agents.values()):
            prev_status = agent.status
            msgs = poll_agent(agent)
            all_messages.extend(msgs)

            # Detect layer completion → feed task queue
            if prev_status == "working" and agent.status == "done":
                if getattr(agent, '_is_idea_factory_s7_only', False):
                    all_messages.extend(_on_idea_factory_s7_done(state, agent))
                elif getattr(agent, '_is_idea_factory', False):
                    all_messages.extend(_on_idea_factory_done(state, agent))
                elif getattr(agent, '_is_discussion_s8', False):
                    all_messages.extend(_on_discussion_s8_done(state, agent))
                else:
                    all_messages.extend(on_agent_done(state, agent))

            # Detect failure → mark task failed, release GPU
            if prev_status == "working" and agent.status == "error":
                if agent.assigned_task_id:
                    for q in state.queues.values():
                        q.fail(agent.assigned_task_id)
                if agent.layer == "execution" and agent.project_id:
                    released = state.gpu_allocator.release(agent.project_id)
                    if released:
                        all_messages.append(msg_log(agent, f"GPU {released} 已释放 (错误后)", "warning"))
                # Clean up discussion group if S7-only idea factory agent failed
                _batch_id = getattr(agent, '_idea_factory_batch_id', None)
                if _batch_id and _batch_id in state.discussion_groups:
                    _grp = state.discussion_groups[_batch_id]
                    if agent.id in _grp.agent_ids:
                        _grp.agent_ids.remove(agent.id)
                        _grp.run_dirs.pop(agent.id, None)
                        _grp.completed_s7.discard(agent.id)
                    if len(_grp.agent_ids) < 2:
                        for _rem_aid in list(_grp.agent_ids):
                            _rem_a = state.agents.get(_rem_aid)
                            if _rem_a and _rem_a.status == "waiting_discussion":
                                _rem_a.status = "idle"
                                _rem_a.current_task = "沟通讨论取消（伙伴失败）"
                                _rem_a._is_idea_factory = False  # type: ignore[attr-defined]
                                all_messages.append(msg_agent_update(_rem_a))
                                all_messages.append(msg_log(_rem_a, "沟通讨论组人数不足，取消讨论", "warning", DISCUSSION_STAGE))
                        del state.discussion_groups[_batch_id]
                agent.assigned_task_id = None
                agent.project_id = ""
                agent.status = "idle"
                agent.current_task = "等待任务..."
                all_messages.append(msg_agent_update(agent))

        # Poll active discussions
        for group in list(state.discussion_groups.values()):
            all_messages.extend(_poll_discussion(state, group))

        # Schedule idle agents
        sched_msgs = schedule_idle_agents(state)
        all_messages.extend(sched_msgs)

        await broadcast(state, all_messages)


# ── Startup ─────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace):
    state = BridgeState(
        python_path=args.python,
        agent_package_dir=args.agent_dir,
        runs_base_dir=args.runs_dir,
        gpu_allocator=GpuAllocator(args.total_gpus, args.gpus_per_project),
        auto_loop=args.auto_loop,
        discussion_mode=args.discussion_mode,
        discussion_rounds=args.discussion_rounds,
        discussion_models=[m.strip() for m in args.discussion_models.split(",") if m.strip()],
        idea_factory_topic=args.idea_topic,
        idea_factory_config=args.idea_config,
        idea_factory_remaining=args.idea_count,
    )

    # Initialize shared results registry
    _shared_results_path = Path(state.runs_base_dir).parent / "shared_results"
    try:
        from result_registry import ResultRegistry
        state.result_registry = ResultRegistry(str(_shared_results_path))
    except Exception:
        pass

    state.projects_dir().mkdir(parents=True, exist_ok=True)
    state.queues_dir().mkdir(parents=True, exist_ok=True)

    # Initialize queues (load from disk)
    for queue_name in list(QUEUE_NAMES.keys()) + ["init_to_idea"]:
        q = TaskQueue(name=queue_name, path=state.queues_dir() / f"{queue_name}.json")
        q.load()
        state.queues[queue_name] = q

    # Create default lobster pool (configurable via --pool)
    pool_sizes = {"idea": args.pool_idea, "experiment": args.pool_exp,
                  "coding": args.pool_code, "execution": args.pool_exec,
                  "writing": args.pool_write}
    pool_names = {"idea": "调研", "experiment": "实验", "coding": "码农", "execution": "执行", "writing": "写作"}
    default_pool = []
    for layer, count in pool_sizes.items():
        for i in range(count):
            tag = chr(ord('A') + i) if count > 1 else ""
            default_pool.append((f"🦞 {pool_names[layer]}·{tag}".rstrip("·"), layer))
    for name, layer in default_pool:
        create_agent(state, name, layer)

    queued_tasks = sum(q.pending_count() for q in state.queues.values())

    print(f"🦞 Agent Bridge v2 starting on ws://0.0.0.0:{args.port}")
    print(f"   Agent package: {args.agent_dir}")
    print(f"   Runs base:     {args.runs_dir}")
    print(f"   Python:        {args.python}")
    print(f"   Lobsters:      {len(state.agents)}")
    print(f"   GPUs:          {args.total_gpus}x ({args.gpus_per_project}/project, max {args.total_gpus // max(args.gpus_per_project, 1)} parallel)")
    print(f"   Auto-loop:     {'ON' if args.auto_loop else 'OFF'}")
    _disc_info = f"ON ({args.discussion_rounds} rounds, models: {args.discussion_models})" if args.discussion_mode else "OFF"
    print(f"   Discussion:    {_disc_info}")
    print(f"   Queued tasks:  {queued_tasks}")
    print()

    handler = lambda ws: ws_handler(state, ws)
    async with websockets.serve(handler, "0.0.0.0", args.port):
        await poll_loop(state, args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Bridge v2")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--agent-dir",
                        default=str(Path(__file__).resolve().parent.parent / "agent"))
    parser.add_argument("--runs-dir",
                        default=str(Path(__file__).resolve().parent.parent / "runs"))
    parser.add_argument("--pool-idea", type=int, default=2)
    parser.add_argument("--pool-exp", type=int, default=2)
    parser.add_argument("--pool-code", type=int, default=3)
    parser.add_argument("--pool-exec", type=int, default=4)
    parser.add_argument("--pool-write", type=int, default=2)
    parser.add_argument("--total-gpus", type=int, default=8,
                        help="Total number of GPUs available")
    parser.add_argument("--gpus-per-project", type=int, default=2,
                        help="GPUs allocated per project in execution layer")
    parser.add_argument("--auto-loop", action="store_true", default=False,
                        help="Enable auto-loop: L4 completion feeds back to L1 for new research cycle")
    parser.add_argument("--discussion-mode", action="store_true", default=False,
                        help="Enable L1 discussion: agents discuss after S7 before generating hypotheses")
    parser.add_argument("--discussion-rounds", type=int, default=3,
                        help="Number of LLM discussion rounds (default: 3)")
    parser.add_argument("--discussion-models", default="gpt-5.3-codex-spark,claude-opus-4-6",
                        help="Comma-separated LLM models for discussion agents (different companies for diversity)")
    parser.add_argument("--idea-count", type=int, default=0,
                        help="Idea factory: number of ideas to produce (0=disabled, -1=infinite)")
    parser.add_argument("--idea-topic", default="",
                        help="Idea factory: research topic for idea generation")
    parser.add_argument("--idea-config", default="",
                        help="Idea factory: config file path")
    args = parser.parse_args()
    asyncio.run(main(args))
