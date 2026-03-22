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
    python agent_bridge.py [--port 8786] [--agent-dir /path/to/agent]
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
    9: "experiment", 11: "experiment",
    10: "coding",
    12: "execution", 13: "execution", 14: "execution", 15: "execution",
}

LAYER_STAGES: dict[str, list[int]] = {
    "idea": [1, 2, 3, 4, 5, 6, 7, 8],
    "experiment": [9],
    "coding": [10, 11],
    "execution": [12, 13, 14, 15],
}

LAYER_RANGE: dict[str, tuple[int, int]] = {
    "idea": (1, 8),
    "experiment": (9, 9),    # L2 runs S9 only (experiment design)
    "coding": (10, 11),      # L3 runs S10 (code gen) + S11 (resource planning)
    "execution": (12, 15),
}

PASSTHROUGH_LAYERS: set[str] = set()

STAGE_NAMES: dict[int, str] = {
    1: "TOPIC_INIT", 2: "PROBLEM_DECOMPOSE", 3: "SEARCH_STRATEGY",
    4: "LITERATURE_COLLECT", 5: "LITERATURE_SCREEN", 6: "KNOWLEDGE_EXTRACT",
    7: "SYNTHESIS", 8: "HYPOTHESIS_GEN", 9: "EXPERIMENT_DESIGN",
    10: "CODE_GENERATION", 11: "RESOURCE_PLANNING", 12: "EXPERIMENT_RUN",
    13: "ITERATIVE_REFINE", 14: "RESULT_ANALYSIS", 15: "RESEARCH_DECISION",
}

STAGE_OUTPUTS: dict[int, list[str]] = {
    1: ["goal.md", "hardware_profile.json"], 2: ["problem_tree.md"],
    3: ["search_plan.yaml", "sources.json", "queries.json"], 4: ["candidates.jsonl"],
    5: ["shortlist.jsonl"], 6: ["cards/"], 7: ["synthesis.md"], 8: ["hypotheses.md"],
    9: ["exp_plan.yaml"], 10: ["experiment/", "experiment_spec.md"],
    11: ["schedule.json"], 12: ["runs/"],
    13: ["refinement_log.json", "experiment_final/"],
    14: ["analysis.md", "experiment_summary.json", "charts/"], 15: ["decision.md"],
}

REPO_FOR_STAGE: dict[int, str] = {
    1: "knowledge", 2: "knowledge", 3: "knowledge", 4: "knowledge",
    5: "knowledge", 6: "knowledge", 7: "knowledge", 8: "knowledge",
    9: "exp_design", 11: "exp_design", 10: "codebase",
    12: "results", 13: "results", 14: "results", 15: "results",
}

# Queue names between layers
QUEUE_NAMES: dict[str, tuple[str, str]] = {
    "idea_to_experiment":     ("idea",       "experiment"),
    "experiment_to_coding":   ("experiment", "coding"),
    "coding_to_execution":    ("coding",     "execution"),
    "execution_feedback":     ("execution",  "idea"),
}

# Which queue a completing layer feeds into
LAYER_OUTPUT_QUEUE: dict[str, str] = {
    "idea":       "idea_to_experiment",
    "experiment": "experiment_to_coding",
    "coding":     "coding_to_execution",
    "execution":  "execution_feedback",
}

# Which queue a layer pulls tasks from
LAYER_INPUT_QUEUE: dict[str, str] = {
    "experiment": "idea_to_experiment",
    "coding":     "experiment_to_coding",
    "execution":  "coding_to_execution",
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

def _save_feedback(state: "BridgeState", content: str, target_layer: str, message_id: str) -> None:
    """Persist human feedback to disk so running agents can pick it up.

    Writes to:
    1. Global ``runs_base/feedback/feedback_log.jsonl`` — full audit trail
    2. Each matching project's ``run_dir/human_feedback.jsonl`` — consumed by
       the executor's ``_load_human_feedback()`` before each pipeline stage
    """
    feedback_dir = Path(state.runs_base_dir) / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": message_id,
        "content": content,
        "targetLayer": target_layer,
        "timestamp": _now_ms(),
    }
    log_path = feedback_dir / "feedback_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    latest_path = feedback_dir / "latest_feedback.json"
    _write_json(latest_path, entry)

    injected_to: list[str] = []
    for agent in state.agents.values():
        if not agent.run_dir or agent.status not in ("working", "idle"):
            continue
        if target_layer != "all" and agent.layer != target_layer:
            continue
        run_dir = Path(agent.run_dir)
        if not run_dir.exists():
            continue
        fb_path = run_dir / "human_feedback.jsonl"
        try:
            with open(fb_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            injected_to.append(f"{agent.name}({agent.project_id})")
        except OSError:
            pass

    if injected_to:
        print(f"[feedback] Injected to {len(injected_to)} project(s): {', '.join(injected_to)}")


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


@dataclass
class BridgeState:
    agents: dict[str, LobsterAgent] = field(default_factory=dict)
    queues: dict[str, TaskQueue] = field(default_factory=dict)
    clients: set = field(default_factory=set)
    python_path: str = ""
    agent_package_dir: str = ""
    runs_base_dir: str = ""
    llm_client: object | None = field(default=None, repr=False)
    llm_config_path: str = ""

    def projects_dir(self) -> Path:
        return Path(self.runs_base_dir) / "projects"

    def queues_dir(self) -> Path:
        return Path(self.runs_base_dir) / "queues"


# ── LLM integration for feedback analysis ───────────────────────────────────

def _init_llm_client(agent_package_dir: str, config_path: str = "") -> object | None:
    """Try to initialize an LLM client from the ResearchClaw config."""
    try:
        _agent_dir = Path(agent_package_dir)
        if str(_agent_dir) not in sys.path:
            sys.path.insert(0, str(_agent_dir))

        from researchclaw.config import load_config
        from researchclaw.llm.client import LLMClient, LLMConfig

        if config_path and Path(config_path).exists():
            rc_config = load_config(config_path, check_paths=False)
            client = LLMClient.from_rc_config(rc_config)
        else:
            candidates = [
                _agent_dir / "config_gpu_project.yaml",
                _agent_dir / "config.researchclaw.yaml",
                _agent_dir / "config.researchclaw.example.yaml",
            ]
            for c in candidates:
                if c.exists():
                    rc_config = load_config(str(c), check_paths=False)
                    client = LLMClient.from_rc_config(rc_config)
                    print(f"   LLM config: {c}")
                    return client
            return None

        return client
    except Exception as e:
        print(f"[warn] LLM client init failed: {e}")
        return None


def _gather_pipeline_context(state: BridgeState, target_layer: str) -> str:
    """Collect current pipeline state for LLM analysis."""
    parts: list[str] = []

    for agent in state.agents.values():
        if target_layer != "all" and agent.layer != target_layer:
            continue
        status_cn = {"idle": "空闲", "working": "运行中", "error": "出错", "done": "完成"}.get(agent.status, agent.status)
        line = f"- {agent.name} [{agent.layer}层] 状态={status_cn}"
        if agent.current_stage:
            sname = STAGE_NAMES.get(agent.current_stage, f"S{agent.current_stage}")
            line += f" 当前阶段=S{agent.current_stage}({sname})"
        if agent.current_task:
            line += f" 任务={agent.current_task}"
        if agent.project_id:
            line += f" 项目={agent.project_id}"
        parts.append(line)

        if agent.run_dir and Path(agent.run_dir).exists():
            cp = _read_json(Path(agent.run_dir) / "checkpoint.json")
            if cp:
                parts.append(f"  checkpoint: 已完成到 S{cp.get('last_completed_stage', '?')} ({cp.get('last_completed_name', '')})")

    for qname, q in state.queues.items():
        s = q.summary()
        if s["total"] > 0:
            parts.append(f"- 队列 {qname}: 总={s['total']} 待处理={s['pending']} 进行中={s['assigned']} 完成={s['completed']}")

    return "\n".join(parts) if parts else "当前无活跃的 Agent 或任务。"


_FEEDBACK_SYSTEM_PROMPT = """\
你是「Pyramid Research Team」的智能调度助手。用户（人类研究员）通过前端对话框提供了反馈或指令。

你需要：
1. 理解用户的反馈意图
2. 结合当前 pipeline 运行状态，分析反馈对研究计划的影响
3. 给出具体的计划调整建议（哪些阶段需要重新执行、参数如何调整、方向是否需要改变等）
4. 用简洁明确的中文回复

回复格式要求：
- 先用一句话确认理解了用户的反馈
- 然后给出具体的计划调整建议
- 最后说明这些调整将如何生效
"""


async def _process_feedback_with_llm(
    state: BridgeState,
    content: str,
    target_layer: str,
    message_id: str,
) -> dict | None:
    """Call LLM to analyze human feedback and generate plan update."""
    if state.llm_client is None:
        return None

    context = _gather_pipeline_context(state, target_layer)
    target_desc = "全局" if target_layer == "all" else f"{target_layer}层"

    user_prompt = (
        f"## 当前 Pipeline 状态\n{context}\n\n"
        f"## 人类研究员反馈\n"
        f"目标层级: {target_desc}\n"
        f"反馈内容: {content}\n\n"
        f"请分析这条反馈并给出计划调整建议。"
    )

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: state.llm_client.chat(
                [{"role": "user", "content": user_prompt}],
                system=_FEEDBACK_SYSTEM_PROMPT,
                max_tokens=1024,
            ),
        )
        reply_text = resp.content.strip()

        plan_lines = []
        for line in reply_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("-", "•", "·", "1", "2", "3", "4", "5")) or "调整" in stripped or "建议" in stripped:
                plan_lines.append(stripped)
        plan_summary = "\n".join(plan_lines[:5]) if plan_lines else reply_text[:200]

        return msg_plan_update(reply_text, target_layer, plan_summary)

    except Exception as e:
        print(f"[warn] LLM feedback analysis failed: {e}")
        return None


# ── Message builders ────────────────────────────────────────────────────────

def msg_agent_update(agent: LobsterAgent) -> dict:
    return {"type": "agent_update", "payload": agent.to_frontend()}

def msg_stage_update(agent_id: str, stage: int, status: str) -> dict:
    return {"type": "stage_update", "payload": {"agentId": agent_id, "stage": stage, "status": status}}

def msg_artifact(repo_id: str, filename: str, agent_name: str, size: str, project_id: str = "") -> dict:
    return {"type": "artifact_produced", "payload": {
        "id": _uid(), "repoId": repo_id, "projectId": project_id, "filename": filename,
        "producedBy": agent_name, "timestamp": _now_ms(), "size": size, "status": "fresh",
    }}

def msg_log(agent: LobsterAgent, message: str, level: str = "info", stage: int | None = None) -> dict:
    return {"type": "log", "payload": {
        "id": _uid(), "agentId": agent.id, "agentName": agent.name,
        "layer": agent.layer, "stage": stage or agent.current_stage,
        "message": message, "level": level, "timestamp": _now_ms(),
    }}

def msg_queue_update(queues: dict[str, TaskQueue]) -> dict:
    return {"type": "queue_update", "payload": {name: q.summary() for name, q in queues.items()}}

def msg_feedback_ack(message_id: str, content: str, target_layer: str = "all", plan_update: str = "") -> dict:
    return {"type": "feedback_ack", "payload": {
        "id": f"sys-{_uid()}", "sender": "system", "content": content,
        "timestamp": _now_ms(), "targetLayer": target_layer, "planUpdate": plan_update,
    }}

def msg_plan_update(content: str, target_layer: str = "all", plan_update: str = "") -> dict:
    return {"type": "plan_update", "payload": {
        "id": f"plan-{_uid()}", "sender": "system", "content": content,
        "timestamp": _now_ms(), "targetLayer": target_layer, "planUpdate": plan_update,
    }}


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
                    messages.append(msg_artifact(
                        REPO_FOR_STAGE.get(s, "knowledge"), expected, agent.name, size, agent.project_id,
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
        layer_range = LAYER_RANGE.get(agent.layer, (1, 15))

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

    if agent.process is not None:
        retcode = agent.process.poll()
        if retcode is not None:
            # Final read: catch any checkpoint/artifact updates written before exit
            layer_range = LAYER_RANGE.get(agent.layer, (1, 15))
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
        log_path = Path(task.run_dir) / f"agent_{agent.id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=state.agent_package_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
    """Submit a brand-new project — goes into init_to_idea queue."""
    messages: list[dict] = []
    run_dir = str(state.projects_dir() / project_id)
    os.makedirs(run_dir, exist_ok=True)

    task = Task(
        id=f"task-{_uid()}", project_id=project_id, run_dir=run_dir,
        config_path=config_path, topic=topic,
        source_layer="init", target_layer="idea",
        created_at=_now_ms(),
    )
    state.queues["init_to_idea"].push(task)

    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
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

    # Create follow-up task in the next queue
    output_queue_name = LAYER_OUTPUT_QUEUE.get(agent.layer)
    if output_queue_name and output_queue_name in state.queues and agent.project_id:
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

    # Reset agent for next task
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
        if agent.status != "idle" or agent.process is not None:
            continue
        if agent.assigned_task_id:
            continue

        # Idea agents pull from init_to_idea AND execution_feedback
        if agent.layer == "idea":
            candidate_queues = ["init_to_idea", "execution_feedback"]
        else:
            q_name = LAYER_INPUT_QUEUE.get(agent.layer, "")
            candidate_queues = [q_name] if q_name else []

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
            break

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

    elif cmd == "human_feedback":
        content = data.get("content", "")
        target_layer = data.get("targetLayer", "all")
        message_id = data.get("messageId", "")

        sys_agent = LobsterAgent(
            id="system", name="系统", layer=target_layer if target_layer != "all" else "idea",
            run_id="", run_dir="", config_path="",
        )
        messages.append(msg_log(sys_agent, f"收到人工反馈: {content[:80]}{'...' if len(content) > 80 else ''}", "info"))

        _save_feedback(state, content, target_layer, message_id)

        injected_projects = []
        for agent in state.agents.values():
            if not agent.run_dir or agent.status not in ("working", "idle"):
                continue
            if target_layer != "all" and agent.layer != target_layer:
                continue
            if agent.project_id:
                injected_projects.append(agent.project_id)

        if state.llm_client is not None:
            ack_text = "正在分析你的反馈，请稍候..."
            if injected_projects:
                unique = sorted(set(injected_projects))
                ack_text = f"已注入 {len(unique)} 个项目，正在调用大模型分析反馈..."
            messages.append(msg_feedback_ack(message_id, ack_text, target_layer, ""))
            asyncio.ensure_future(_async_llm_feedback(state, content, target_layer, message_id))
        else:
            target_desc = "全局" if target_layer == "all" else STAGE_NAMES.get(
                LAYER_STAGES.get(target_layer, [0])[0], target_layer
            )
            if injected_projects:
                unique = sorted(set(injected_projects))
                plan_hint = (
                    f"已将反馈注入 {len(unique)} 个项目的 prompt 上下文中 "
                    f"({', '.join(unique)})。"
                    f"当前阶段完成后，下一个阶段的 LLM 将读取并参考你的反馈来调整执行计划。"
                    f"\n(未配置 LLM，无法提供即时智能分析)"
                )
            else:
                plan_hint = (
                    f"已记录针对 [{target_desc}] 的反馈。"
                    f"当前无匹配的运行中项目，反馈将在新任务启动时生效。"
                )
            messages.append(msg_feedback_ack(message_id, plan_hint, target_layer, plan_hint))

    return messages


async def _async_llm_feedback(state: BridgeState, content: str, target_layer: str, message_id: str):
    """Background task: call LLM to analyze feedback, then broadcast the result."""
    try:
        result = await _process_feedback_with_llm(state, content, target_layer, message_id)
        if result:
            await broadcast(state, [result])
        else:
            fallback = msg_plan_update(
                "LLM 分析暂不可用，你的反馈已保存并将在下一个阶段自动注入 prompt 中。",
                target_layer,
                "反馈已保存，等待下一阶段执行时生效。",
            )
            await broadcast(state, [fallback])
    except Exception as e:
        print(f"[error] Async LLM feedback failed: {e}")
        fallback = msg_plan_update(
            f"分析反馈时出错: {e}。你的反馈已保存，将在下一个阶段自动生效。",
            target_layer,
            "反馈已保存，分析出错。",
        )
        await broadcast(state, [fallback])


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
                all_messages.extend(on_agent_done(state, agent))

            # Detect failure → mark task failed
            if prev_status == "working" and agent.status == "error":
                if agent.assigned_task_id:
                    for q in state.queues.values():
                        q.fail(agent.assigned_task_id)
                agent.assigned_task_id = None
                agent.status = "idle"
                agent.current_task = "等待任务..."
                all_messages.append(msg_agent_update(agent))

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
    )
    state.projects_dir().mkdir(parents=True, exist_ok=True)
    state.queues_dir().mkdir(parents=True, exist_ok=True)

    # Initialize queues (load from disk)
    for queue_name in list(QUEUE_NAMES.keys()) + ["init_to_idea"]:
        q = TaskQueue(name=queue_name, path=state.queues_dir() / f"{queue_name}.json")
        q.load()
        state.queues[queue_name] = q

    # Create default lobster pool (configurable via --pool)
    pool_sizes = {"idea": args.pool_idea, "experiment": args.pool_exp,
                  "coding": args.pool_code, "execution": args.pool_exec}
    pool_names = {"idea": "调研", "experiment": "实验", "coding": "码农", "execution": "执行"}
    default_pool = []
    for layer, count in pool_sizes.items():
        for i in range(count):
            tag = chr(ord('A') + i) if count > 1 else ""
            default_pool.append((f"🦞 {pool_names[layer]}·{tag}".rstrip("·"), layer))
    for name, layer in default_pool:
        create_agent(state, name, layer)

    queued_tasks = sum(q.pending_count() for q in state.queues.values())

    state.llm_config_path = args.llm_config
    llm = _init_llm_client(args.agent_dir, args.llm_config)
    state.llm_client = llm

    print(f"🦞 Agent Bridge v2 starting on ws://0.0.0.0:{args.port}")
    print(f"   Agent package: {args.agent_dir}")
    print(f"   Runs base:     {args.runs_dir}")
    print(f"   Python:        {args.python}")
    print(f"   Lobsters:      {len(state.agents)}")
    print(f"   Queued tasks:  {queued_tasks}")
    print(f"   LLM feedback:  {'✅ 已启用' if llm else '❌ 未配置 (--llm-config)'}")
    print()

    handler = lambda ws: ws_handler(state, ws)
    async with websockets.serve(handler, "0.0.0.0", args.port):
        await poll_loop(state, args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Bridge v2")
    parser.add_argument("--port", type=int, default=8786)
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
    parser.add_argument("--llm-config", default="",
                        help="Path to ResearchClaw YAML config for feedback LLM")
    args = parser.parse_args()
    asyncio.run(main(args))
