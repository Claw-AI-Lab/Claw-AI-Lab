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
import base64
import hashlib
import json
import os
import re
import shutil
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
    19: ["outline.md"], 20: ["paper_draft.md"], 21: ["reviews.md"], 22: ["paper_revised.md", "latex_package.zip"],
}

# Curated artifacts to display in the frontend DataShelf (subset of STAGE_OUTPUTS)
DISPLAY_ARTIFACTS: set[str] = {
    # Idea 仓库 — only S8 hypotheses
    "hypotheses.md",
    # 知识库
    "knowledge_entry.json",
    # 论文仓库 — final paper + LaTeX package
    "paper_revised.md", "latex_package.zip",
    # 结果
    "analysis.md", "charts/", "decision.md",
    # 实验设计
    "exp_plan.yaml",
    # 代码
    "experiment/", "experiment_spec.md", "experiment_final/",
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

# ── Human Feedback Persistence ───────────────────────────────────────────────

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


_intent_llm_client: "object | None" = None
_intent_llm_init_done: bool = False

_INTENT_SYSTEM_PROMPT = (
    "你是一个意图分类器。用户在一个 AI 研究 pipeline 的控制面板中输入了一条消息。"
    "判断这条消息是【查询】(想了解当前运行状态/进度/阶段) 还是【反馈】(想给 pipeline 提供指导/建议/修改指令)。"
    "只回复一个词: query 或 feedback"
)


def _init_intent_llm(state: "BridgeState") -> None:
    """Lazily create a lightweight LLM client for intent classification."""
    global _intent_llm_client, _intent_llm_init_done
    if _intent_llm_init_done:
        return
    _intent_llm_init_done = True
    try:
        agent_dir = state.agent_package_dir
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        from researchclaw.llm.client import LLMClient, LLMConfig

        api_key = os.environ.get("RESEARCHCLAW_API_KEY", "")

        # Read api_key from project YAML configs if not in env
        if not api_key:
            import yaml as _yaml
            for proj_dir in state.projects_dir().iterdir():
                meta = _read_json(proj_dir / "project_meta.json")
                if not meta or not meta.get("config_path"):
                    continue
                cfg_path = Path(meta["config_path"])
                if not cfg_path.exists():
                    continue
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        raw = _yaml.safe_load(f) or {}
                    llm_sec = raw.get("llm", {})
                    api_key = llm_sec.get("api_key", "")
                    if api_key:
                        break
                except Exception:
                    continue

        if not api_key:
            print("[intent-llm] No API key found, using keyword fallback")
            return

        from researchclaw.llm import resolve_provider_base_url
        base_url = resolve_provider_base_url("openai-compatible", "")
        _intent_llm_client = LLMClient(LLMConfig(
            base_url=base_url,
            api_key=api_key,
            primary_model="claude-opus-4-1-20250805",
            fallback_models=["gpt-4o-mini"],
            max_retries=1,
            timeout_sec=10,
        ))
        print(f"[intent-llm] Initialized ({base_url})")
    except Exception as exc:
        print(f"[intent-llm] Init failed, will use keyword fallback: {exc}")


def _classify_chat_intent_keywords(text: str) -> str:
    """Fast keyword-based fallback for intent classification."""
    t = text.lower()
    q, f = 0, 0
    for kw in ("状态", "进度", "进展", "阶段", "跑到", "做到", "到哪", "到第几",
               "什么阶段", "什么状态", "查看", "查询", "怎么样了", "情况",
               "status", "progress", "stage", "how far"):
        if kw in t:
            q += 1
    for kw in ("请", "应该", "建议", "不要", "换成", "改成", "使用",
               "注意", "确保", "调整", "修改", "尝试", "模型", "参数",
               "checkpoint", "路径", "下载"):
        if kw in t:
            f += 1
    if t.rstrip()[-1:] in ("?", "？"):
        q += 2
    if any(p in t for p in ("吗", "呢")):
        q += 1
    if len(t) > 80:
        f += 1
    return "query" if q > f else "feedback"


async def _classify_chat_intent(text: str, state: "BridgeState") -> str:
    """Classify user chat intent: LLM first, keyword fallback on failure."""
    _init_intent_llm(state)
    if _intent_llm_client is not None:
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _intent_llm_client.chat(  # type: ignore[union-attr]
                    [{"role": "user", "content": text}],
                    system=_INTENT_SYSTEM_PROMPT,
                    max_tokens=10,
                    temperature=0,
                ),
            )
            answer = resp.content.strip().lower()
            if "query" in answer:
                return "query"
            if "feedback" in answer:
                return "feedback"
        except Exception as exc:
            print(f"[intent-llm] Call failed, falling back to keywords: {exc}")
    return _classify_chat_intent_keywords(text)


def _pause_project(state: "BridgeState", project_id: str) -> list[dict]:
    """Pause a running project: stop agents and remove queued tasks, but keep all files."""
    messages: list[dict] = []
    sys_agent = LobsterAgent(
        id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="",
    )

    if not project_id:
        return messages

    stopped = 0
    for agent in list(state.agents.values()):
        if agent.project_id == project_id:
            if agent.process is not None and agent.process.poll() is None:
                agent.process.terminate()
                try:
                    agent.process.wait(timeout=5)
                except Exception:
                    agent.process.kill()
                stopped += 1
            _reset_agent_idle(agent)
            messages.append(msg_agent_update(agent))

    removed = 0
    for q in state.queues.values():
        before = len(q.tasks)
        q.tasks = [t for t in q.tasks if t.project_id != project_id]
        removed += before - len(q.tasks)

    released = state.gpu_allocator.release(project_id)
    if released:
        messages.append(msg_log(sys_agent, f"GPU {released} 已释放 (项目暂停)", "info"))

    messages.append(msg_log(
        sys_agent,
        f"项目 [{project_id}] 已暂停 (停止 {stopped} 个 Agent, 移除 {removed} 个队列任务)",
        "warning",
    ))
    return messages


def _restart_project(state: "BridgeState", project_id: str) -> list[dict]:
    """Restart a project from scratch: stop agents, clear checkpoint, re-submit."""
    import shutil

    messages: list[dict] = []
    sys_agent = LobsterAgent(
        id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="",
    )

    if not project_id:
        return messages

    proj_dir = state.projects_dir() / project_id
    if not proj_dir.exists():
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 不存在", "error"))
        return messages

    messages.extend(_pause_project(state, project_id))

    meta = _read_json(proj_dir / "project_meta.json")
    config_path = meta.get("config_path", "") if meta else ""
    topic = meta.get("topic", "") if meta else ""
    mode = meta.get("mode", "lab") if meta else "lab"

    if not config_path:
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 缺少配置文件路径, 无法重启", "error"))
        return messages

    # Lab mode with run-* sub-dirs: clear progress inside each angle but keep structure
    angle_dirs = sorted(proj_dir.glob("run-*"))
    if mode == "lab" and angle_dirs:
        for angle_dir in angle_dirs:
            if not angle_dir.is_dir():
                continue
            for item in angle_dir.iterdir():
                if item.name in ("project_meta.json",):
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        # Also clean any non-run files at project root (except meta)
        for item in proj_dir.iterdir():
            if item.name == "project_meta.json" or item.name.startswith("run-"):
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 已清除进度，正在重新启动…", "info"))
        messages.extend(resume_project(state, project_id))
    else:
        for item in proj_dir.iterdir():
            if item.name == "project_meta.json":
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 已清除进度，正在重新启动…", "info"))
        messages.extend(submit_new_project(state, project_id, config_path, topic, mode=mode))
    return messages


def _delete_project(state: "BridgeState", project_id: str) -> list[dict]:
    """Delete a project: stop any running processes, clean up state, remove files."""
    import shutil

    messages: list[dict] = []
    sys_agent = LobsterAgent(
        id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="",
    )

    if not project_id:
        messages.append(msg_feedback_ack(f"del-{_uid()}", "请指定要删除的项目 ID。"))
        return messages

    for agent in list(state.agents.values()):
        if agent.project_id == project_id:
            if agent.process is not None and agent.process.poll() is None:
                agent.process.terminate()
                try:
                    agent.process.wait(timeout=5)
                except Exception:
                    agent.process.kill()
            _reset_agent_idle(agent)
            messages.append(msg_agent_update(agent))

    released = state.gpu_allocator.release(project_id)
    if released:
        messages.append(msg_log(sys_agent, f"GPU {released} 已释放 (项目删除)", "info"))

    for q in state.queues.values():
        q.tasks = [t for t in q.tasks if t.project_id != project_id]

    state._fail_counts.pop(project_id, None)

    # Clean up discussion state
    for aid in list(state.discussion_waiting):
        a = state.discussion_waiting[aid]
        if a.project_id == "" or a.project_id == project_id:
            state.discussion_waiting.pop(aid, None)
    state.discussion_groups.pop(project_id, None)

    proj_dir = state.projects_dir() / project_id
    if proj_dir.exists() and proj_dir.is_dir():
        try:
            shutil.rmtree(proj_dir)
            messages.append(msg_log(sys_agent, f"项目 [{project_id}] 已删除", "success"))
        except OSError as exc:
            messages.append(msg_log(sys_agent, f"删除项目目录失败: {exc}", "error"))
    else:
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 目录不存在", "warning"))

    return messages


def _build_status_summary(state: "BridgeState", target_layer: str = "all") -> str:
    """Build a human-readable status summary for all running/recent projects."""
    lines: list[str] = []
    projects_dir = state.projects_dir()

    active_projects: dict[str, LobsterAgent] = {}
    for agent in state.agents.values():
        if agent.project_id and agent.status in ("working", "idle"):
            active_projects[agent.project_id] = agent

    project_dirs = sorted(
        (d for d in projects_dir.iterdir() if d.is_dir() and not d.name.startswith("_")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    ) if projects_dir.is_dir() else []

    if not project_dirs:
        return "当前没有任何项目。"

    for proj_dir in project_dirs[:5]:
        pid = proj_dir.name
        if target_layer != "all" and pid not in active_projects:
            continue

        lines.append(f"📋 项目: {pid}")

        agent = active_projects.get(pid)
        if agent:
            layer_cn = {"idea": "调研", "experiment": "设计", "coding": "编码",
                        "execution": "执行", "writing": "写作"}.get(agent.layer, agent.layer)
            status_cn = {"working": "运行中", "idle": "空闲", "error": "错误",
                         "done": "完成"}.get(agent.status, agent.status)
            lines.append(f"  状态: {status_cn} | 层: {layer_cn}")
            if agent.current_stage:
                sname = STAGE_NAMES.get(agent.current_stage, "?")
                lines.append(f"  当前阶段: S{agent.current_stage} {sname}")
        else:
            lines.append("  状态: 未活跃")

        stage_statuses = []
        for s in range(1, 23):
            health = _read_json(proj_dir / f"stage-{s:02d}" / "stage_health.json")
            if health:
                st = health.get("status", "?")
                dur = health.get("duration_sec")
                err = health.get("error")
                icon = "✅" if st == "done" else "❌" if st == "failed" else "🔄"
                sname = STAGE_NAMES.get(s, "?")
                entry = f"  {icon} S{s} {sname}"
                if dur is not None:
                    if dur < 60:
                        entry += f" ({dur:.0f}s)"
                    else:
                        entry += f" ({dur / 60:.1f}min)"
                if err:
                    entry += f" — {err[:60]}"
                stage_statuses.append(entry)

        if stage_statuses:
            last_done = [s for s in stage_statuses if "✅" in s]
            failed = [s for s in stage_statuses if "❌" in s]
            lines.append(f"  已完成: {len(last_done)}/22 阶段" + (f", {len(failed)} 失败" if failed else ""))
            for s in stage_statuses:
                lines.append(s)
        else:
            lines.append("  暂无阶段数据")

        heartbeat = _read_json(proj_dir / "heartbeat.json")
        if heartbeat:
            ts = heartbeat.get("timestamp", "")
            lines.append(f"  最后心跳: {ts}")

        lines.append("")

    gpu_info = state.gpu_allocator.summary()
    lines.append(f"🖥️ GPU: {gpu_info['free']}/{gpu_info['total']} 空闲")
    if gpu_info["assignments"]:
        for proj, gpus in gpu_info["assignments"].items():
            lines.append(f"  {proj} → GPU {gpus}")

    return "\n".join(lines)


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
    completed_s8: set[str] = field(default_factory=set)       # agent_ids done with S8
    best_agent_id: str = ""
    status: str = "gathering"    # gathering | waiting | discussing | done
    discussion_process: subprocess.Popen | None = field(default=None, repr=False)
    discussion_output_dir: str = ""

    def all_ready(self) -> bool:
        return len(self.completed_s7) >= len(self.agent_ids) and len(self.agent_ids) >= 2

    def all_s8_done(self) -> bool:
        return len(self.completed_s8) >= len(self.agent_ids) and len(self.agent_ids) >= 2

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
    role_tag: str = ""
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
            "projectId": self.project_id,
            "roleTag": self.role_tag,
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
    discussion_mode: bool = True
    discussion_groups: dict[str, DiscussionGroup] = field(default_factory=dict)
    discussion_rounds: int = 3
    discussion_models: list[str] = field(default_factory=lambda: ["gpt-5.3-codex-spark", "claude-opus-4-6"])
    # Cross-project discussion: agents waiting for a peer to discuss with
    discussion_waiting: dict[str, "LobsterAgent"] = field(default_factory=dict)
    # Idea factory: L1 idle → produce ideas via S7+S8
    idea_factory_topic: str = ""
    idea_factory_config: str = ""
    idea_factory_remaining: int = 0  # 0=disabled, -1=infinite, N=count
    idea_factory_produced: int = 0
    _fail_counts: dict[str, int] = field(default_factory=dict)  # project_id → consecutive fail count
    # Lab mode: track which sub-projects belong to the same batch
    lab_batches: dict[str, list[str]] = field(default_factory=dict)  # base_id → [sub_project_ids]

    def projects_dir(self) -> Path:
        return Path(self.runs_base_dir) / "projects"

    def queues_dir(self) -> Path:
        return Path(self.runs_base_dir) / "queues"


# ── Message builders ────────────────────────────────────────────────────────

def msg_agent_update(agent: LobsterAgent) -> dict:
    return {"type": "agent_update", "payload": agent.to_frontend()}

def msg_stage_update(agent_id: str, stage: int, status: str) -> dict:
    return {"type": "stage_update", "payload": {"agentId": agent_id, "stage": stage, "status": status}}

def msg_artifact(repo_id: str, filename: str, agent_name: str, size: str,
                  project_id: str = "", content: str = "", stage: int = 0) -> dict:
    payload: dict = {
        "id": _uid(), "repoId": repo_id, "projectId": project_id, "filename": filename,
        "producedBy": agent_name, "timestamp": _now_ms(), "size": size, "status": "fresh",
    }
    if content:
        payload["content"] = content
    if stage:
        payload["stage"] = stage
    return {"type": "artifact_produced", "payload": payload}


_NO_CONTENT_ARTIFACTS: set[str] = {"paper_revised.md", "paper_draft.md", "outline.md", "reviews.md"}

def _extract_artifact_summary(path: Path, filename: str, max_chars: int = 500) -> str:
    """Extract a human-readable summary from an artifact file."""
    if filename in _NO_CONTENT_ARTIFACTS:
        return ""
    try:
        if path.is_dir():
            children = list(path.iterdir())
            file_count = sum(1 for c in children if c.is_file())
            md_titles = []
            for c in sorted(children)[:8]:
                if c.suffix == ".md" and c.is_file():
                    first_line = c.read_text(encoding="utf-8", errors="ignore").strip().split("\n")[0]
                    title = first_line.lstrip("#").strip()
                    if title:
                        md_titles.append(title)
            if md_titles:
                return f"{file_count} files: " + "; ".join(md_titles[:6])
            return f"{file_count} files"

        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return ""

        if filename == "hypotheses.md":
            lines = text.strip().split("\n")
            hyp_titles = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("##") and "hypothesis" in stripped.lower():
                    title = stripped.lstrip("#").strip().lstrip("—").lstrip("-").strip()
                    title = title.removeprefix("Final Hypothesis").strip()
                    title = title.removeprefix("Hypothesis").strip()
                    title = title.lstrip("0123456789").strip().lstrip("—").lstrip("-").strip()
                    title = title.replace("**", "").replace("*", "")
                    if title:
                        hyp_titles.append(f"• {title}")
            if hyp_titles:
                return "\n".join(hyp_titles)[:max_chars]

        if filename.endswith(".md"):
            lines = text.strip().split("\n")
            summary_parts = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if summary_parts:
                        break
                    continue
                clean = stripped.lstrip("#").strip().rstrip("---").strip()
                if clean:
                    summary_parts.append(clean)
                if len(" ".join(summary_parts)) > max_chars:
                    break
            return " ".join(summary_parts)[:max_chars]

        if filename.endswith((".yaml", ".yml")):
            lines = text.strip().split("\n")
            top_keys = []
            for line in lines[:15]:
                if line and not line.startswith(" ") and not line.startswith("#") and ":" in line:
                    top_keys.append(line.split(":")[0].strip())
            return f"keys: {', '.join(top_keys[:8])}" if top_keys else ""

        if filename.endswith(".json"):
            data = json.loads(text)
            if isinstance(data, dict):
                # knowledge_entry.json: extract topic + hypothesis names
                if "topic" in data and "hypotheses" in data and isinstance(data["hypotheses"], list):
                    topic = str(data["topic"])[:120]
                    hyp_names = []
                    for h in data["hypotheses"][:5]:
                        if isinstance(h, dict):
                            name = h.get("name") or h.get("id", "")
                            status = h.get("status", "")
                            entry = str(name)[:60]
                            if status:
                                entry += f" ({status[:20]})"
                            hyp_names.append(entry)
                    summary = topic
                    if hyp_names:
                        summary += "\n" + "\n".join(f"• {n}" for n in hyp_names)
                    return summary[:max_chars]

                keys = list(data.keys())[:8]
                preview_parts = []
                for k in keys[:4]:
                    v = data[k]
                    if isinstance(v, str) and len(v) < 80:
                        preview_parts.append(f"{k}: {v}")
                    elif isinstance(v, (int, float, bool)):
                        preview_parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        preview_parts.append(f"{k}: [{len(v)} items]")
                return "; ".join(preview_parts) if preview_parts else f"keys: {', '.join(keys)}"
            if isinstance(data, list):
                return f"{len(data)} entries"

        if filename.endswith(".jsonl"):
            line_count = text.count("\n")
            first_line = text.strip().split("\n")[0] if text.strip() else ""
            if first_line:
                try:
                    entry = json.loads(first_line)
                    title = entry.get("title") or entry.get("name") or entry.get("id", "")
                    if title:
                        return f"{line_count} entries — first: {str(title)[:80]}"
                except Exception:
                    pass
            return f"{line_count} entries"

    except Exception:
        pass
    return ""

def msg_log(agent: LobsterAgent, message: str, level: str = "info", stage: int | None = None) -> dict:
    return {"type": "log", "payload": {
        "id": _uid(), "agentId": agent.id, "agentName": agent.name,
        "layer": agent.layer, "stage": stage or agent.current_stage,
        "message": message, "level": level, "timestamp": _now_ms(),
    }}

def msg_queue_update(queues: dict[str, TaskQueue]) -> dict:
    return {"type": "queue_update", "payload": {name: q.summary() for name, q in queues.items()}}

def msg_project_list(projects: list[dict]) -> dict:
    return {"type": "project_list", "payload": projects}


def msg_feedback_ack(message_id: str, content: str, target_layer: str = "all", plan_update: str = "") -> dict:
    return {"type": "chat_message", "payload": {
        "id": f"sys-{_uid()}", "role": "system", "content": content,
        "timestamp": _now_ms(), "targetLayer": target_layer,
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
                    if expected not in DISPLAY_ARTIFACTS:
                        continue
                    size = "dir" if artifact_path.is_dir() else f"{artifact_path.stat().st_size / 1024:.1f} KB"
                    content = _extract_artifact_summary(artifact_path, expected)
                    messages.append(msg_artifact(
                        REPO_FOR_STAGE.get(s, "knowledge"), expected, agent.name, size, agent.project_id, content, stage=s,
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
    if not hasattr(agent, '_base_name'):
        agent._base_name = agent.name  # type: ignore[attr-defined]

    agent.project_id = task.project_id
    agent.run_dir = task.run_dir
    agent.run_id = task.project_id
    agent.config_path = task.config_path
    agent.assigned_task_id = task.id
    agent._topic = task.topic
    agent.status = "working"
    layer_stages = LAYER_STAGES.get(agent.layer, [])
    agent.stage_progress = {s: "pending" for s in layer_stages}
    agent.current_stage = layer_stages[0] if layer_stages else 0
    agent.current_task = f"准备执行 [{task.project_id}]"
    agent._prev_heartbeat = {}
    agent._prev_checkpoint = {}
    agent._known_artifacts = set()

    # Lab mode: extract role tag from topic pattern "[RoleName] topic"
    import re as _re
    _role_match = _re.match(r"^\[(.+?)\]\s", task.topic or "")
    agent.role_tag = _role_match.group(1) if _role_match else ""


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
                    if expected not in DISPLAY_ARTIFACTS:
                        continue
                    size = "dir" if artifact_path.is_dir() else f"{artifact_path.stat().st_size / 1024:.1f} KB"
                    content = _extract_artifact_summary(artifact_path, expected)
                    messages.append(msg_artifact(
                        REPO_FOR_STAGE.get(s, "codebase"), expected, agent.name, size, agent.project_id, content, stage=s,
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
    # Reproduce mode skips discussion → runs full S1-S8
    _task_meta = _read_project_meta(task.run_dir) if task.run_dir else None
    _is_reproduce = _task_meta.get("mode") == "reproduce" if _task_meta else False
    if state.discussion_mode and agent.layer == "idea" and not _is_reproduce:
        layer_range = LAYER_RANGE_PHASE1["idea"]
    else:
        layer_range = LAYER_RANGE.get(agent.layer, (1, 15))
    fs, ts = layer_range

    # Checkpoint-aware resume: skip already-completed stages within this layer
    cp = _read_json(Path(task.run_dir) / "checkpoint.json")
    if cp:
        last_done = cp.get("last_completed_stage", 0)
        resume_stage = last_done + 1

        # Discussion mode: if S1-S7 already done, skip straight to discussion/S8
        if state.discussion_mode and agent.layer == "idea" and not _is_reproduce and last_done >= ts:
            for s in range(fs, ts + 1):
                agent.stage_progress[s] = "completed"
            messages.append(msg_log(
                agent,
                f"S1-S7 已完成 (checkpoint={last_done}), 跳过重跑 → 直接进入讨论/S8",
                "info",
            ))
            messages.extend(_skip_discussion_proceed_s8(state, agent))
            return messages

        if fs <= resume_stage <= ts:
            for s in range(fs, resume_stage):
                if s in STAGE_TO_LAYER:
                    agent.stage_progress[s] = "completed"
            fs = resume_stage
            agent.current_stage = fs
            agent.current_task = f"断点恢复 → {STAGE_NAMES.get(fs, f'S{fs}')}"
            messages.append(msg_agent_update(agent))
            messages.append(msg_log(
                agent,
                f"断点恢复: 跳过已完成阶段, 从 {STAGE_NAMES.get(fs, f'S{fs}')} 开始",
                "info",
            ))

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
        proc_env.pop("CUDA_VISIBLE_DEVICES", None)

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

def _save_project_meta(run_dir: str, project_id: str, config_path: str, topic: str, mode: str = "lab") -> None:
    """Persist project metadata so it can be recovered on restart."""
    meta = {
        "project_id": project_id,
        "config_path": config_path,
        "topic": topic,
        "mode": mode,
        "created_at": _now_ms(),
    }
    meta_path = Path(run_dir) / "project_meta.json"
    if not meta_path.exists():
        _write_json(meta_path, meta)


def _read_project_meta(run_dir: str) -> dict | None:
    return _read_json(Path(run_dir) / "project_meta.json")


def _determine_resume_target(run_dir: str) -> tuple[str, int] | None:
    """Read checkpoint and return (target_layer, next_stage) or None if no checkpoint."""
    cp = _read_json(Path(run_dir) / "checkpoint.json")
    if not cp:
        return None
    last_done = cp.get("last_completed_stage", 0)
    if last_done <= 0:
        return None
    next_stage = last_done + 1
    if next_stage > 22:
        return None
    target_layer = STAGE_TO_LAYER.get(next_stage)
    if not target_layer:
        return None
    return (target_layer, next_stage)


def _queue_for_layer(target_layer: str) -> str:
    """Return the input queue name that feeds into target_layer."""
    if target_layer == "idea":
        return "init_to_idea"
    return LAYER_INPUT_QUEUE.get(target_layer, "init_to_idea")


def submit_new_project(state: BridgeState, project_id: str, config_path: str, topic: str = "", mode: str = "lab") -> list[dict]:
    """Submit a project — auto-detects checkpoint and resumes from where it left off.

    In cross-project discussion mode, each project gets ONE agent.
    After S7, agents from different projects discuss with each other.
    """
    messages: list[dict] = []
    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")

    run_dir = str(state.projects_dir() / project_id)
    os.makedirs(run_dir, exist_ok=True)
    _save_project_meta(run_dir, project_id, config_path, topic, mode=mode)

    if state.discussion_mode:
        messages.append(msg_log(
            sys_agent,
            f"新项目 [{project_id}] 跨 project 讨论模式: 分配 1 个 agent, S7 后与其他 project agent 讨论",
            "info", DISCUSSION_STAGE,
        ))

    # Check for checkpoint to enable resume
    resume_info = _determine_resume_target(run_dir)
    if resume_info:
        target_layer, next_stage = resume_info
        queue_name = _queue_for_layer(target_layer)
        source_layer = {
            "idea": "init", "experiment": "idea", "coding": "experiment",
            "execution": "coding", "writing": "execution",
        }.get(target_layer, "init")

        task = Task(
            id=f"task-{_uid()}", project_id=project_id, run_dir=run_dir,
            config_path=config_path, topic=topic,
            source_layer=source_layer, target_layer=target_layer,
            created_at=_now_ms(),
        )
        state.queues[queue_name].push(task)
        stage_name = STAGE_NAMES.get(next_stage, f"S{next_stage}")
        messages.append(msg_log(
            sys_agent,
            f"项目 [{project_id}] 检测到断点 → 从 {stage_name} (Stage {next_stage}) 恢复",
            "success",
        ))
    else:
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


def _check_s12_sanity_failure(state: "BridgeState", agent: "LobsterAgent") -> list[dict]:
    """If S12 sanity_report.json shows fail, pause the project and return notification messages."""
    messages: list[dict] = []
    _s12_dir = Path(agent.run_dir) / "stage-12"
    _sanity_path = _s12_dir / "sanity_report.json"
    if not _sanity_path.exists():
        return messages
    try:
        _sanity = json.loads(_sanity_path.read_text(encoding="utf-8"))
    except Exception:
        return messages
    if _sanity.get("status") != "fail":
        return messages

    _fix_log_path = _s12_dir / "fix_log.json"
    _exp_dir = None
    for _sd in sorted(Path(agent.run_dir).glob("stage-11*"), reverse=True):
        _ed = _sd / "experiment"
        if _ed.exists():
            _exp_dir = str(_ed)
            break

    _last_error = ""
    _iters = _sanity.get("iterations", [])
    if _iters:
        _last_iter = _iters[-1]
        _failed_checks = [c for c in _last_iter.get("checks", []) if not c.get("passed")]
        if _failed_checks:
            _fc = _failed_checks[-1]
            _last_error = (_fc.get("stderr_tail") or _fc.get("stderr") or "")[-800:]

    _detail = (
        f"⚠️ S12 SANITY_CHECK 循环修复失败，需要手动介入\n"
        f"项目: {agent.project_id}\n"
        f"修复轮次: {_sanity.get('total_iterations', '?')}/{_sanity.get('max_fix_iterations', '?')}\n"
        f"实验代码: {_exp_dir or 'N/A'}\n"
        f"修复日志: {_fix_log_path}\n"
        f"检查报告: {_sanity_path}"
    )
    if _last_error:
        _detail += f"\n最后报错:\n{_last_error}"

    # Persist intervention reason so the frontend can display it
    _meta_path = Path(agent.run_dir) / "project_meta.json"
    if _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            _meta["intervention"] = _detail
            _meta_path.write_text(json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    messages.append(msg_log(agent, _detail, "error", 12))
    messages.extend(_pause_project(state, agent.project_id))
    messages.append(msg_project_list(list_all_projects(state)))
    return messages


def on_agent_done(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """When an agent finishes, complete its task and create a follow-up task for the next layer."""
    messages: list[dict] = []

    state._fail_counts.pop(agent.project_id, None)  # reset fail counter on success

    # Complete assigned task
    if agent.assigned_task_id:
        for q in state.queues.values():
            q.complete(agent.assigned_task_id)

    # Discussion mode: L1 agent completed S7 → enter discussion
    # Reproduce mode skips discussion entirely
    _agent_proj_dir = state.projects_dir() / agent.project_id if agent.project_id else None
    _agent_meta = _read_project_meta(str(_agent_proj_dir)) if _agent_proj_dir and _agent_proj_dir.exists() else None
    _agent_is_reproduce = _agent_meta.get("mode") == "reproduce" if _agent_meta else False
    if state.discussion_mode and agent.layer == "idea" and not _agent_is_reproduce:
        state.discussion_waiting[agent.id] = agent
        agent.current_stage = DISCUSSION_STAGE
        agent.stage_progress[DISCUSSION_STAGE] = "running"
        agent.status = "waiting_discussion"
        agent.current_task = "S7 完成，等待讨论伙伴..."
        messages.append(msg_agent_update(agent))

        pid = agent.project_id
        expected_count = state.lab_batches.get(pid, 0)

        if expected_count >= 2:
            # Lab mode: same project_id, wait for all N agents to finish S7
            waiting_same_proj = [
                a for a in state.discussion_waiting.values()
                if a.project_id == pid
            ]
            if len(waiting_same_proj) >= expected_count:
                messages.append(msg_log(
                    agent,
                    f"项目 [{pid}] 全部 {len(waiting_same_proj)} 个方向 S7 完成 → 启动跨领域讨论",
                    "info", DISCUSSION_STAGE,
                ))
                group = DiscussionGroup(
                    project_id=pid,
                    topic=getattr(agent, '_topic', '') or agent.current_task,
                    config_path=agent.config_path,
                    agent_ids=[a.id for a in waiting_same_proj],
                    run_dirs={a.id: a.run_dir for a in waiting_same_proj},
                )
                for a in waiting_same_proj:
                    state.discussion_waiting.pop(a.id, None)
                    group.completed_s7.add(a.id)
                state.discussion_groups[pid] = group
                messages.extend(_trigger_discussion(state, group))
            else:
                messages.append(msg_log(
                    agent,
                    f"S7 完成，等待同项目其他方向 ({len(waiting_same_proj)}/{expected_count})",
                    "info", DISCUSSION_STAGE,
                ))
            return messages

        # Non-Lab: cross-project pair discussion (original logic)
        # 1) Find another agent that also completed S7 (different project)
        peers = [a for a in state.discussion_waiting.values()
                 if a.id != agent.id and a.project_id != agent.project_id]

        if peers:
            peer = peers[0]
            messages.append(msg_log(agent, f"S7 完成，与 [{peer.project_id}] 的 agent 开始跨 project 讨论", "info", DISCUSSION_STAGE))
            messages.extend(_trigger_cross_project_discussion(state, agent, peer))
            return messages

        # 2) Find an idle agent (no project, can act as reviewer/critic)
        idle_agents = [
            a for a in state.agents.values()
            if a.layer == "idea" and a.status == "idle"
            and a.id != agent.id and not a.assigned_task_id
        ]
        if idle_agents:
            reviewer = idle_agents[0]
            reviewer.status = "discussing"
            reviewer.current_stage = DISCUSSION_STAGE
            reviewer.current_task = f"讨论评审: [{agent.project_id}]"
            reviewer.stage_progress[DISCUSSION_STAGE] = "running"
            messages.append(msg_agent_update(reviewer))
            messages.append(msg_log(agent, f"S7 完成，与空闲 agent [{reviewer.name}] 开始讨论评审", "info", DISCUSSION_STAGE))
            messages.extend(_trigger_cross_project_discussion(state, agent, reviewer))
            return messages

        # 3) No peer available at all — skip discussion
        messages.append(msg_log(agent, "S7 完成，无可用讨论伙伴，跳过讨论直接进入 S8", "info", DISCUSSION_STAGE))
        messages.extend(_skip_discussion_proceed_s8(state, agent))
        return messages

    # L3 (coding) → check S12 sanity_report: if failed, pause and notify user
    if agent.layer == "coding" and agent.project_id:
        _s12_msgs = _check_s12_sanity_failure(state, agent)
        if _s12_msgs:
            messages.extend(_s12_msgs)
            return messages

    # Create follow-up task in the next queue
    output_queue_name = LAYER_OUTPUT_QUEUE.get(agent.layer)
    if output_queue_name and output_queue_name in state.queues and agent.project_id:
        # L4→L5: only push if S17 PROCEED
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
    _reset_agent_idle(agent)
    messages.append(msg_agent_update(agent))
    messages.append(msg_queue_update(state.queues))

    return messages


def _reset_agent_idle(agent: LobsterAgent) -> None:
    """Reset agent to idle state, clearing all project-related fields."""
    agent.assigned_task_id = None
    agent.project_id = ""
    agent.status = "idle"
    agent.current_task = "等待任务..."
    agent.run_id = ""
    agent.run_dir = ""
    agent.config_path = ""
    agent.current_stage = 0
    agent.stage_progress = {}
    agent.role_tag = ""
    agent.process = None
    base = getattr(agent, '_base_name', None)
    if base:
        agent.name = base


def list_all_projects(state: BridgeState) -> list[dict]:
    """Scan runs/projects/ and return status info for all projects."""
    projects_dir = state.projects_dir()
    result: list[dict] = []
    if not projects_dir.exists():
        return result

    running_project_ids: set[str] = set()
    for a in state.agents.values():
        if a.project_id and a.process is not None and a.process.poll() is None:
            running_project_ids.add(a.project_id)

    queued_project_ids: set[str] = set()
    for q in state.queues.values():
        for t in q.tasks:
            if t.status in ("pending", "assigned") and t.project_id:
                queued_project_ids.add(t.project_id)

    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir() or proj_dir.name.startswith("_"):
            continue
        project_id = proj_dir.name
        meta = _read_json(proj_dir / "project_meta.json")
        project_mode = meta.get("mode", "lab") if meta else "lab"

        # Read checkpoint: top-level first, then aggregate from sub-runs (Lab mode)
        cp = _read_json(proj_dir / "checkpoint.json")
        if not cp:
            best_cp = None
            best_stage = 0
            for sub in proj_dir.iterdir():
                if sub.is_dir() and sub.name.startswith("run-"):
                    sub_cp = _read_json(sub / "checkpoint.json")
                    if sub_cp and sub_cp.get("last_completed_stage", 0) > best_stage:
                        best_stage = sub_cp.get("last_completed_stage", 0)
                        best_cp = sub_cp
            cp = best_cp

        last_stage = cp.get("last_completed_stage", 0) if cp else 0
        last_name = cp.get("last_completed_name", "") if cp else ""
        timestamp = cp.get("timestamp", "") if cp else ""

        first_stage = 1
        total_stages = 22
        completed_threshold = total_stages

        if last_stage >= completed_threshold:
            status = "completed"
        elif project_id in running_project_ids:
            status = "running"
        elif project_id in queued_project_ids:
            status = "queued"
        elif last_stage > 0:
            status = "interrupted"
        else:
            status = "new"

        topic = ""
        config_path = ""
        if meta:
            topic = meta.get("topic", "")
            config_path = meta.get("config_path", "")
        if not topic:
            goal_path = proj_dir / "stage-01" / "goal.md"
            if not goal_path.exists():
                for sub in proj_dir.iterdir():
                    if sub.is_dir() and sub.name.startswith("run-"):
                        goal_path = sub / "stage-01" / "goal.md"
                        if goal_path.exists():
                            break
            if goal_path.exists():
                try:
                    topic = goal_path.read_text(encoding="utf-8")[:300]
                except OSError:
                    pass

        intervention = meta.get("intervention", "") if meta else ""

        result.append({
            "projectId": project_id,
            "status": status,
            "lastCompletedStage": last_stage,
            "lastCompletedName": last_name,
            "firstStage": first_stage,
            "totalStages": total_stages,
            "timestamp": timestamp,
            "topic": topic,
            "configPath": config_path,
            "intervention": intervention,
        })

    return result


def resume_project(state: BridgeState, project_id: str) -> list[dict]:
    """Resume a project from its last checkpoint."""
    messages: list[dict] = []
    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")

    state._fail_counts.pop(project_id, None)  # reset fail counter on manual resume

    proj_dir = state.projects_dir() / project_id
    if not proj_dir.exists():
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 不存在", "error"))
        return messages

    for a in state.agents.values():
        if a.project_id == project_id and a.process is not None and a.process.poll() is None:
            messages.append(msg_log(sys_agent, f"项目 [{project_id}] 已在运行中", "warning"))
            return messages

    meta = _read_json(proj_dir / "project_meta.json")
    config_path = meta.get("config_path", "") if meta else ""
    topic = meta.get("topic", "") if meta else ""
    mode = meta.get("mode", "lab") if meta else "lab"

    # Clear intervention flag on resume
    if meta and meta.get("intervention"):
        meta.pop("intervention", None)
        try:
            (proj_dir / "project_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    if not config_path:
        messages.append(msg_log(sys_agent, f"项目 [{project_id}] 缺少配置文件路径, 无法恢复", "error"))
        return messages

    # Lab mode with run-* sub-directories: resume each angle separately
    angle_dirs = sorted(proj_dir.glob("run-*"))
    if mode == "lab" and angle_dirs:
        task_count = 0
        for angle_dir in angle_dirs:
            if not angle_dir.is_dir():
                continue
            slug = angle_dir.name.removeprefix("run-")
            angle_config_key = f"{project_id}--{slug}"
            angle_config = str(Path(state.runs_base_dir) / "project_configs" / f"{angle_config_key}.yaml")
            if not Path(angle_config).exists():
                angle_config = config_path

            run_dir = str(angle_dir)
            resume_info = _determine_resume_target(run_dir)
            if resume_info:
                target_layer, next_stage = resume_info
                queue_name = _queue_for_layer(target_layer)
                source_layer = {
                    "idea": "init", "experiment": "idea", "coding": "experiment",
                    "execution": "coding", "writing": "execution",
                }.get(target_layer, "init")
                stage_name = STAGE_NAMES.get(next_stage, f"S{next_stage}")
                messages.append(msg_log(
                    sys_agent,
                    f"  方向 [{slug}] 断点恢复 → {stage_name} (Stage {next_stage})",
                    "success",
                ))
            else:
                queue_name = "init_to_idea"
                source_layer = "init"
                target_layer = "idea"
                messages.append(msg_log(sys_agent, f"  方向 [{slug}] 从头开始", "info"))

            task = Task(
                id=f"task-{_uid()}",
                project_id=project_id,
                run_dir=run_dir,
                config_path=angle_config,
                topic=f"[{slug}] {topic}",
                source_layer=source_layer,
                target_layer=target_layer,
                created_at=_now_ms(),
            )
            state.queues[queue_name].push(task)
            task_count += 1

        if task_count >= 2:
            state.lab_batches[project_id] = task_count

        messages.append(msg_queue_update(state.queues))
        messages.append(msg_project_list(list_all_projects(state)))
        messages.append(msg_log(
            sys_agent,
            f"Lab 模式: 项目 [{project_id}] — {task_count} 个方向已恢复",
            "success",
        ))
        return messages

    messages.extend(submit_new_project(state, project_id, config_path, topic, mode=mode))
    return messages


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn arbitrary text into a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = text.strip('-')[:max_len].rstrip('-')
    if not text:
        text = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return text


def _generate_config_from_template(
    state: BridgeState, project_id: str, topic: str, role_prompt: str = "",
    reference_papers: list[str] | None = None,
    codebases_dir: str = "", datasets_dir: str = "", checkpoints_dir: str = "",
) -> str:
    """Generate a project-specific YAML config from the default template.

    If role_prompt is provided (Lab mode), it's prepended to the topic so the
    pipeline agent operates from that specialist perspective.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    template_path = repo_root / "examples" / "config_template.yaml"
    if not template_path.exists():
        template_path = Path(state.agent_package_dir).parent / "config_template.yaml"
    if not template_path.exists():
        template_path = Path(__file__).resolve().parent.parent / "config_template.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found at {template_path}")

    full_topic = f"{role_prompt}\n\n研究主题: {topic}" if role_prompt else topic

    content = template_path.read_text(encoding="utf-8")
    content = content.replace("__PROJECT_ID__", project_id)
    content = content.replace("__TOPIC__", full_topic.replace('"', '\\"'))

    if reference_papers:
        yaml_list = "\n".join(f'    - "{p}"' for p in reference_papers)
        content = content.replace("  reference_papers: __REFERENCE_PAPERS__",
                                  f"  reference_papers:\n{yaml_list}")
    else:
        content = content.replace("  reference_papers: __REFERENCE_PAPERS__",
                                  "  reference_papers: []")

    import re as _re
    if codebases_dir:
        content = _re.sub(r'(codebases_dir:\s*)"[^"]*"', f'\\1"{codebases_dir}"', content)
    if datasets_dir:
        content = _re.sub(r'(datasets_dir:\s*)"[^"]*"', f'\\1"{datasets_dir}"', content)
    if checkpoints_dir:
        content = _re.sub(r'(checkpoints_dir:\s*)"[^"]*"', f'\\1"{checkpoints_dir}"', content)

    configs_dir = Path(state.runs_base_dir) / "project_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{project_id}.yaml"
    config_path.write_text(content, encoding="utf-8")
    return str(config_path)


def _safe_reference_upload_name(filename: str) -> str:
    base = Path(filename or "reference.pdf").name
    cleaned = re.sub(r"[^\w.\-]+", "_", base).strip("._")
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned or 'reference'}.pdf"
    return cleaned


def _persist_reference_uploads(
    project_dir: Path,
    reference_uploads: list[dict[str, str]] | None,
) -> list[str]:
    if not reference_uploads:
        return []

    upload_dir = project_dir / "reference_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for item in reference_uploads:
        if not isinstance(item, dict):
            continue
        name = _safe_reference_upload_name(str(item.get("name", "reference.pdf")))
        content_b64 = str(item.get("contentBase64", "")).strip()
        if not content_b64:
            continue
        try:
            raw = base64.b64decode(content_b64, validate=True)
        except Exception:
            continue
        stem = Path(name).stem
        suffix = Path(name).suffix or ".pdf"
        target = upload_dir / f"{stem}-{uuid.uuid4().hex[:6]}{suffix}"
        target.write_bytes(raw)
        saved_paths.append(str(target.resolve()))

    return saved_paths


KNOWN_LAB_ANGLES: dict[str, str] = {
    "CV": (
        "你是实验室的「计算机视觉 (CV)」方向研究员。"
        "你的专长是图像识别、目标检测、语义分割、图像生成、视频理解。"
        "请从 CV 的视角进行深入调研，"
        "重点关注: 视觉骨干网络（ViT/CNN/Mamba）、"
        "自监督/对比学习、生成模型（Diffusion/GAN/Flow Matching）、"
        "3D 视觉、视频时序建模、以及 CV 在多模态与具身场景中的应用。"
    ),
    "VLM": (
        "你是实验室的「视觉语言模型 (VLM)」方向研究员。"
        "你的专长是多模态理解、视觉-语言对齐、图文推理、视觉 Grounding。"
        "请从 VLM 的视角进行深入调研，"
        "重点关注: 视觉编码器选型、跨模态融合架构、指令微调策略、"
        "视觉推理能力评估、以及 VLM 在具身场景中的感知与决策应用。"
    ),
    "World Model": (
        "你是实验室的「世界模型 (World Model)」方向研究员。"
        "你的专长是环境建模、视频预测、物理仿真、因果推理。"
        "请从 World Model 的视角进行深入调研，"
        "重点关注: 世界模型的架构设计（自回归/扩散/状态空间）、"
        "时空表征学习、动力学建模、长时序预测、"
        "以及世界模型在具身智能中的规划与想象能力。"
    ),
    "VLA": (
        "你是实验室的「视觉-语言-动作模型 (VLA)」方向研究员。"
        "你的专长是端到端策略学习、动作生成、机器人操作、模仿学习。"
        "请从 VLA 的视角进行深入调研，"
        "重点关注: VLA 模型架构（RT-2、OpenVLA、π₀ 等）、"
        "动作 tokenization 与解码策略、多任务泛化、"
        "sim-to-real 迁移、以及 VLA 在真实机器人上的部署与评估。"
    ),
}

DEFAULT_LAB_ANGLES: list[dict[str, str]] = [
    {"name": "CV", "prompt": KNOWN_LAB_ANGLES["CV"]},
]


def _build_role_prompt(angle_name: str, main_topic: str) -> str:
    """Build a role prompt for a Lab mode agent. Uses predefined prompts for known
    angles, otherwise generates a reasonable prompt from the angle name."""
    if angle_name in KNOWN_LAB_ANGLES:
        return KNOWN_LAB_ANGLES[angle_name]
    return (
        f"你是实验室的「{angle_name}」方向研究员。"
        f"请从 {angle_name} 的专业视角对研究主题进行深入调研，"
        f"重点关注该方向最相关的理论、方法、数据集和最新进展。"
    )


def quick_submit_project(
    state: BridgeState, topic: str, project_id: str = "",
    mode: str = "lab",
    research_angles: list[str] | None = None,
    reference_papers: list[str] | None = None,
    reference_uploads: list[dict[str, str]] | None = None,
    path_overrides: dict[str, str] | None = None,
) -> list[dict]:
    """Create a project from a topic string.

    Modes:
      - "lab": Multi-angle parallel research (default). If no angles provided,
        uses 3 default perspectives. Each agent gets a specialized role prompt.
      - "reproduce": Single-agent focused pipeline for paper reproduction.
    """
    messages: list[dict] = []
    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")

    if not topic.strip():
        messages.append(msg_log(sys_agent, "请输入研究主题", "error"))
        return messages

    base_id = project_id or _slugify(topic)

    # ── Reproduce mode: single-agent standard pipeline ──
    if mode == "reproduce":
        if not project_id:
            existing = state.projects_dir() / base_id
            if existing.exists():
                base_id = f"{base_id}-{_uid()[:4]}"
        project_dir = state.projects_dir() / base_id
        project_dir.mkdir(parents=True, exist_ok=True)
        saved_reference_paths = _persist_reference_uploads(project_dir, reference_uploads)
        all_reference_papers = [*(reference_papers or []), *saved_reference_paths]
        _po = path_overrides or {}
        try:
            config_path = _generate_config_from_template(
                state, base_id, topic.strip(),
                reference_papers=all_reference_papers,
                codebases_dir=_po.get("codebases_dir", ""),
                datasets_dir=_po.get("datasets_dir", ""),
                checkpoints_dir=_po.get("checkpoints_dir", ""),
            )
        except Exception as e:
            messages.append(msg_log(sys_agent, f"配置生成失败: {e}", "error"))
            return messages
        if saved_reference_paths:
            messages.append(msg_log(sys_agent, f"已接收 {len(saved_reference_paths)} 个本地 PDF 参考文件", "info"))
        messages.append(msg_log(sys_agent, f"复现模式: 项目 [{base_id}] 单 Agent 全流程启动", "success"))
        messages.extend(submit_new_project(state, base_id, config_path, topic.strip(), mode="reproduce"))
        return messages

    # ── Lab mode: parallel research (ONE project, N agents — or 1 agent if single direction) ──
    angles: list[dict[str, str]]
    if research_angles and len(research_angles) >= 1:
        angles = [
            {"name": a.strip(), "prompt": _build_role_prompt(a.strip(), topic.strip())}
            for a in research_angles if a.strip()
        ]
    if not research_angles or not angles:
        angles = DEFAULT_LAB_ANGLES

    # Deduplicate project id
    existing = state.projects_dir() / base_id
    if existing.exists():
        base_id = f"{base_id}-{_uid()[:4]}"

    project_dir = state.projects_dir() / base_id
    project_dir.mkdir(parents=True, exist_ok=True)
    saved_reference_paths = _persist_reference_uploads(project_dir, reference_uploads)
    all_reference_papers = [*(reference_papers or []), *saved_reference_paths]

    _po = path_overrides or {}
    try:
        project_config_path = _generate_config_from_template(
            state, base_id, topic.strip(),
            reference_papers=all_reference_papers,
            codebases_dir=_po.get("codebases_dir", ""),
            datasets_dir=_po.get("datasets_dir", ""),
            checkpoints_dir=_po.get("checkpoints_dir", ""),
        )
    except Exception as e:
        messages.append(msg_log(sys_agent, f"配置生成失败: {e}", "error"))
        return messages

    _save_project_meta(str(project_dir), base_id, project_config_path, topic.strip(), mode="lab")

    messages.append(msg_log(
        sys_agent,
        f"Lab 模式: 项目 [{base_id}] — {len(angles)} 个方向并行调研",
        "info",
    ))
    if saved_reference_paths:
        messages.append(msg_log(sys_agent, f"已接收 {len(saved_reference_paths)} 个本地 PDF 参考文件", "info"))

    task_count = 0
    for i, angle in enumerate(angles):
        name = angle["name"]
        role_prompt = angle["prompt"]
        slug = _slugify(name, 20)

        # Each direction gets its own run sub-directory within the project
        run_dir = str(project_dir / f"run-{slug}")
        os.makedirs(run_dir, exist_ok=True)

        _po = path_overrides or {}
        try:
            config_path = _generate_config_from_template(
                state, f"{base_id}--{slug}", topic.strip(), role_prompt,
                reference_papers=all_reference_papers,
                codebases_dir=_po.get("codebases_dir", ""),
                datasets_dir=_po.get("datasets_dir", ""),
                checkpoints_dir=_po.get("checkpoints_dir", ""),
            )
        except Exception as e:
            messages.append(msg_log(sys_agent, f"配置生成失败 [{name}]: {e}", "error"))
            continue

        task = Task(
            id=f"task-{_uid()}",
            project_id=base_id,
            run_dir=run_dir,
            config_path=config_path,
            topic=f"[{name}] {topic.strip()}",
            source_layer="init",
            target_layer="idea",
            created_at=_now_ms(),
        )
        state.queues["init_to_idea"].push(task)
        task_count += 1

        messages.append(msg_log(
            sys_agent,
            f"  方向 {i+1}/{len(angles)}: {name}",
            "info",
        ))

    # Register Lab batch: same project_id, expect N agents to finish S7
    if task_count >= 2:
        state.lab_batches[base_id] = task_count

    messages.append(msg_queue_update(state.queues))
    messages.append(msg_project_list(list_all_projects(state)))
    messages.append(msg_log(
        sys_agent,
        f"{task_count} 个方向 agent S7 完成后将自动讨论 → 合并为统一假设 → 进入 L2",
        "success",
    ))
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
    _reset_agent_idle(agent)
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
        _reset_agent_idle(agent)
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


def _trigger_cross_project_discussion(
    state: BridgeState, agent1: LobsterAgent, agent2: LobsterAgent,
) -> list[dict]:
    """Launch a discussion between two agents from different projects."""
    messages: list[dict] = []

    for a in (agent1, agent2):
        a.status = "discussing"
        a.current_task = f"跨 project 讨论: {agent1.project_id} × {agent2.project_id}"
        messages.append(msg_agent_update(a))

    p1_id = agent1.project_id or agent1.name
    p2_id = agent2.project_id or agent2.name
    disc_name = f"{p1_id}_x_{p2_id}"
    disc_dir = str(state.projects_dir() / "_cross_discussions" / disc_name)
    os.makedirs(disc_dir, exist_ok=True)

    synthesis_dirs = []
    for a in (agent1, agent2):
        s7 = Path(a.run_dir) / "stage-07" if a.run_dir else None
        if s7 and s7.exists():
            synthesis_dirs.append(str(s7))

    group = DiscussionGroup(
        project_id=disc_name,
        topic=f"{p1_id} | {p2_id}",
        config_path=agent1.config_path,
    )
    group.agent_ids = [agent1.id, agent2.id]
    group.run_dirs = {agent1.id: agent1.run_dir, agent2.id: agent2.run_dir}
    group.status = "discussing"
    group.discussion_output_dir = disc_dir
    group._cross_project = True  # type: ignore[attr-defined]

    runner_path = str(Path(__file__).resolve().parent / "discussion_runner.py")
    cmd = [
        state.python_path, runner_path,
        "--config", agent1.config_path,
        "--synthesis-dirs", *synthesis_dirs,
        "--output", disc_dir,
        "--rounds", str(state.discussion_rounds),
        "--topic", group.topic,
    ]

    try:
        log_path = Path(disc_dir) / "discussion.log"
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=state.agent_package_dir,
            stdout=log_file, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        group.discussion_process = proc
        state.discussion_groups[disc_name] = group

        sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        messages.append(msg_log(
            sys_agent,
            f"跨 project 讨论开始: [{agent1.project_id}] × [{agent2.project_id}], {state.discussion_rounds} 轮 (PID={proc.pid})",
            "info", DISCUSSION_STAGE,
        ))
    except Exception as e:
        sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")
        messages.append(msg_log(sys_agent, f"跨 project 讨论启动失败: {e}", "error", DISCUSSION_STAGE))
        for a in (agent1, agent2):
            state.discussion_waiting.pop(a.id, None)
        messages.extend(_skip_discussion_proceed_s8(state, agent1))
        messages.extend(_skip_discussion_proceed_s8(state, agent2))

    return messages


def _skip_discussion_proceed_s8(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """Skip discussion and proceed directly to S8 for a single agent."""
    messages: list[dict] = []
    state.discussion_waiting.pop(agent.id, None)

    agent.stage_progress[DISCUSSION_STAGE] = "completed"
    messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "completed"))

    fs, ts = LAYER_RANGE_PHASE2["idea"]
    agent.status = "working"
    agent.current_task = f"项目 {agent.project_id} · S8 假设生成 (跳过讨论)"
    agent.stage_progress[8] = "running"
    messages.append(msg_agent_update(agent))
    messages.append(msg_stage_update(agent.id, 8, "running"))
    messages.append(msg_log(agent, "跳过讨论 → 直接启动 S8 假设生成", "info", 8))

    cmd = [
        state.python_path, "-m", "researchclaw", "run",
        "--config", agent.config_path,
        "--output", agent.run_dir,
        "--from-stage", STAGE_NAMES.get(fs, str(fs)),
        "--to-stage", STAGE_NAMES.get(ts, str(ts)),
        "--auto-approve",
        "--skip-preflight",
    ]
    if agent.project_id:
        cmd.extend(["--topic", agent.project_id])

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

    is_cross = getattr(group, "_cross_project", False)

    if retcode != 0:
        group.status = "done"
        messages.append(msg_log(sys_agent, f"讨论 [{group.project_id}] 失败 (exit={retcode})", "error", DISCUSSION_STAGE))
        for aid in group.agent_ids:
            agent = state.agents.get(aid)
            if agent:
                state.discussion_waiting.pop(aid, None)
                if is_cross:
                    messages.append(msg_log(agent, "讨论失败，跳过讨论直接进入 S8", "warning", DISCUSSION_STAGE))
                    messages.extend(_skip_discussion_proceed_s8(state, agent))
                else:
                    agent.status = "error"
                    agent.current_task = f"沟通讨论失败 (exit={retcode})"
                    agent.stage_progress[DISCUSSION_STAGE] = "failed"
                    messages.append(msg_agent_update(agent))
                    messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "failed"))
        return messages

    consensus_file = Path(group.discussion_output_dir) / "consensus_synthesis.md"
    if not consensus_file.exists():
        messages.append(msg_log(sys_agent, f"讨论 [{group.project_id}] 完成但未产生共识", "warning", DISCUSSION_STAGE))
        group.status = "done"
        for aid in group.agent_ids:
            agent = state.agents.get(aid)
            if agent:
                state.discussion_waiting.pop(aid, None)
                has_project = bool(agent.run_dir and (Path(agent.run_dir) / "stage-07").exists())
                if has_project:
                    messages.extend(_skip_discussion_proceed_s8(state, agent))
                else:
                    _reset_agent_idle(agent)
                    agent.current_stage = 0
                    messages.append(msg_agent_update(agent))
        return messages

    consensus_text = consensus_file.read_text(encoding="utf-8")
    messages.append(msg_log(sys_agent, f"讨论 [{group.project_id}] 完成，共识已生成，启动假设生成", "success", DISCUSSION_STAGE))
    for aid in group.agent_ids:
        agent = state.agents.get(aid)
        if agent:
            agent.stage_progress[DISCUSSION_STAGE] = "completed"
            messages.append(msg_stage_update(agent.id, DISCUSSION_STAGE, "completed"))

    transcript_file = Path(group.discussion_output_dir) / "discussion_transcript.md"
    if transcript_file.exists():
        messages.append(msg_artifact(
            "knowledge", "discussion_transcript.md",
            "沟通讨论", f"{transcript_file.stat().st_size / 1024:.1f} KB",
            group.project_id,
        ))

    group.status = "done"

    # Collect pre-discussion syntheses from all agents for ablation data
    pre_discussion_parts: list[str] = []
    for i, _aid in enumerate(group.agent_ids):
        _ag = state.agents.get(_aid)
        if not _ag:
            continue
        _s7_synth = Path(_ag.run_dir) / "stage-07" / "synthesis.md"
        if _s7_synth.exists():
            _text = _s7_synth.read_text(encoding="utf-8")
            pre_discussion_parts.append(f"## Agent {i+1} ({_aid[:8]})\n\n{_text}")

    for aid in group.agent_ids:
        agent = state.agents.get(aid)
        if not agent:
            continue
        state.discussion_waiting.pop(aid, None)

        has_project = bool(agent.run_dir and (Path(agent.run_dir) / "stage-07").exists())

        if has_project:
            s7_dir = Path(agent.run_dir) / "stage-07"
            s7_dir.mkdir(parents=True, exist_ok=True)
            existing_synthesis = s7_dir / "synthesis.md"
            if existing_synthesis.exists():
                original = existing_synthesis.read_text(encoding="utf-8")
                enriched = (
                    f"{original}\n\n"
                    f"---\n\n"
                    f"# {'Cross-Project' if is_cross else 'Multi-Agent'} Discussion Consensus\n\n"
                    f"{consensus_text}"
                )
                existing_synthesis.write_text(enriched, encoding="utf-8")
            else:
                (s7_dir / "synthesis.md").write_text(consensus_text, encoding="utf-8")

            # Save discussion artifacts for L5 paper ablation study
            disc_artifact_dir = Path(agent.run_dir) / "discussion"
            disc_artifact_dir.mkdir(parents=True, exist_ok=True)
            if pre_discussion_parts:
                (disc_artifact_dir / "pre_discussion_syntheses.md").write_text(
                    "\n\n---\n\n".join(pre_discussion_parts), encoding="utf-8"
                )
            (disc_artifact_dir / "consensus_synthesis.md").write_text(
                consensus_text, encoding="utf-8"
            )
            if transcript_file.exists():
                shutil.copy2(str(transcript_file), str(disc_artifact_dir / "discussion_transcript.md"))

            messages.extend(_launch_s8_for_agent(state, agent, group))
        else:
            _reset_agent_idle(agent)
            agent.current_stage = 0
            agent.stage_progress[DISCUSSION_STAGE] = "completed"
            messages.append(msg_agent_update(agent))
            messages.append(msg_log(agent, "讨论评审完成，恢复空闲", "info", DISCUSSION_STAGE))

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


def _select_best_hypothesis(state: BridgeState, group: DiscussionGroup) -> str:
    """Pick the best agent perspective based on hypothesis quality heuristics.

    Scores each agent's hypotheses.md by: number of hypotheses found,
    total length (richer detail = better), and presence of novelty_report.
    Returns the agent_id with the highest score.
    """
    best_id = group.agent_ids[0]
    best_score = -1
    for aid in group.agent_ids:
        rd = group.run_dirs.get(aid, "")
        if not rd:
            continue
        score = 0
        hypo_file = Path(rd) / "stage-08" / "hypotheses.md"
        if hypo_file.exists():
            text = hypo_file.read_text(encoding="utf-8", errors="replace")
            score += len(text)
            score += text.lower().count("hypothesis") * 500
            score += text.lower().count("## ") * 300
        novelty_file = Path(rd) / "stage-08" / "novelty_report.json"
        if novelty_file.exists():
            score += 2000
        if score > best_score:
            best_score = score
            best_id = aid
    return best_id


def _on_discussion_s8_done(state: BridgeState, agent: LobsterAgent) -> list[dict]:
    """Handle S8 completion — wait for all agents, then pick the best hypothesis
    and create only ONE downstream task to avoid duplicate experiments."""
    messages: list[dict] = []
    agent._is_discussion_s8 = False  # type: ignore[attr-defined]

    project_id = agent.project_id
    group = state.discussion_groups.get(project_id)
    if not group:
        for g in state.discussion_groups.values():
            if agent.id in g.agent_ids:
                group = g
                break

    if group:
        group.completed_s8.add(agent.id)
        messages.append(msg_log(
            agent,
            f"S8 完成 ({len(group.completed_s8)}/{len(group.agent_ids)})，等待其他 agent...",
            "info",
        ))

    # Not all agents done yet — park this agent, wait for peers
    if group and not group.all_s8_done():
        _reset_agent_idle(agent)
        messages.append(msg_agent_update(agent))
        return messages

    # All S8 done — select the best hypothesis and create ONE downstream task
    sys_agent = LobsterAgent(id="system", name="系统", layer="idea", run_id="", run_dir="", config_path="")

    merged_project_id = group.project_id if group else project_id
    if group:
        best_id = _select_best_hypothesis(state, group)
        group.best_agent_id = best_id
        best_run_dir = group.run_dirs.get(best_id, agent.run_dir)
        best_config = group.config_path
        if not best_config:
            best_agent = state.agents.get(best_id)
            best_config = (best_agent.config_path if best_agent else "") or agent.config_path
        other_ids = [a for a in group.agent_ids if a != best_id]
        messages.append(msg_log(
            sys_agent,
            f"项目 [{merged_project_id}] 所有 agent S8 完成 → 选择最优假设 (agent {best_id})，"
            f"合并为单一实验路径 → 进入 L2（淘汰 {', '.join(other_ids)}）",
            "success",
        ))
    else:
        best_run_dir = agent.run_dir
        best_config = agent.config_path

    output_queue_name = LAYER_OUTPUT_QUEUE.get("idea")
    if output_queue_name and output_queue_name in state.queues and merged_project_id:
        _, target_layer = QUEUE_NAMES[output_queue_name]
        follow_task = Task(
            id=f"task-{_uid()}",
            project_id=merged_project_id,
            run_dir=best_run_dir,
            config_path=best_config,
            topic=getattr(agent, '_topic', '') or (group.topic if group else ''),
            source_layer="idea",
            target_layer=target_layer,
            created_at=_now_ms(),
        )
        state.queues[output_queue_name].push(follow_task)
        messages.append(msg_log(
            sys_agent,
            f"项目 [{merged_project_id}] 最优假设已加入 {output_queue_name} 队列 → 进入 L2 实验设计",
            "success",
        ))

    # Reset this agent (peers were already reset when they finished earlier)
    _reset_agent_idle(agent)
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
            if state._fail_counts.get(task.project_id, 0) >= 3:
                continue

            state._fail_counts.pop(task.project_id, None)  # reset on successful assignment
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
        messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "add_lobster":
        name = data.get("name", f"龙虾-{_uid()}")
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
        messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "list_projects":
        messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "resume_project":
        project_id = data.get("projectId", "")
        if project_id:
            messages.extend(resume_project(state, project_id))
            messages.extend(schedule_idle_agents(state))
            messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "quick_submit":
        topic = data.get("topic", "")
        project_id = data.get("projectId", "")
        mode = data.get("mode", "lab")
        angles = data.get("researchAngles")
        if isinstance(angles, str) and angles.strip():
            angles = [a.strip() for a in re.split(r"[,，、;；]", angles) if a.strip()]
        elif not isinstance(angles, list):
            angles = None
        ref_papers = data.get("referencePapers")
        if isinstance(ref_papers, str) and ref_papers.strip():
            ref_papers = [p.strip() for p in re.split(r"[\n,，;；]", ref_papers) if p.strip()]
        elif not isinstance(ref_papers, list):
            ref_papers = None
        reference_files = data.get("referenceFiles")
        if not isinstance(reference_files, list):
            reference_files = None
        path_overrides = {
            "codebases_dir": data.get("codebasesDir", ""),
            "datasets_dir": data.get("datasetsDir", ""),
            "checkpoints_dir": data.get("checkpointsDir", ""),
        }
        messages.extend(quick_submit_project(state, topic, project_id, mode, angles, ref_papers, reference_files, path_overrides))
        messages.extend(schedule_idle_agents(state))
        messages.append(msg_project_list(list_all_projects(state)))

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

    elif cmd == "chat_input":
        content = data.get("content", "").strip()
        target_layer = data.get("targetLayer", "all")
        intent = await _classify_chat_intent(content, state)
        if intent == "query":
            reply = _build_status_summary(state, target_layer)
            messages.append(msg_feedback_ack(f"qs-{_uid()}", reply, target_layer))
        else:
            data["command"] = "human_feedback"
            messages.extend(await handle_command(state, data))

    elif cmd == "query_status":
        target_layer = data.get("targetLayer", "all")
        reply = _build_status_summary(state, target_layer)
        messages.append(msg_feedback_ack(f"qs-{_uid()}", reply, target_layer))

    elif cmd == "pause_project":
        project_id = data.get("projectId", "")
        if project_id:
            messages.extend(_pause_project(state, project_id))
            messages.extend(schedule_idle_agents(state))
            messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "restart_project":
        project_id = data.get("projectId", "")
        if project_id:
            messages.extend(_restart_project(state, project_id))
            messages.extend(schedule_idle_agents(state))
            messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "delete_project":
        project_id = data.get("projectId", "")
        messages.extend(_delete_project(state, project_id))
        state.lab_batches.pop(project_id, None)
        messages.append(msg_project_list(list_all_projects(state)))

    elif cmd == "human_feedback":
        content = data.get("content", "")
        target_layer = data.get("targetLayer", "all")
        message_id = data.get("messageId", f"fb-{_uid()}")

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

        if injected_projects:
            unique = sorted(set(injected_projects))
            plan_hint = (
                f"已将反馈注入 {len(unique)} 个项目的 prompt 上下文中 "
                f"({', '.join(unique)})。"
                f"当前阶段完成后，下一个阶段的 LLM 将读取并参考你的反馈来调整执行计划。"
            )
        else:
            plan_hint = (
                f"已记录反馈。当前无匹配的运行中项目，反馈将在新任务启动时生效。"
            )
        messages.append(msg_feedback_ack(message_id, plan_hint, target_layer))

    elif cmd == "get_download_url":
        project_id = data.get("projectId", "")
        filename = data.get("filename", "latex_package.zip")
        if project_id:
            messages.append({
                "type": "download_url",
                "payload": {
                    "projectId": project_id,
                    "filename": filename,
                    "url": f"/download/{project_id}/{filename}",
                },
            })

    return messages


def _scan_existing_artifacts(state: BridgeState) -> list[dict]:
    """Scan all project directories for completed stage artifacts to send on connect."""
    messages: list[dict] = []
    projects_dir = state.projects_dir()
    if not projects_dir.exists():
        return messages

    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        project_id = proj_dir.name

        # Collect all run dirs: project root + any run-* sub-dirs (Lab mode)
        run_dirs: list[Path] = []
        angle_dirs = list(proj_dir.glob("run-*"))
        if angle_dirs:
            run_dirs.extend(d for d in angle_dirs if d.is_dir())
        else:
            run_dirs.append(proj_dir)

        seen: set[str] = set()
        for run_dir in run_dirs:
            for s, outputs in STAGE_OUTPUTS.items():
                stage_dir = run_dir / f"stage-{s:02d}"
                if not stage_dir.is_dir():
                    continue
                for expected in outputs:
                    if expected not in DISPLAY_ARTIFACTS:
                        continue
                    artifact_path = stage_dir / expected.rstrip("/")
                    dedup_key = f"{project_id}:{s}:{expected}"
                    if dedup_key in seen or not artifact_path.exists():
                        continue
                    seen.add(dedup_key)
                    size = "dir" if artifact_path.is_dir() else f"{artifact_path.stat().st_size / 1024:.1f} KB"
                    content = _extract_artifact_summary(artifact_path, expected)
                    messages.append(msg_artifact(
                        REPO_FOR_STAGE.get(s, "knowledge"), expected, project_id, size, project_id, content, stage=s,
                    ))
            discussion_path = run_dir / "discussion" / "discussion_transcript.md"
            dedup_key = f"{project_id}:discussion:discussion_transcript.md"
            if dedup_key not in seen and discussion_path.is_file():
                seen.add(dedup_key)
                size = f"{discussion_path.stat().st_size / 1024:.1f} KB"
                content = _extract_artifact_summary(discussion_path, "discussion_transcript.md")
                messages.append(msg_artifact(
                    "knowledge", "discussion_transcript.md", "沟通讨论", size, project_id, content
                ))
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

    # Send existing artifacts from completed stages
    try:
        for msg in _scan_existing_artifacts(state):
            await websocket.send(json.dumps(msg, ensure_ascii=False))
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

            # Detect failure → mark task failed, release GPU, track retry count
            if prev_status == "working" and agent.status == "error":
                _fail_pid = agent.project_id or "unknown"
                state._fail_counts[_fail_pid] = state._fail_counts.get(_fail_pid, 0) + 1
                _n_fails = state._fail_counts[_fail_pid]
                _MAX_RETRIES = 3

                if agent.assigned_task_id:
                    for q in state.queues.values():
                        q.fail(agent.assigned_task_id)

                # S12 sanity check failure → pause project and notify user
                if agent.layer == "coding" and agent.project_id:
                    _s12_err_msgs = _check_s12_sanity_failure(state, agent)
                    if _s12_err_msgs:
                        all_messages.extend(_s12_err_msgs)
                        continue

                if agent.layer == "execution" and agent.project_id:
                    released = state.gpu_allocator.release(agent.project_id)
                    if released:
                        all_messages.append(msg_log(agent, f"GPU {released} 已释放 (错误后)", "warning"))

                if _n_fails >= _MAX_RETRIES:
                    all_messages.append(msg_log(
                        agent,
                        f"项目 [{_fail_pid}] 连续失败 {_n_fails} 次，已停止自动重试。请检查日志后手动恢复。",
                        "error",
                    ))

                # Clean up discussion group if agent failed
                _batch_id = getattr(agent, '_idea_factory_batch_id', None)
                _disc_key = _batch_id or (agent.project_id if agent.project_id in state.discussion_groups else None)
                if _disc_key and _disc_key in state.discussion_groups:
                    _grp = state.discussion_groups[_disc_key]
                    if agent.id in _grp.agent_ids:
                        _grp.agent_ids.remove(agent.id)
                        _grp.run_dirs.pop(agent.id, None)
                        _grp.completed_s7.discard(agent.id)
                    remaining = [state.agents.get(a) for a in _grp.agent_ids if state.agents.get(a)]
                    waiting = [a for a in remaining if a.status == "waiting_discussion"]
                    if waiting and len(_grp.agent_ids) < 2:
                        sole = waiting[0]
                        all_messages.append(msg_log(
                            sole,
                            f"伙伴 agent 失败，跳过讨论 → 直接进入 S8 假设生成",
                            "warning", DISCUSSION_STAGE,
                        ))
                        sole.stage_progress[DISCUSSION_STAGE] = "skipped"
                        all_messages.append(msg_stage_update(sole.id, DISCUSSION_STAGE, "skipped"))
                        all_messages.extend(_launch_s8_for_agent(state, sole, _grp))
                        _grp.status = "done"
                    elif not remaining:
                        del state.discussion_groups[_disc_key]
                _reset_agent_idle(agent)
                all_messages.append(msg_agent_update(agent))

        # Poll active discussions
        for group in list(state.discussion_groups.values()):
            all_messages.extend(_poll_discussion(state, group))

        # Schedule idle agents
        sched_msgs = schedule_idle_agents(state)
        all_messages.extend(sched_msgs)

        # Periodically broadcast project list (every ~10 poll cycles)
        if not hasattr(state, '_project_list_counter'):
            state._project_list_counter = 0  # type: ignore[attr-defined]
        state._project_list_counter += 1  # type: ignore[attr-defined]
        if state._project_list_counter >= 10:  # type: ignore[attr-defined]
            state._project_list_counter = 0  # type: ignore[attr-defined]
            all_messages.append(msg_project_list(list_all_projects(state)))

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

    # Initialize queues (load from disk, clean stale tasks from prior run)
    _completed_projects: set[str] = set()
    for _pd in state.projects_dir().iterdir():
        if _pd.is_dir() and not _pd.name.startswith("_"):
            _cp = _read_json(_pd / "checkpoint.json")
            if _cp and _cp.get("last_completed_stage", 0) >= 22:
                _completed_projects.add(_pd.name)
    for queue_name in list(QUEUE_NAMES.keys()) + ["init_to_idea"]:
        q = TaskQueue(name=queue_name, path=state.queues_dir() / f"{queue_name}.json")
        q.load()
        _stale = 0
        _cleaned = 0
        for t in q.tasks:
            if t.status == "assigned":
                t.status = "failed"
                _stale += 1
        orig_len = len(q.tasks)
        q.tasks = [t for t in q.tasks if not (
            t.project_id in _completed_projects and t.status in ("pending", "assigned", "failed")
        )]
        _cleaned = orig_len - len(q.tasks)
        if _stale or _cleaned:
            q.save()
        if _stale:
            print(f"   [queue] {queue_name}: reset {_stale} stale assigned task(s)")
        if _cleaned:
            print(f"   [queue] {queue_name}: removed {_cleaned} task(s) for completed projects")
        state.queues[queue_name] = q

    # Create default lobster pool (configurable via --pool)
    pool_sizes = {"idea": args.pool_idea, "experiment": args.pool_exp,
                  "coding": args.pool_code, "execution": args.pool_exec,
                  "writing": args.pool_write}
    pool_names = {"idea": "L1", "experiment": "L2", "coding": "L3", "execution": "L4", "writing": "L5"}
    default_pool = []
    for layer, count in pool_sizes.items():
        for i in range(count):
            tag = chr(ord('A') + i) if count > 1 else ""
            default_pool.append((f"{pool_names[layer]}·{tag}".rstrip("·"), layer))
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

    def _make_process_request(st: BridgeState):
        """Create HTTP request handler for file downloads."""
        from http import HTTPStatus
        from websockets.http11 import Response as WSResponse

        def _http_response(status_code: int, body: bytes, content_type: str = "text/plain",
                           extra_headers: dict | None = None) -> WSResponse:
            reason = HTTPStatus(status_code).phrase
            headers = {"Content-Type": content_type, "Content-Length": str(len(body)),
                        "Access-Control-Allow-Origin": "*"}
            if extra_headers:
                headers.update(extra_headers)
            return WSResponse(status_code, reason, websockets.Headers(headers), body)

        async def process_request(connection, request):
            if request.path.startswith("/download/"):
                from urllib.parse import unquote
                parts = unquote(request.path[len("/download/"):]).split("/", 1)
                if len(parts) < 2:
                    return _http_response(404, b"Not found\n")
                project_id, filename = parts[0], parts[1]
                proj_dir = Path(st.runs_base_dir) / "projects" / project_id
                file_path = None
                search_roots: list[Path] = []
                if proj_dir.is_dir():
                    search_roots.append(proj_dir)
                cross_proj_dir = Path(st.runs_base_dir) / "projects" / "_cross_discussions" / project_id
                if cross_proj_dir.is_dir():
                    search_roots.append(cross_proj_dir)
                if not search_roots:
                    return _http_response(404, f"Project {project_id} not found\n".encode())

                for root_dir in search_roots:
                    for stage_dir in sorted(root_dir.glob("run-*/stage-*"), reverse=True):
                        candidate = stage_dir / filename
                        if candidate.is_file():
                            file_path = candidate
                            break
                    if file_path:
                        break
                    for discussion_dir in sorted(root_dir.glob("run-*/discussion"), reverse=True):
                        candidate = discussion_dir / filename
                        if candidate.is_file():
                            file_path = candidate
                            break
                    if file_path:
                        break
                    candidate = root_dir / "discussion" / filename
                    if candidate.is_file():
                        file_path = candidate
                        break
                if not file_path:
                    return _http_response(404, f"File {filename} not found\n".encode())
                try:
                    data = file_path.read_bytes()
                    return _http_response(200, data, "application/octet-stream",
                                          {"Content-Disposition": f'attachment; filename="{file_path.name}"'})
                except Exception as e:
                    return _http_response(500, f"Error: {e}\n".encode())
            return None
        return process_request

    handler = lambda ws: ws_handler(state, ws)
    async with websockets.serve(
        handler, "0.0.0.0", args.port,
        process_request=_make_process_request(state),
        max_size=64 * 1024 * 1024,
    ):
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
    parser.add_argument("--discussion-mode", action="store_true", default=True,
                        help="Enable L1 discussion: agents discuss after S7 before generating hypotheses")
    parser.add_argument("--no-discussion-mode", action="store_false", dest="discussion_mode",
                        help="Disable L1 discussion mode")
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
