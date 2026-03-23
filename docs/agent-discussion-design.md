# Agent 讨论模式 — 设计与实现文档

## 一、基本思想

### 问题背景

在原有的 ResearchClaw 架构中，L1 层（调研与创意）的每个 Agent 独立工作：各自完成 S7（知识综合）和 S8（假设生成），彼此之间没有任何信息交换。这导致每个 Agent 的假设只基于自身的文献调研，缺乏多视角的碰撞和互补。

### 解决方案

在 L1 层的 S7（知识综合）和 S8（假设生成）之间插入一个**多 Agent 讨论阶段（Stage 100）**。核心流程：

```
Agent A ──► S7 综合 ──┐
                      ├──► Stage 100: 多Agent讨论 ──► S8 假设生成（各自）
Agent B ──► S7 综合 ──┘
```

1. **独立调研**：多个 L1 Agent 各自独立运行 S7（知识综合），产出各自的 `synthesis.md`
2. **结构化讨论**：所有 Agent 完成 S7 后，启动 3 轮 LLM 驱动的讨论：
   - **Round 1 — 陈述观点**：分析各 Agent 的独特发现、共同主题、知识空白和矛盾之处
   - **Round 2 — 批判审查**：评估证据强度、识别偏差、发现互补结论、提出最有前景的方向
   - **Round 3 — 建立共识**：整合最强发现、解决矛盾、保留独特洞见、生成统一共识文档
3. **共识注入**：将讨论产生的共识文档注入到每个 Agent 的 `stage-07/synthesis.md` 中
4. **增强假设生成**：每个 Agent 基于「自身综合 + 共识文档」独立运行 S8，产出更高质量的假设

## 二、系统架构

### 数据流

```
schedule_idle_agents()
    │
    ├─ 收集 2 个空闲 L1 agent
    ├─ 创建 DiscussionGroup (batch_id)
    └─ 批量启动 S7-only (_launch_idea_factory_run, s7_only=True)
         │
         ▼
poll_loop() 检测 S7 完成
    │
    ├─ _on_idea_factory_s7_done() → agent 状态变为 waiting_discussion
    ├─ 当所有 agent 就绪 → _trigger_discussion()
    │     └─ 启动 discussion_runner.py 子进程（3 轮 LLM 讨论）
    │
    ▼
_poll_discussion() 检测讨论完成
    │
    ├─ 读取 consensus_synthesis.md
    ├─ 注入到每个 agent 的 stage-07/synthesis.md
    └─ 为每个 agent 启动 S8 (_launch_s8_for_agent)
         │
         ▼
poll_loop() 检测 S8 完成
    │
    └─ _on_idea_factory_done() → 提取假设 → 推入 L2 队列
```

### Agent 状态流转

```
idle → working (S7) → done → waiting_discussion → discussing → working (S8) → done → idle
```

### 虚拟阶段 Stage 100

讨论不是 ResearchClaw pipeline 的原生阶段，因此定义了一个虚拟阶段号 `100`：
- 后端常量：`DISCUSSION_STAGE = 100`
- 前端类型：`RCStage.DISCUSSION = 100`
- 在 L1 的 stage bar 中显示为 S7 和 S8 之间的 "💬 Agent 讨论"

## 三、涉及的代码文件

### 后端

| 文件 | 作用 |
|------|------|
| `backend/services/agent_bridge.py` | 核心编排逻辑，所有讨论流程的控制代码 |
| `backend/services/discussion_runner.py` | 独立的讨论执行器（子进程），负责 LLM 多轮对话 |

### 前端

| 文件 | 作用 |
|------|------|
| `frontend/src/types.ts` | 定义 Stage 100 和新状态类型 |
| `frontend/src/components/LayerPanel.tsx` | L1 面板中讨论阶段的 UI 渲染 |
| `frontend/src/components/LogPanel.tsx` | 全局日志中讨论事件的显示 |
| `frontend/src/App.css` | 讨论阶段的样式（动画、颜色等） |

### 配置

| 文件 | 作用 |
|------|------|
| `start.sh` | 启动参数 `--discussion-mode --discussion-rounds 3` |

---

## 四、代码详解

### 4.1 `backend/services/agent_bridge.py` — 核心改动

#### 4.1.1 常量与数据结构

```python
# 第 66-68 行
LAYER_RANGE_PHASE2: dict[str, tuple[int, int]] = {"idea": (8, 8)}
DISCUSSION_STAGE = 100
```

```python
# 第 248-261 行 — DiscussionGroup 数据类
@dataclass
class DiscussionGroup:
    project_id: str                                      # 批次 ID (idea-batch-XXXX)
    topic: str
    config_path: str
    agent_ids: list[str]                                 # 参与讨论的 agent
    run_dirs: dict[str, str]                             # agent_id → run_dir 映射
    completed_s7: set[str]                               # 已完成 S7 的 agent
    status: str = "gathering"                            # gathering|waiting|discussing|done
    discussion_process: subprocess.Popen | None = None   # 讨论子进程
    discussion_output_dir: str = ""

    def all_ready(self) -> bool:
        return len(self.completed_s7) >= len(self.agent_ids) and len(self.agent_ids) >= 2
```

#### 4.1.2 `_launch_idea_factory_run()` — S7-only 模式

关键参数 `s7_only: bool = False`：
- `s7_only=False`（默认）：运行 S7+S8，原有行为
- `s7_only=True`（讨论模式）：只运行 S7，`layer_range = (7, 7)`

同时在 agent 上设置标记属性：
```python
agent._is_idea_factory = True
agent._is_idea_factory_s7_only = s7_only
```

#### 4.1.3 `schedule_idle_agents()` — 批量启动

在原有循环之后，新增讨论模式的批量启动逻辑：
1. 收集所有空闲 L1 agent
2. 如果 >= 2 个空闲：
   - 创建 `DiscussionGroup`，分配共享的 `batch_id`
   - 为每个 agent 调用 `_launch_idea_factory_run(s7_only=True)`
   - 在每个 agent 上记录 `_idea_factory_batch_id = batch_id`
3. 如果 < 2 个空闲：等待（不启动单个 agent）

#### 4.1.4 `_on_idea_factory_s7_done()` — S7 完成处理

当 S7-only 进程退出后触发：
1. 通过 `agent._idea_factory_batch_id` 找到对应的 `DiscussionGroup`
2. 将 agent 加入 `group.completed_s7`
3. 设置 agent 状态为 `waiting_discussion`，当前阶段为 `DISCUSSION_STAGE`
4. 如果所有 agent 都就绪（`group.all_ready()`）→ 触发 `_trigger_discussion()`

#### 4.1.5 `_trigger_discussion()` — 启动讨论子进程

1. 创建讨论输出目录 `projects/idea-batch-XXXX/discussion/`
2. 将所有 agent 状态设为 `discussing`
3. 收集所有 agent 的 `stage-07/` 目录路径
4. 启动 `discussion_runner.py` 子进程，传入：
   - `--config`：LLM 配置文件
   - `--synthesis-dirs`：各 agent 的 stage-07 目录
   - `--output`：讨论输出目录
   - `--rounds`：讨论轮数
   - `--topic`：研究主题

#### 4.1.6 `_poll_discussion()` — 监控讨论进程

每个 poll 周期检查讨论子进程是否完成：
- **失败**：标记所有 agent 为 error
- **成功**：
  1. 读取 `consensus_synthesis.md`
  2. 将共识追加到每个 agent 的 `stage-07/synthesis.md`
  3. 为每个 agent 调用 `_launch_s8_for_agent()` 启动 S8

#### 4.1.7 `poll_loop()` 完成检测逻辑

```python
if prev_status == "working" and agent.status == "done":
    if getattr(agent, '_is_idea_factory_s7_only', False):
        # S7-only 完成 → 进入讨论流程
        _on_idea_factory_s7_done(state, agent)
    elif getattr(agent, '_is_idea_factory', False):
        # 完整 Idea Factory 完成（S7+S8 或讨论后的 S8）
        _on_idea_factory_done(state, agent)
    elif getattr(agent, '_is_discussion_s8', False):
        # 非 Idea Factory 项目的讨论后 S8
        _on_discussion_s8_done(state, agent)
    else:
        on_agent_done(state, agent)
```

#### 4.1.8 错误处理

当 S7-only agent 失败时：
1. 从 `DiscussionGroup` 中移除该 agent
2. 如果组内 agent < 2，取消讨论，释放等待中的 agent

---

### 4.2 `backend/services/discussion_runner.py` — 讨论执行器

独立的 Python 脚本，作为子进程运行。

**输入**：
- 多个 agent 的 `stage-07/synthesis.md` 文件
- LLM 配置（从 `config.arc.yaml` 加载）
- 讨论轮数

**3 轮讨论 Prompt 设计**：

| 轮次 | System Prompt 要点 | 输出 |
|------|-------------------|------|
| Round 1 — 陈述 | 分析各视角的独特发现、共同主题、知识空白、矛盾 | 综合分析 |
| Round 2 — 批判 | 评估证据强度、识别偏差、发现互补、最有前景方向 | 批判审查 |
| Round 3 — 共识 | 整合最强发现、解决矛盾、保留独特洞见、生成统一文档 | 共识综合 |

**输出**：
- `discussion_transcript.md` — 完整讨论记录
- `consensus_synthesis.md` — 共识文档（注入到各 agent 的 S7 产出中）
- `heartbeat.json` — 进度跟踪

---

### 4.3 `frontend/src/types.ts` — 类型定义

关键改动：
- `RCStage` 新增 `DISCUSSION: 100`
- `STAGE_META` 新增 Stage 100 元数据：`{ id: 100, name: 'Agent 讨论', key: 'DISCUSSION', ... }`
- `LAYER_META[AgentLayer.IDEA].stages` 插入 `100`：`[1, 2, 3, 4, 5, 6, 7, 100, 8]`
- `StageStatus` 新增 `'waiting'` 和 `'discussing'`

### 4.4 `frontend/src/components/LayerPanel.tsx` — Stage Bar 渲染

- Stage 100 在 stage bar 中显示为 "💬 Agent 讨论"
- 根据 agent 状态显示不同样式：
  - `waiting_discussion` → ⏳ 等待中
  - `discussing` → 💬 讨论中（带脉冲动画）
  - 完成 → ✓ 已完成

### 4.5 `frontend/src/components/LogPanel.tsx` — 全局日志

- Stage 100 的日志条目显示为 "💬讨论" 标签

### 4.6 `frontend/src/App.css` — 样式

- `.stage-chip.stage-discussion`：讨论阶段的颜色和动画
- `.agent-stage-badge.discussion-badge`：agent 卡片上的讨论标签

---

## 五、配置参数

在 `start.sh` 中通过命令行参数控制：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--discussion-mode` | 启用 | 开启讨论模式（不加则为单 agent 模式） |
| `--discussion-rounds 3` | 3 | LLM 讨论轮数 |
| `--pool-idea 2` | 2 | L1 agent 数量（讨论需要 >= 2） |

也可以通过前端 WebSocket 动态切换：发送 `set_discussion_mode` 命令。

## 六、输出文件结构

```
backend/runs/projects/idea-batch-XXXX/     ← 讨论组目录
└── discussion/
    ├── discussion.log                      ← 讨论执行日志
    ├── discussion_transcript.md            ← 3 轮讨论完整记录
    ├── consensus_synthesis.md              ← 共识文档
    └── heartbeat.json                      ← 进度跟踪

backend/shared_results/idea_runs/idea-YYYY/ ← 各 Agent 的独立工作目录
├── stage-06/                               ← 预置数据
├── stage-07/
│   └── synthesis.md                        ← Agent 自身综合 + 共识（讨论后追加）
├── stage-08/
│   ├── hypotheses.md                       ← 基于共识增强的假设
│   └── perspectives/                       ← 多视角假设
├── checkpoint.json
├── heartbeat.json
├── agent_L-XXXX.log                        ← S7 日志
└── agent_L-XXXX_s8.log                     ← S8 日志
```
