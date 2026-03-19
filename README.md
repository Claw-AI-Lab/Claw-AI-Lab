# 🦞 Pyramid Research Team

金字塔架构的 AI 研究龙虾军团 —— 基于 [AutoResearchClaw](backend/agent/) 的多 Agent 并行研究系统。

## 架构概览

```
         ┌─────────────────────┐
         │  L1 · 调研与创意     │  S1→S8  文献调研 → 假设生成
         └────────┬────────────┘
            💡 Idea 仓库
         ┌────────┴────────────┐
         │  L2 · 实验设计       │  S9     实验方案设计
         └────────┬────────────┘
            🧪 实验设计仓库
       ┌──────────┴──────────────┐
       │  L3 · 代码与资源         │  S10→S11 代码生成 + 资源规划
       └──────────┬──────────────┘
            💻 代码仓库
     ┌────────────┴────────────────┐
     │  L4 · 执行与修正             │  S12→S15 实验执行 → 结果分析
     └─────────────────────────────┘
            📊 结果仓库
```

每层部署多只龙虾 Agent，通过层间任务队列（FIFO）自动调度：上层完成后产物进入队列，下层空闲龙虾自动领取任务。不同项目的文件在共享仓库中按项目隔离。

## 系统要求

- **Python** >= 3.10（推荐 Miniforge / Conda）
- **Node.js** >= 18（通过 fnm 安装）
- **GPU**：支持 NVIDIA GPU（通过 `nvidia-smi` 采集状态）
- **操作系统**：Linux

## 安装

### 1. 克隆项目

```bash
git clone <repo-url> PyramidResearchTeam
cd PyramidResearchTeam
```

### 2. 安装 Python 依赖

```bash
# 如果没有 conda/miniforge，先安装：
# curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
# bash Miniforge3-Linux-x86_64.sh

# 安装 ResearchClaw agent 包（editable mode）
cd backend/agent
pip install -e .
pip install websockets

# 验证
python -m researchclaw doctor
```

### 3. 安装 Node.js 和前端依赖

```bash
# 如果没有 Node.js，通过 fnm 安装：
curl -fsSL https://fnm.vercel.app/install | bash
source ~/.bashrc
fnm install --lts

# 安装前端依赖
cd frontend
npm install
```

### 4. 配置 LLM API

编辑 `backend/agent/config_gpu_project.yaml`（或创建自己的配置文件）：

```yaml
llm:
  provider: "openai-compatible"
  base_url: "https://your-api-endpoint/v1"   # OpenAI 兼容接口
  api_key: "your-api-key"
  primary_model: "gpt-4o"
```

## 使用

### 一键启动

```bash
./start.sh              # 启动所有服务（资源监控 + Agent Bridge + 前端）
./start.sh stop         # 停止所有服务
./start.sh restart      # 重启
./start.sh status       # 查看状态
```

启动后访问 **http://localhost:5173/** 打开控制面板。

### 龙虾池配置

编辑 `start.sh` 中的参数调整每层龙虾数量：

```bash
--pool-idea 2     # L1 调研层：2 只
--pool-exp 2      # L2 实验层：2 只
--pool-code 3     # L3 代码层：3 只
--pool-exec 4     # L4 执行层：4 只
```

### 提交研究项目

**方式一：使用提交脚本**

```bash
python submit_test.py    # 提交预设的 4 个 GPU 研究项目
```

**方式二：通过 WebSocket 命令**

```python
import asyncio, websockets, json

async def submit():
    async with websockets.connect("ws://localhost:8766") as ws:
        await ws.send(json.dumps({
            "command": "submit_project",
            "projectId": "my-research",
            "configPath": "/path/to/config.yaml",
            "topic": "Your research topic description"
        }))

asyncio.run(submit())
```

**方式三：未来可通过前端 UI 提交**（待实现）

## 项目结构

```
PyramidResearchTeam/
├── start.sh                    # 一键启动脚本
├── submit_test.py              # 测试项目提交脚本
├── frontend/                   # React + Vite 前端
│   ├── src/
│   │   ├── App.tsx             # 主应用（状态管理 + WebSocket）
│   │   ├── types.ts            # 类型定义（Stage/Layer/Queue/Artifact）
│   │   ├── mock.ts             # 模拟数据（bridge 离线时自动启用）
│   │   └── components/
│   │       ├── LayerPanel.tsx   # 金字塔层面板
│   │       ├── DataShelf.tsx    # 层间数据仓库
│   │       ├── QueuePanel.tsx   # 任务队列面板
│   │       ├── LogPanel.tsx     # 事件日志
│   │       └── ResourceMonitor.tsx  # CPU/GPU 资源监控
│   └── vite.config.ts          # Vite 配置（含 WS 代理）
├── backend/
│   ├── services/
│   │   ├── resource_monitor.py # 系统资源监控 WS 服务 (port 8765)
│   │   └── agent_bridge.py     # Agent 管理 + 任务队列 WS 服务 (port 8766)
│   ├── agent/                  # AutoResearchClaw 研究 Agent
│   │   ├── researchclaw/       # 核心包（已 patch: --to-stage 支持）
│   │   ├── config_gpu_project.yaml
│   │   └── config_test_minimal.yaml
│   └── runs/                   # 运行时目录（gitignore）
│       ├── projects/           # 各项目独立 run_dir
│       └── queues/             # 任务队列持久化 JSON
└── logs/                       # 服务日志
```

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 UI | 5173 | React 控制面板，通过 Vite 代理转发 WS |
| 资源监控 | 8765 | 实时 CPU / GPU / 内存数据（nvidia-smi + psutil） |
| Agent Bridge | 8766 | 龙虾管理、任务队列、文件监听 |

## WebSocket 协议

前端通过 `/ws/agents` 和 `/ws/resources` 与后端通信（Vite 代理到 8766 和 8765）。

### 消息类型

| 类型 | 方向 | 说明 |
|------|------|------|
| `agent_update` | Server→Client | Agent 状态变更 |
| `stage_update` | Server→Client | Stage 完成/开始 |
| `artifact_produced` | Server→Client | 新文件产出 |
| `log` | Server→Client | 日志事件 |
| `queue_update` | Server→Client | 任务队列状态 |
| `resource_stats` | Server→Client | CPU/GPU/内存数据 |

### 命令（Client→Server）

| 命令 | 说明 |
|------|------|
| `submit_project` | 提交新研究项目 |
| `add_lobster` | 动态添加龙虾到指定层 |
| `remove_lobster` | 移除龙虾 |
| `stop_agent` | 停止指定 Agent |
| `list_agents` | 列出所有 Agent |
| `get_queues` | 获取队列状态 |

## License

MIT
