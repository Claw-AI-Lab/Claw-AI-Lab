# 🦞 Claw AI Lab

金字塔架构的 AI 研究龙虾军团 —— 基于 [AutoResearchClaw](backend/agent/) 的多 Agent 并行研究系统。

## 架构概览

```
                ┌───────────────────────┐
                │  L1 · 调研与创意 (×2)   │  S1→S8
                │  文献调研 → 假设生成     │
                └──────────┬────────────┘
                     💡 Idea 仓库
                ┌──────────┴────────────┐
                │  L2 · 实验设计 (×2)    │  S9
                │  实验方案设计           │
                └──────────┬────────────┘
                     🧪 实验设计仓库
             ┌─────────────┴───────────────┐
             │  L3 · 代码与资源 (×3)         │  S10→S13
             │  代码库检索 → 代码生成 →       │
             │  代码检验 → 资源规划           │
             └─────────────┬───────────────┘
                     💻 代码仓库
        ┌──────────────────┴────────────────────┐
        │  L4 · 执行与修正 (×4)                   │  S14→S18
        │  实验执行 → 迭代优化 → 结果分析 →         │
        │  研究决策 → 知识归纳                     │
        └──────────────────┬────────────────────┘
                     📊 结果仓库
                     🧠 知识库 ──→ 反馈 L1
```

每层部署多只龙虾 Agent，通过层间任务队列（FIFO）自动调度。上层完成后产物进入队列，下层空闲龙虾自动领取任务。不同项目的文件在共享仓库中按项目隔离。

## 26-Stage Pipeline

| Stage | 名称 | 层级 | 模型 | 说明 |
|-------|------|------|------|------|
| S1 | TOPIC_INIT | L1 | gpt-4o | SMART 目标 + 硬件检测 |
| S2 | PROBLEM_DECOMPOSE | L1 | gpt-4o | 子问题树 |
| S3 | SEARCH_STRATEGY | L1 | gpt-4o | 检索策略规划 |
| S4 | LITERATURE_COLLECT | L1 | gpt-4o | OpenAlex / arXiv / Semantic Scholar |
| S5 | LITERATURE_SCREEN ⛩ | L1 | gpt-4o | 文献筛选 (GATE) |
| S6 | KNOWLEDGE_EXTRACT | L1 | gpt-4o | 结构化知识卡片 |
| S7 | SYNTHESIS | L1 | gpt-4o | 主题聚类 + 研究空白识别 |
| S8 | HYPOTHESIS_GEN | L1 | gpt-4o | 假设生成 (参考知识库) |
| S9 | EXPERIMENT_DESIGN ⛩ | L2 | gpt-4o | 实验方案 YAML (GATE) |
| S10 | CODEBASE_SEARCH | L3 | gpt-4o | 搜索 GitHub 可复用代码库 |
| S11 | CODE_GENERATION | L3 | claude-opus-4-6 | 实验代码生成 (Coding 专用模型) |
| S12 | SANITY_CHECK | L3 | gpt-4o | 代码冒烟测试 + LLM 自动修复 |
| S13 | RESOURCE_PLANNING | L3 | gpt-4o | GPU / 时间调度 |
| S14 | EXPERIMENT_RUN | L4 | — | Sandbox 执行 (分配的 GPU) |
| S15 | ITERATIVE_REFINE | L4 | gpt-4o | Edit-Run-Eval 循环 |
| S16 | RESULT_ANALYSIS | L4 | gpt-4o | 指标分析 + 图表 + 注册共享结果 |
| S17 | RESEARCH_DECISION | L4 | gpt-4o | PROCEED / PIVOT / REFINE |
| S18 | KNOWLEDGE_SUMMARY | L4 | gpt-4o | 结论归纳写入知识库 |
| S19-S26 | Paper Writing + Finalization | — | — | 论文写作 (暂未启用) |

## 系统要求

- **Python** >= 3.10 (推荐 Miniforge / Conda)
- **Node.js** >= 18 (通过 fnm 安装)
- **GPU**: NVIDIA GPU (通过 nvidia-smi 采集状态, 支持多卡分配)
- **OS**: Linux

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/wufan-cse/Claw-AI-Lab.git
cd Claw-AI-Lab
git checkout preview-v1.0.0
```

### 2. 安装 Python 依赖

```bash
cd backend/agent
pip install -e .
pip install websockets

# ML 依赖 (用于实验执行)
pip install diffusers transformers accelerate safetensors huggingface_hub \
            opencv-python pandas matplotlib scikit-image

# 验证
python -m researchclaw doctor
```

### 3. 安装 Node.js 和前端依赖

```bash
# 通过 fnm 安装 Node.js
curl -fsSL https://fnm.vercel.app/install | bash
source ~/.bashrc
fnm install --lts

# 安装前端
cd frontend
npm install
```

### 4. 配置 LLM API

编辑 `backend/agent/config_gpu_project.yaml`:

```yaml
llm:
  provider: "openai-compatible"
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key"
  primary_model: "gpt-4o"
  coding_model: "claude-opus-4-6"   # S11 代码生成专用 (留空则用主模型)
  fallback_models:
    - "gpt-4.1"
```

## 使用

### 一键启动

```bash
./start.sh              # 启动 (资源监控 + Agent Bridge + 前端)
./start.sh stop         # 停止
./start.sh restart      # 重启
./start.sh status       # 状态
```

访问 **http://localhost:5173/** 打开控制面板。

### 龙虾池配置

编辑 `start.sh` 中的参数:

```bash
--pool-idea 2           # L1 调研层
--pool-exp 2            # L2 实验层
--pool-code 3           # L3 代码层
--pool-exec 4           # L4 执行层
--total-gpus 8          # GPU 总数
--gpus-per-project 2    # 每项目分配 GPU 数
```

### 提交研究项目

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

或使用批量提交脚本: `python submit_test.py`

## 核心特性

### 金字塔任务调度
- 5 条 FIFO 任务队列连接相邻层级
- 空闲龙虾自动从队列领取任务 (每 2 秒扫描)
- 项目完成后自动创建下一层 follow-up 任务

### GPU 多卡分配
- `GpuAllocator` 管理 GPU 占用表
- L4 执行层启动时注入 `CUDA_VISIBLE_DEVICES`
- GPU 不足时龙虾等待,不领取任务
- 完成/失败后自动释放 GPU

### 代码质量保障
- S10: 搜索 GitHub 可复用代码库
- S11: 可配置 Coding 专用模型 (如 claude-opus-4-6)
- S12: 冒烟测试 (import 检查 + dry run + LLM 自动修复)
- 本地 datasets / checkpoints / codebases 路径注入

### 知识闭环
- S16: Baseline metrics 注册到共享结果库 (LLM 语义匹配复用)
- S18: 实验结论、洞察、后续方向写入知识库
- S8: 新项目假设生成时参考已有知识库

### 实时监控 UI
- 金字塔四层面板: 龙虾状态、stage 进度
- 资源监控: CPU / 内存 / 多 GPU 利用率 + 温度
- 层间数据仓库: 按项目分组, 可折叠文件夹, 知识库内容可展开
- 任务队列: 5 条队列实时计数
- 事件日志: 按层过滤

## 项目结构

```
Claw-AI-Lab/
├── start.sh                        # 一键启动脚本
├── submit_test.py                  # 批量提交脚本
├── frontend/                       # React + Vite + TypeScript
│   ├── src/
│   │   ├── App.tsx                 # 主应用 (双 WebSocket + useReducer)
│   │   ├── types.ts                # 类型定义 (26 Stage + 5 Repo + Queue)
│   │   ├── mock.ts                 # 模拟数据 (bridge 离线时自动启用)
│   │   └── components/
│   │       ├── LayerPanel.tsx      # 金字塔层面板
│   │       ├── DataShelf.tsx       # 层间数据仓库 (可展开文件内容)
│   │       ├── QueuePanel.tsx      # 任务队列
│   │       ├── LogPanel.tsx        # 事件日志
│   │       └── ResourceMonitor.tsx # CPU/GPU 资源监控
│   └── vite.config.ts              # WS 代理配置
├── backend/
│   ├── services/
│   │   ├── resource_monitor.py     # 系统资源 WS (port 8765)
│   │   ├── agent_bridge.py         # Agent 管理 + 队列 + GPU 分配 (port 8766)
│   │   └── result_registry.py      # 跨项目结果共享
│   ├── agent/                      # AutoResearchClaw (魔改版)
│   │   ├── researchclaw/
│   │   │   ├── pipeline/
│   │   │   │   ├── stages.py       # 26-stage 定义
│   │   │   │   ├── contracts.py    # Stage I/O 契约
│   │   │   │   ├── executor.py     # Stage 执行器 (含新增 S10/S12/S18)
│   │   │   │   └── runner.py       # Pipeline 运行器 (含 --to-stage 补丁)
│   │   │   ├── llm/client.py       # LLM 客户端 (DeepSeek 兼容修复)
│   │   │   └── config.py           # 配置 (含 coding_model)
│   │   ├── config_gpu_project.yaml # GPU 项目配置
│   │   └── config_test_minimal.yaml
│   ├── datasets/                   # 本地数据集 (用户放入)
│   ├── checkpoints/                # 模型权重 (自动下载)
│   ├── codebases/                  # 参考代码库 (如 FreeCustom)
│   ├── shared_results/             # 跨项目共享
│   │   ├── index.json              # Baseline metrics 索引
│   │   └── knowledge_base/         # 知识库 (结论+洞察+方向)
│   └── runs/                       # 运行时 (gitignore)
│       ├── projects/               # 各项目独立 run_dir
│       └── queues/                 # 任务队列 JSON
└── logs/                           # 服务日志
```

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 UI | 5173 | React 控制面板, Vite 代理 WS |
| 资源监控 | 8765 | CPU / GPU / 内存 (nvidia-smi + psutil) |
| Agent Bridge | 8766 | 龙虾管理, 任务队列, GPU 分配, 文件监听 |

## License

MIT
