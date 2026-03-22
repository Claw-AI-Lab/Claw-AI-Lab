# 🦞 Claw AI Lab

金字塔架构的 AI 研究龙虾军团 —— 基于 [AutoResearchClaw](backend/agent/) 的多 Agent 并行研究系统。

> **v1.0.1 新特性**: 多 Agent 沟通讨论模式 (S8)、多模型支持、代码质量增强 — [查看讨论对比](#-s8-agent-沟通讨论前后对比)

## 架构概览

```
                ┌───────────────────────┐
                │  L1 · 调研与创意 (×2)   │  S1→S7
                │  文献调研 → 知识综合     │
                └──────────┬────────────┘
                     💬 S8 沟通讨论
                  (多 Agent 交叉审议)
                ┌──────────┴────────────┐
                │  S9 假设生成            │
                │  (基于共识综合)          │
                └──────────┬────────────┘
                     💡 Idea 仓库
                ┌──────────┴────────────┐
                │  L2 · 实验设计 (×2)    │  S10
                │  实验方案设计           │
                └──────────┬────────────┘
                     🧪 实验设计仓库
             ┌─────────────┴───────────────┐
             │  L3 · 代码与资源 (×2)         │  S11→S14
             │  代码库检索 → 代码生成 →       │
             │  代码检验 → 资源规划           │
             └─────────────┬───────────────┘
                     💻 代码仓库
        ┌──────────────────┴────────────────────┐
        │  L4 · 执行与修正 (×2)                   │  S15→S19
        │  实验执行 → 迭代优化 → 结果分析 →         │
        │  研究决策 → 知识归纳                     │
        └──────────────────┬────────────────────┘
                     📊 结果仓库
                     🧠 知识库 ──→ 反馈 L1
        ┌──────────────────┴────────────────────┐
        │  L5 · 论文写作 (×2)                     │  S20→S23
        │  大纲 → 初稿 → 审稿 → 修订              │
        └───────────────────────────────────────┘
```

每层部署多只龙虾 Agent，通过层间任务队列（FIFO）自动调度。上层完成后产物进入队列，下层空闲龙虾自动领取任务。不同项目的文件在共享仓库中按项目隔离。

## Pipeline 阶段 (S1 → S23)

| 显示编号 | 名称 | 层级 | 模型 | 说明 |
|---------|------|------|------|------|
| S1 | TOPIC_INIT | L1 | opus-4-6 | SMART 目标 + 硬件检测 |
| S2 | PROBLEM_DECOMPOSE | L1 | opus-4-6 | 子问题树 |
| S3 | SEARCH_STRATEGY | L1 | opus-4-6 | 检索策略规划 |
| S4 | LITERATURE_COLLECT | L1 | opus-4-6 | OpenAlex / arXiv / Semantic Scholar |
| S5 | LITERATURE_SCREEN ⛩ | L1 | opus-4-6 | 文献筛选 (GATE) |
| S6 | KNOWLEDGE_EXTRACT | L1 | opus-4-6 | 结构化知识卡片 |
| S7 | SYNTHESIS | L1 | opus-4-6 | 主题聚类 + 研究空白识别 |
| **S8** | **💬 沟通讨论** | **L1** | **多模型** | **多 Agent 交叉审议 → 共识综合** |
| S9 | HYPOTHESIS_GEN | L1 | opus-4-6 | 假设生成 (基于共识综合 + 知识库) |
| S10 | EXPERIMENT_DESIGN ⛩ | L2 | opus-4-6 | 实验方案 YAML (GATE) |
| S11 | CODEBASE_SEARCH | L3 | opus-4-6 | 搜索 GitHub 可复用代码库 |
| S12 | CODE_GENERATION | L3 | opus-4-6 | 实验代码生成 (Coding 专用模型) |
| S13 | SANITY_CHECK | L3 | opus-4-6 | 代码冒烟测试 + LLM 自动修复 |
| S14 | RESOURCE_PLANNING | L3 | opus-4-6 | GPU / 时间调度 |
| S15 | EXPERIMENT_RUN | L4 | — | Sandbox 执行 (分配的 GPU) |
| S16 | ITERATIVE_REFINE | L4 | opus-4-6 | Edit-Run-Eval 循环 |
| S17 | RESULT_ANALYSIS | L4 | opus-4-6 | 指标分析 + 图表 + 注册共享结果 |
| S18 | RESEARCH_DECISION | L4 | opus-4-6 | PROCEED / PIVOT / REFINE |
| S19 | KNOWLEDGE_SUMMARY | L4 | opus-4-6 | 结论归纳写入知识库 |
| S20 | PAPER_OUTLINE | L5 | opus-4-6 | 论文大纲 |
| S21 | PAPER_DRAFT | L5 | opus-4-6 | 论文初稿 |
| S22 | PAPER_REVIEW | L5 | opus-4-6 | 自动审稿 |
| S23 | PAPER_REVISION | L5 | opus-4-6 | 论文修订终稿 |

## v1.0.1 新特性

### 多 Agent 沟通讨论模式 (S8)

在 L1 调研与创意层，多个 Agent 独立完成 S1-S7（文献调研→知识综合）后，进入 **S8 沟通讨论** 阶段。讨论采用三轮结构化流程：

1. **Present（陈述）**: 每个 Agent 展示自己的独立综合结果
2. **Critique（交叉审议）**: Agent 之间相互评审，指出对方遗漏或矛盾
3. **Consensus（共识）**: 基于讨论内容生成统一的共识综合

讨论中不同 Agent 使用**不同厂商的 LLM**（如 Claude Opus 4.6 + Claude Opus 4.5），以避免同质化输出。

### 代码质量增强

- **Prompt 重构**: 代码生成提示词采用三层优先级体系（MANDATORY / IMPORTANT / NICE TO HAVE）
- **`main.py` 入口检查**: 自动检测缺失的 `def main()` 和 `if __name__ == '__main__'`
- **假指标检测**: 识别 `np.random`、公式化指标、字符串哈希等伪造实验结果
- **安全回退机制**: CodeAgent 审阅后代码质量下降时自动回退到审阅前版本；Executor 修复失败时回退到 CodeAgent 原始输出

### 可配置 LLM 超时

通过 YAML 配置 `timeout_sec` 控制 LLM API 超时时间（默认 600 秒），适配大模型的长响应场景。

---

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
git checkout preview-v1.0.1
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

---

## 💬 S8 Agent 沟通讨论前后对比

以下为一次实际运行中的 Agent 讨论效果对比。研究主题为 **Training-Free Image Generation Using Attention Manipulation**。

### 讨论前：Agent 独立综合 (S7)

两个 Agent 在 S1-S7 阶段独立完成文献调研和知识综合，各自生成了独立的 Synthesis 报告。

#### Agent A 综合要点

- 识别出 **4 个主题聚类**: 组合多主体生成、布局引导空间控制、免训练主体驱动个性化、基于注意力的图像编辑
- 发现 **5 个研究空白**: 可扩展多主体生成（>3 个主体退化）、跨控制模态统一框架、视频时序一致性、现代架构适配（DiT/SD3/Flux）、定量注意力控制理论
- 提出 **4 个研究方向**: 系统基准测试、可组合注意力操纵原语、注意力图属性理论分析、视频原生注意力控制

#### Agent B 综合要点

- 识别出 **5 个主题聚类**: 组合与布局引导生成、注意力注入图像编辑、主体驱动身份保持生成、空间与语义控制机制、注意力动态理论理解
- 发现 **6 个研究空白**: 跨架构泛化、定量基准与评估标准、真实图像编辑鲁棒性、细粒度多粒度控制、视频时序一致性、理论基础
- 提出 **7 个优先机会**: DiT/Transformer 适配、统一基准、混合注意力操纵+轻量适配器、少步模型免反演编辑、视频扩散注意力操纵、形式化理论框架、多模态注意力操纵

#### 讨论前差异

| 维度 | Agent A | Agent B |
|------|---------|---------|
| 聚类数量 | 4 个 | 5 个（多出"空间语义控制"和"理论理解"两个独立聚类） |
| 空白识别 | 5 个，侧重技术可行性 | 6 个，多出"细粒度控制"维度 |
| 优先级排序 | 统一框架 > DiT适配 > 理论 | DiT适配 > 基准 > 混合方法 |
| 理论深度 | 提及但未展开 | 有独立聚类分析注意力动态 |
| 独特贡献 | CFG 与注意力操纵的联系、Token Merging 交叉点 | PCA 引导子空间（FreeControl）、IP-Adapter 作为混合方法的潜力 |

### 讨论后：共识综合 (S8)

经过三轮结构化讨论（陈述 → 交叉审议 → 共识），两个 Agent 达成了显著更丰富的共识综合，具体新增内容包括：

#### 新增洞见

1. **干预分类法 (Intervention Taxonomy)**: 提出 Type 0-3 四级分类（无修改 → 直接替换 → 梯度优化 → 轻量适配器），解决了"免训练"定义的模糊性
2. **注意力容量瓶颈假说**: 将交叉注意力早期稳定化（~20-40%）与多主体退化（>3 个）关联，提出可测试预测——注意力图有效秩随主体数增加而下降
3. **少步模型中的注意力可解释性**: 提出 1-4 步模型中每一步编码更广的时间层级范围，使操纵更强大但也更脆弱
4. **子空间正交性组合假说**: 不同类型的注意力操纵（布局/风格/身份）可能操作在近似正交的子空间上，从而实现干净的组合

#### 解决的矛盾

| 矛盾 | 解决方案 |
|------|---------|
| 架构适配 vs 统一框架的优先级之争 | 确认为**顺序依赖**关系：架构兼容性是统一框架的前提 |
| "注意力操纵"的范围定义 | 核心定义为注意力机制的直接干预；skip connection 等为相关但不同的范畴 |
| 理论理解：空白 vs 已有知识 | 两者都对——有丰富的经验性理解但缺乏形式化理论框架 |

#### 共识研究路线图

| 层级 | 时间 | 方向 |
|------|------|------|
| Tier 1 | 0-6 月 | (1) SD3/Flux 联合注意力结构表征 (2) 少步模型注意力操纵 (3) 系统化基准开发 |
| Tier 2 | 6-18 月 | (4) 可组合操纵规则 (5) 失败模式表征 (6) 层级注意力多主体生成 |
| Tier 3 | 18-36 月 | (7) 注意力语义理论基础 (8) 视频原生注意力控制 (9) 多模态/3D 注意力操纵 |

#### 讨论效果总结

| 指标 | 讨论前 (S7) | 讨论后 (S8 共识) | 提升 |
|------|------------|------------------|------|
| 方法分类维度 | 4-5 聚类 | 统一分类 + 4 级干预类型学 | 结构化程度显著提升 |
| 研究假说 | 0（仅列方向） | 5 个可测试假说 | 从定性到定量 |
| 矛盾解决 | 存在分歧 | 3 个核心矛盾已解决 | 共识质量提升 |
| 路线图 | 各自独立排序 | 统一三层 9 方向路线图 | 可执行性提升 |
| 独特洞见保留 | 分散 | 5 个跨 Agent 洞见整合 | 信息无损合并 |

---

## License

MIT
