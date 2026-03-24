<p align="center">
  <img src="image/logo.png" width="700" alt="Claw AI Lab">
</p>

<h2 align="center"><b>Claw AI Lab: Autonomous Multi-Agent Research Team</b></h2>

<p align="center">
  <b><i>One Command. A Complete AI Team.</i></b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://nodejs.org"><img src="https://img.shields.io/badge/Node.js-18%2B-339933?logo=node.js&logoColor=white" alt="Node.js 18+"></a>
  <a href="https://github.com/wufan-cse/Claw-AI-Lab"><img src="https://img.shields.io/badge/GitHub-Claw--AI--Lab-181717?logo=github" alt="GitHub"></a>
</p>

---

## What Is This?

**Claw AI Lab** 是基于 [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) 的多 Agent 并行研究系统。

输入一个研究方向 — 系统自动完成文献调研、假设生成、实验设计、代码编写、实验执行、结果分析、论文写作。多个龙虾 Agent 并行工作，通过金字塔分层调度协作完成端到端的研究流程。

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/wufan-cse/Claw-AI-Lab.git
cd Claw-AI-Lab

# Backend
cd backend/agent
pip install -e ".[all]"
pip install websockets

# ML dependencies
pip install torch torchvision diffusers transformers accelerate safetensors \
            huggingface_hub opencv-python pandas matplotlib scikit-image scipy einops tqdm

# OpenHands Beast Mode (optional, recommended)
pip install openhands

# Frontend
cd ../../frontend
npm install
```

### 2. Configure

Fill in your LLM API key:

```yaml
# examples/config_template.yaml

llm:
  provider: "openai-compatible"
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key"
```

### 3. Run

```bash
./start.sh              # Start all services
./start.sh stop         # Stop
./start.sh restart      # Restart
./start.sh status       # Status check
./start.sh fresh        # Clean restart (reset all data)
```

Open **http://localhost:5903/** → You will see the system.

---

## Project Modes

| Mode | Description |
|------|-------------|
| **Lab Explore** | Autonomous research from scratch. Full pipeline S1→S22. |
| **Lab Discuss** | Multi-agent cross-review discussion before experiments. |
| **Reproduce** | Reproduce experiments from an existing paper. |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Discussion** | Multiple agents with different LLMs debate and reach consensus, avoiding homogeneous outputs. |
| **Beast Mode Code Generation** | Complex experiments auto-routed to OpenHands for multi-file project generation. |
| **Dynamic GPU Allocation** | Automatically detects free GPUs based on utilization. No manual `CUDA_VISIBLE_DEVICES`. |
| **Checkpoint & Resume** | Auto-saves progress after each stage. Resume from any checkpoint after restart. |
| **Manual Intervention** | Auto-pauses on code test failures. Yellow ⚠ indicator on UI with detailed error info. |
| **Knowledge Loop** | Experiment results and insights feed back into the knowledge base for future projects. |
| **Real-time Monitoring** | Web UI with agent status, GPU metrics, task queues, and event logs. |
| **Paper with Figures** | Auto-generates experiment charts, renders concept figures, and injects them into the paper. |

---

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCHCLAW_API_KEY` | — | LLM API Key |
| `FRONTEND_PORT` | `5903` | Web UI port |
| `AGENT_BRIDGE_PORT` | `8906` | Agent Bridge WebSocket port |
| `RESOURCE_MONITOR_PORT` | `8905` | Resource monitor WebSocket port |

---

## Requirements

| Dependency | Version | Note |
|-----------|---------|------|
| Linux | Ubuntu 20.04+ | — |
| Python | >= 3.11 | [Miniforge](https://github.com/conda-forge/miniforge) recommended |
| Node.js | >= 18 | [fnm](https://github.com/Schniz/fnm) recommended |
| GPU | NVIDIA (CUDA 11.8+) | Multi-GPU supported |
| OpenHands | >= 1.13 | Optional, for Beast Mode code generation |

---

## License

MIT — see [LICENSE](LICENSE) for details.

<p align="center">
  <sub>Built with 🦞 by the Claw AI Lab team</sub>
</p>
