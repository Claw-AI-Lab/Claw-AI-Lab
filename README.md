<p align="center">
  <img src="image/logo.png" width="700" alt="Claw AI Lab">
</p>

<h2 align="center"><b>Claw AI Lab: Autonomous Multi-Agent Research Team</b></h2>

<p align="center">
  <b><i>One Command. A Complete Team.</i></b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://nodejs.org"><img src="https://img.shields.io/badge/Node.js-18%2B-339933?logo=node.js&logoColor=white" alt="Node.js 18+"></a>
  <a href="https://github.com/wufan-cse/Claw-AI-Lab"><img src="https://img.shields.io/badge/GitHub-Claw--AI--Lab-181717?logo=github" alt="GitHub"></a>
</p>

---

## 🤔 What Is This?

**Claw AI Lab** is a fully autonomous multi-agent research system. Given a research topic, it automatically conducts literature review, designs and runs GPU-accelerated experiments, analyzes results, and writes a complete academic paper — end-to-end, with no human intervention.
Multiple agents collaborate across a 5-layer pyramid (survey → design → coding → execution → writing), coordinated via task queues with real-time web monitoring. It supports autonomous exploration, multi-agent debate with heterogeneous LLMs, and paper reproduction workflows.

**We welcome contributions from the community to make this project better together.**

---

## 🔥 Updates

- __[2026.03.25]__: We released Claw AI Lab v1.0.0.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/wufan-cse/Claw-AI-Lab.git
cd Claw-AI-Lab

# Create python environment
conda create -n clawailab python=3.11
conda activate clawailab

# Backend
cd backend/agent
pip install -e ".[all]"
pip install websockets

# Frontend
cd ../../frontend
npm install

# ML dependencies
# You can add more packages based on your research project
pip install torch torchvision diffusers transformers accelerate safetensors \
            huggingface_hub opencv-python pandas matplotlib scikit-image scipy einops tqdm

# OpenHands Beast Mode (optional, recommended)
pip install openhands
```

### 2. Configure

Fill in following configurations in examples/config_template.yaml:
```
llm:
  provider: "openai-compatible"
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key"

sandbox:
  python_path: "/path/to/python3"
```

### 3. Run

```bash
./start.sh              # Start all services
./start.sh stop         # Stop
./start.sh restart      # Restart
./start.sh status       # Status check
./start.sh fresh        # Clean restart (reset all data)
```

Open **http://localhost:5903/** → You will see the system, then submit your projects.

---

## ⚙️ Configuration Details

Description of each configuration in examples/config_template.yaml.
<details>
<summary>Click to expand</summary>

```yaml
# === Project ===
project:
  name: "my-project"              # Project identifier, used for directory naming and UI display
  mode: "full-auto"               # Pipeline mode: "full-auto" runs all stages without human gates

# === Research ===
research:
  topic: "Your research topic"    # The research topic or paper to reproduce (required)
  domains:                        # Research domains for literature search scope
    - "deep-learning"
  daily_paper_count: 5            # Number of papers to retrieve per search query
  quality_threshold: 3.0          # Minimum relevance score (1-5) for literature screening

# === Runtime ===
runtime:
  timezone: "Asia/Shanghai"       # Timezone for timestamps in logs and reports
  max_parallel_tasks: 1           # Max concurrent tasks per agent (keep 1 for stability)
  approval_timeout_hours: 1       # Timeout for human approval at gate stages
  retry_limit: 2                  # Number of retries on stage failure before giving up

# === Notifications ===
notifications:
  channel: "console"              # Notification channel: "console" | "discord" | "slack"
  target: ""                      # Channel target (e.g. Discord webhook URL, leave empty for console)
  on_stage_start: true            # Notify when a stage begins
  on_stage_fail: true             # Notify when a stage fails
  on_gate_required: true          # Notify when human approval is needed

# === Knowledge Base ===
knowledge_base:
  backend: "markdown"             # Storage format: "markdown" | "obsidian"
  root: "docs/kb"                 # Root directory for knowledge base files

# === OpenClaw Bridge ===
openclaw_bridge:
  use_cron: false                 # Enable scheduled research runs
  use_message: false              # Enable progress notifications via messaging platforms
  use_memory: false               # Enable cross-session knowledge persistence
  use_sessions_spawn: false       # Enable spawning parallel sub-sessions
  use_web_fetch: false            # Enable live web search during literature review
  use_browser: false              # Enable browser-based paper collection

# === LLM ===
llm:
  provider: "openai-compatible"   # LLM provider: "openai-compatible" | "openai" | "deepseek" | "acp"
  base_url: "https://api.example.com/v1"  # API endpoint (OpenAI-compatible format)
  api_key: "sk-your-key"          # API key (can also use api_key_env to read from environment)
  api_key_env: "RESEARCHCLAW_API_KEY"     # Environment variable name for API key (fallback if api_key is empty)
  primary_model: "gpt-5.4"       # Main model for research, analysis, and writing stages
  coding_model: "gpt-5.4"        # Model for code generation (S11). Leave empty to use primary_model
  image_model: "gemini-3-pro-image-preview"  # Model for figure generation in paper writing (L5)
  fallback_models:                # Fallback model chain — used when primary model fails
    - "gpt-4o"
    - "gpt-4.1"
  timeout_sec: 600                # LLM API request timeout in seconds

# === Security ===
security:
  hitl_required_stages: []        # Stage numbers requiring human approval (e.g. [5, 9, 20])
  allow_publish_without_approval: true   # Allow paper export without human review
  redact_sensitive_logs: false    # Redact API keys and sensitive data in logs

# === Experiment ===
experiment:
  mode: "sandbox"                 # Execution mode: "sandbox" (local Python) | "docker" | "simulated"
  time_budget_sec: 2400           # Max time budget per experiment run in seconds
  max_iterations: 3               # Number of iterative refinement cycles in S15 (Edit-Run-Eval loop)
  metric_key: "primary_metric"    # Name of the primary evaluation metric
  metric_direction: "minimize"    # Optimization direction: "minimize" | "maximize"
  datasets_dir: "/path/to/datasets"      # Absolute path to datasets directory
  checkpoints_dir: "/path/to/checkpoints"  # Absolute path to model weights directory
  codebases_dir: "/path/to/codebases"    # Absolute path to reference codebases directory
  shared_results_dir: "/path/to/shared_results"  # Directory for cross-project shared results
  paper_length: "short"           # Paper length: "short" (~4 pages) | "long" (~8 pages)

  # Sandbox execution environment
  sandbox:
    python_path: "/path/to/python3"  # Python interpreter path for running experiments
    gpu_required: true            # Whether experiments require GPU
    gpus_per_project: 1           # Number of GPUs allocated per project
    max_memory_mb: 16384          # Max memory limit for experiment processes (MB)
    allowed_imports:              # Whitelist of allowed Python packages in sandbox
      - "numpy"
      - "torch"
      - "transformers"
      - "diffusers"
      # ... add packages as needed

  sanity_check_max_iterations: 6  # Max fix attempts in S12 code testing. 0 = skip fixes, trigger intervention immediately

  # Legacy code agent (disabled by default, use opencode instead)
  code_agent:
    enabled: false

  # OpenHands Beast Mode — delegates complex code generation to OpenHands agent
  opencode:
    enabled: true                 # Master switch for Beast Mode
    auto: true                    # Auto-trigger based on complexity score (vs. manual)
    complexity_threshold: 0.2     # Complexity score threshold (0.0-1.0). Lower = more likely to use Beast Mode
    model: "claude-opus-4-6"      # LLM model used by OpenHands agent
    timeout_sec: 2400             # Max time for Beast Mode code generation (seconds)
    max_retries: 1                # Number of retries if Beast Mode fails to produce main.py
    workspace_cleanup: false      # Whether to delete temporary workspace after completion

# === Prompts ===
prompts:
  custom_file: ""                 # Path to custom prompts YAML file (empty = use defaults)
```

</details>

<!-- ---

## Key Features


- **Multi-Agent Discussion** | Multiple agents with different LLMs debate and reach consensus, avoiding homogeneous outputs. |
- **Beast Mode Code Generation** | Complex experiments auto-routed to OpenHands for multi-file project generation. |
- **Dynamic GPU Allocation** | Automatically detects free GPUs based on utilization. No manual `CUDA_VISIBLE_DEVICES`. |
- **Checkpoint & Resume** | Auto-saves progress after each stage. Resume from any checkpoint after restart. |
- **Manual Intervention** | Auto-pauses on code test failures. Yellow ⚠ indicator on UI with detailed error info. |
- **Knowledge Loop** | Experiment results and insights feed back into the knowledge base for future projects. |
- **Real-time Monitoring** | Web UI with agent status, GPU metrics, task queues, and event logs. |
- **Paper with Figures** | Auto-generates experiment charts, renders concept figures, and injects them into the paper. | -->

---

## 🙏 Acknowledgement

We learned and reused code from the following projects:

[AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw), [AutoResearch](https://github.com/karpathy/autoresearch)

We thank the authors for their contributions to the community!

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

<!-- ## 📌 Citation

If you find Claw AI Lab useful, please cite:

```bibtex
@misc{wu2026clawailab,
  author       = {Wu, Fan and Chen, Cheng and Tan, Zhenshan and Zhang, Taiyu and Gao, Dingcheng and Zhu, Lanyu and Ye, Deheng and Liu, Fayao and Lin, Guosheng and Chen, Tianrun},
  title        = {Claw AI Lab: Autonomous Multi-Agent Research Team},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/wufan-cse/Claw-AI-Lab},
}
``` -->