# Changelog 2026-03-22

**Branch**: `feature/ascend-npu-and-quality-fixes` | **21 files, +778/-42 lines**

## Ascend NPU Adaptation (10 items)
- `hardware.py` — Add `_detect_ascend()` to detect Ascend NPU via npu-smi
- `resource_monitor.py` — Add `get_npu_stats()` to parse 8-card NPU status for frontend
- `docker_sandbox.py` — Add `check_ascend_runtime()` + `--device /dev/davinciN` passthrough
- `ssh_sandbox.py` — Support `ASCEND_VISIBLE_DEVICES` and NPU Docker passthrough
- `config.py` — Add `accelerator_type` field to `DockerSandboxConfig` / `SshRemoteConfig`
- `factory.py` — GPU log distinguishes NPU / CUDA / auto-detect
- `health.py` — Add `check_ascend_runtime()` health check
- `acquirer.py` — `get_baselines(device=None)` auto-detect device
- `Dockerfile.ascend` — Ascend experiment sandbox based on CANN image
- `executor.py` — Code generation prompt injects NPU guidance (torch_npu, device='npu', pin_memory=False, num_workers=0)

## Experiment Quality Zero-Tolerance (5 items)
- `executor.py` S15 — 3 metric rejection gates: ablation invalid / suspicious speed / dummy metric; rejected metrics trigger targeted LLM repair
- `executor.py` S14 — Mark `metrics_trustworthy=false` on integrity failure
- `executor.py` S16 — Mark `experiment_valid=false` when all ablation pairs fail
- `runner.py` — Quality gate checks `experiment_valid` and rejected records in refinement_log
- `executor.py` S17 — Inject "ALL METRICS REJECTED" to force REFINE/PIVOT, block PROCEED

## S12 Sanity Check Fix (3 items)
- `executor.py` — Remove `code[:4000]` truncation, send full source files to LLM
- `executor.py` — Prompt requires returning all code, line count must be >=80% of original
- `executor.py` — Reject patches that shrink file by >40% (truncation safeguard)

## S15 Iterative Refinement Fix (3 items)
- `prompts.py` — `max_tokens` 8192 -> 327680, prevent code truncation
- `prompts.py` — Prompt enforces "NO explanations, start DIRECTLY with code block"
- `executor.py` — `_extract_code_block()` handles unclosed code blocks and explanation prefixes

## Literature Search Optimization (3 items)
- `connectivity.py` — Concurrent network pre-check for all external services (completes in 3s)
- `search.py` — DuckDuckGo session-level reachability cache, skip if unreachable
- `scholar.py` — Google Scholar same as above

## Frontend / Infrastructure (3 items)
- `agent_bridge.py` — Log/artifact history ring buffer (200/100 entries), replay to new clients on connect
- `App.css` — Resource bar max-width 920->1400px + NPU memory text enlarged
- `start.sh` — Python path + CANN env vars + API key adapted for current server
