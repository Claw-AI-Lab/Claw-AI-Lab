#!/usr/bin/env python3
"""End-to-end verification for Docker sandbox.

Run after building the image:
    docker build -t researchclaw/experiment:latest researchclaw/docker/
    python tests/e2e_docker_sandbox.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from researchclaw.config import DockerSandboxConfig, ExperimentConfig
from researchclaw.experiment.docker_sandbox import DockerSandbox
from researchclaw.experiment.factory import create_sandbox

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    tag = PASS if ok else FAIL
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def main() -> None:
    print("=" * 60)
    print("Docker Sandbox End-to-End Verification")
    print("=" * 60)

    # ── Preflight ──────────────────────────────────────────────
    print("\n--- Preflight ---")
    docker_ok = DockerSandbox.check_docker_available()
    check("Docker daemon reachable", docker_ok)
    if not docker_ok:
        print("\nDocker is not available. Cannot proceed.")
        sys.exit(1)

    image_ok = DockerSandbox.ensure_image("researchclaw/experiment:latest")
    check("Image exists locally", image_ok)
    if not image_ok:
        print("\nImage not found. Build it first:")
        print("  docker build -t researchclaw/experiment:latest researchclaw/docker/")
        sys.exit(1)

    # ── Test 1: Basic execution + metrics ──────────────────────
    print("\n--- Test 1: Basic execution + metrics ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        cfg = DockerSandboxConfig(gpu_enabled=False, network_policy="none")
        sandbox = DockerSandbox(cfg, Path(tmp) / "work")
        code = (
            "import numpy as np\n"
            "x = np.random.randn(100)\n"
            "print(f'primary_metric: {float(np.mean(x**2)):.4f}')\n"
            "print(f'std: {float(np.std(x)):.4f}')\n"
            "print('Done.')\n"
        )
        r = sandbox.run(code, timeout_sec=60)
        check("returncode == 0", r.returncode == 0, f"rc={r.returncode}")
        check("metrics parsed", "primary_metric" in r.metrics, str(r.metrics))
        check("stdout non-empty", bool(r.stdout.strip()), repr(r.stdout[:100]))
        check("timed_out is False", r.timed_out is False)
        check("elapsed_sec > 0", r.elapsed_sec > 0, f"{r.elapsed_sec:.2f}s")

    # ── Test 2: Multi-file project ─────────────────────────────
    print("\n--- Test 2: Multi-file project ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        cfg = DockerSandboxConfig(gpu_enabled=False, network_policy="none")
        sandbox = DockerSandbox(cfg, Path(tmp) / "work")

        project = Path(tmp) / "project"
        project.mkdir()
        (project / "utils.py").write_text(
            "def add(a, b): return a + b\n", encoding="utf-8"
        )
        (project / "main.py").write_text(
            "from utils import add\n"
            "result = add(3, 4)\n"
            "print(f'primary_metric: {result}')\n",
            encoding="utf-8",
        )
        r = sandbox.run_project(project, timeout_sec=60)
        check("project returncode == 0", r.returncode == 0, f"rc={r.returncode}")
        check("project metric correct", r.metrics.get("primary_metric") == 7.0,
              str(r.metrics))

    # ── Test 3: results.json ───────────────────────────────────
    print("\n--- Test 3: results.json from volume ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        cfg = DockerSandboxConfig(gpu_enabled=False, network_policy="none")
        sandbox = DockerSandbox(cfg, Path(tmp) / "work")
        code = (
            "import json\n"
            "results = {'accuracy': 0.92, 'f1': 0.88}\n"
            "with open('results.json', 'w') as f:\n"
            "    json.dump(results, f)\n"
            "print('primary_metric: 0.92')\n"
        )
        r = sandbox.run(code, timeout_sec=60)
        check("results.json metric merged", "f1" in r.metrics,
              str(r.metrics))

    # ── Test 4: Network isolation ──────────────────────────────
    print("\n--- Test 4: Network isolation ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        cfg = DockerSandboxConfig(gpu_enabled=False, network_policy="none")
        sandbox = DockerSandbox(cfg, Path(tmp) / "work")
        code = (
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://example.com', timeout=5)\n"
            "    print('NETWORK_ACCESS: yes')\n"
            "except Exception as e:\n"
            "    print('NETWORK_ACCESS: no')\n"
            "    print(f'primary_metric: 1.0')\n"
        )
        r = sandbox.run(code, timeout_sec=30)
        network_blocked = "NETWORK_ACCESS: no" in r.stdout
        check("Network blocked (--network=none)", network_blocked,
              r.stdout.strip()[:200])

    # ── Test 5: GPU visibility ─────────────────────────────────
    print("\n--- Test 5: GPU visibility ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        cfg = DockerSandboxConfig(gpu_enabled=True, network_policy="none")
        sandbox = DockerSandbox(cfg, Path(tmp) / "work")
        code = (
            "import torch\n"
            "gpu_available = torch.cuda.is_available()\n"
            "if gpu_available:\n"
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
            "    print('primary_metric: 1.0')\n"
            "else:\n"
            "    print('GPU: none')\n"
            "    print('primary_metric: 0.0')\n"
        )
        r = sandbox.run(code, timeout_sec=60)
        gpu_visible = "primary_metric" in r.metrics and r.metrics["primary_metric"] == 1.0
        if gpu_visible:
            check("GPU visible in container", True, r.stdout.strip()[:200])
        else:
            # Not a hard failure — might not have NVIDIA runtime
            print(f"  [{SKIP}] GPU not visible (NVIDIA Container Toolkit may not be installed)")
            print(f"         stdout: {r.stdout.strip()[:200]}")
            print(f"         stderr: {r.stderr.strip()[:200]}")

    # ── Test 6: Memory limit ──────────────────────────────────
    print("\n--- Test 6: Memory limit enforcement ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        # Set a very low memory limit to trigger OOM
        cfg = DockerSandboxConfig(
            gpu_enabled=False, network_policy="none", memory_limit_mb=64
        )
        sandbox = DockerSandbox(cfg, Path(tmp) / "work")
        code = (
            "import numpy as np\n"
            "# Allocate ~200MB to exceed 64MB limit\n"
            "x = np.ones((25_000_000,), dtype=np.float64)\n"
            "print(f'primary_metric: {x.sum()}')\n"
        )
        r = sandbox.run(code, timeout_sec=30)
        oom = r.returncode != 0
        check("OOM kills container (64MB limit, 200MB alloc)", oom,
              f"rc={r.returncode}, stderr={r.stderr.strip()[:200]}")

    # ── Test 7: Factory integration ────────────────────────────
    print("\n--- Test 7: Factory integration ---")
    with tempfile.TemporaryDirectory(prefix="rc_e2e_") as tmp:
        config = ExperimentConfig(mode="docker", docker=DockerSandboxConfig(gpu_enabled=False))
        sandbox = create_sandbox(config, Path(tmp) / "work")
        check("Factory returns DockerSandbox", isinstance(sandbox, DockerSandbox))
        r = sandbox.run("print('primary_metric: 42.0')", timeout_sec=30)
        check("Factory sandbox executes", r.returncode == 0 and r.metrics.get("primary_metric") == 42.0,
              str(r.metrics))

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("\nFailed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
