#!/usr/bin/env python3
"""
Real-time system resource monitor via WebSocket.
Sends CPU, memory, and GPU stats to the frontend every 2 seconds.

Usage:
    python resource_monitor.py [--port 8765] [--interval 2]
"""

import argparse
import asyncio
import json
import re
import subprocess
import time
from collections import Counter

import psutil
import websockets

def get_gpu_names() -> dict[int, str]:
    """Query nvidia-smi -L for stable GPU model names."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        names = {}
        for line in result.stdout.strip().split("\n"):
            match = re.match(r"GPU\s+(\d+):\s+(.+?)\s+\(UUID:", line.strip())
            if not match:
                continue
            names[int(match.group(1))] = match.group(2).strip()
        return names
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return {}


def get_gpu_stats() -> list[dict]:
    """Query nvidia-smi for GPU stats."""
    try:
        gpu_names = get_gpu_names()
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpu_id = int(parts[0])
            gpus.append({
                "id": gpu_id,
                "name": gpu_names.get(gpu_id, parts[1]),
                "utilization": float(parts[2]),
                "memUsed": round(float(parts[3]) / 1024, 2),  # MiB -> GiB
                "memTotal": round(float(parts[4]) / 1024, 2),
                "temperature": int(float(parts[5])),
            })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def get_npu_stats() -> list[dict]:
    """Query npu-smi for Ascend NPU stats."""
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        import re
        npus, lines = [], result.stdout.strip().split("\n")
        data_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("|") and "===" not in line and "---" not in line:
                if "npu-smi" in line or "Version" in line: continue
                if "NPU" in line and "Name" in line: continue
                if "Chip" in line and "Bus-Id" in line: continue
                if "Process" in line: break
                data_lines.append(line)
        for i in range(0, len(data_lines) - 1, 2):
            cells1 = [c.strip() for c in data_lines[i].split("|") if c.strip()]
            cells2 = [c.strip() for c in data_lines[i + 1].split("|") if c.strip()]
            if len(cells1) < 3 or len(cells2) < 3: continue
            id_name = cells1[0].split()
            try: npu_id = int(id_name[0])
            except (ValueError, IndexError): continue
            name = f"Ascend {id_name[1]}" if len(id_name) > 1 else "Ascend NPU"
            stats1 = cells1[2].split()
            temperature = int(stats1[1]) if len(stats1) >= 2 else 0
            aicore_m = re.match(r"(\d+)", cells2[2].strip())
            utilization = float(aicore_m.group(1)) if aicore_m else 0.0
            hbm = re.findall(r"(\d+)\s*/\s*(\d+)", cells2[2])
            mem_used = round(float(hbm[-1][0]) / 1024, 2) if hbm else 0.0
            mem_total = round(float(hbm[-1][1]) / 1024, 2) if hbm else 0.0
            npus.append({"id": npu_id, "name": name, "utilization": utilization,
                         "memUsed": mem_used, "memTotal": mem_total, "temperature": temperature})
        return npus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def get_accelerator_stats() -> list[dict]:
    stats = get_gpu_stats()
    return stats if stats else get_npu_stats()


def summarize_accelerator_names(stats: list[dict]) -> str:
    if not stats:
        return ""
    counts = Counter(stat["name"] for stat in stats if stat.get("name"))
    if not counts:
        return ""
    return " + ".join(
        f"{count}x {name}" if count > 1 else name
        for name, count in counts.items()
    )


def get_resource_stats() -> dict:
    mem = psutil.virtual_memory()
    accelerators = get_accelerator_stats()
    return {
        "type": "resource_stats",
        "payload": {
            "cpuPercent": psutil.cpu_percent(interval=None),
            "memUsed": round(mem.used / (1024 ** 3), 2),
            "memTotal": round(mem.total / (1024 ** 3), 2),
            "gpus": accelerators,
            "acceleratorLabel": summarize_accelerator_names(accelerators),
            "timestamp": int(time.time() * 1000),
        },
    }


connected_clients: set = set()


async def handler(websocket):
    connected_clients.add(websocket)
    remote = websocket.remote_address
    print(f"[+] Client connected: {remote}  (total: {len(connected_clients)})")
    try:
        async for _ in websocket:
            pass
    finally:
        connected_clients.discard(websocket)
        print(f"[-] Client disconnected: {remote}  (total: {len(connected_clients)})")


async def broadcast_loop(interval: float):
    psutil.cpu_percent(interval=None)  # prime the first call
    while True:
        await asyncio.sleep(interval)
        if not connected_clients:
            continue
        stats = get_resource_stats()
        msg = json.dumps(stats)
        dead = set()
        for ws in connected_clients:
            try:
                await ws.send(msg)
            except websockets.ConnectionClosed:
                dead.add(ws)
        connected_clients.difference_update(dead)


async def main(host: str, port: int, interval: float):
    print(f"🦞 Resource Monitor WS server starting on ws://{host}:{port}")
    print(f"   Broadcast interval: {interval}s")
    accelerators = get_accelerator_stats()
    if accelerators:
        print(f"   Detected {len(accelerators)} accelerator(s): {summarize_accelerator_names(accelerators)}")
    else:
        print("   No GPU/NPU detected (nvidia-smi and npu-smi not available)")
    mem = psutil.virtual_memory()
    print(f"   System RAM: {mem.total / (1024**3):.1f} GiB")
    print(f"   CPU cores: {psutil.cpu_count(logical=True)}")
    print()

    async with websockets.serve(handler, host, port):
        await broadcast_loop(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resource Monitor WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, args.interval))
