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
import subprocess
import time

import psutil
import websockets

def get_gpu_stats() -> list[dict]:
    """Query nvidia-smi for GPU stats."""
    try:
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
            gpus.append({
                "id": int(parts[0]),
                "name": parts[1],
                "utilization": float(parts[2]),
                "memUsed": round(float(parts[3]) / 1024, 2),  # MiB -> GiB
                "memTotal": round(float(parts[4]) / 1024, 2),
                "temperature": int(float(parts[5])),
            })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def get_resource_stats() -> dict:
    mem = psutil.virtual_memory()
    return {
        "type": "resource_stats",
        "payload": {
            "cpuPercent": psutil.cpu_percent(interval=None),
            "memUsed": round(mem.used / (1024 ** 3), 2),
            "memTotal": round(mem.total / (1024 ** 3), 2),
            "gpus": get_gpu_stats(),
            "timestamp": int(time.time() * 1000),
        },
    }


connected_clients: set[websockets.ServerConnection] = set()


async def handler(websocket: websockets.ServerConnection):
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
    gpus = get_gpu_stats()
    if gpus:
        print(f"   Detected {len(gpus)} GPU(s): {gpus[0]['name']}")
    else:
        print("   No GPU detected (nvidia-smi not available)")
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
