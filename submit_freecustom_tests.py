#!/usr/bin/env python3
"""Submit 2 FreeCustom optimization projects for testing.

Data paths point to /home/user/Claw-AI-Lab-share (shared ckpt/codebase/datasets).
Agent Bridge on port 8896 to avoid conflicts with existing services.
"""

import asyncio
import json
import os
import sys

import websockets

BASE = os.path.dirname(os.path.abspath(__file__))

PROJECTS = [
    {
        "projectId": "freecustom-adaptive-mrsa",
        "configPath": os.path.join(BASE, "config_freecustom_proj1.yaml"),
        "topic": (
            "Adaptive multi-reference self-attention (MRSA) for FreeCustom: "
            "dynamically scaling attention weights based on concept similarity "
            "to improve multi-concept composition quality"
        ),
    },
    {
        "projectId": "freecustom-conflict-aware",
        "configPath": os.path.join(BASE, "config_freecustom_proj2.yaml"),
        "topic": (
            "Conflict-aware mask refinement in FreeCustom: detecting and resolving "
            "spatial conflicts between multiple concept references during MRSA "
            "to reduce concept leakage in generated images"
        ),
    },
]

WS_URL = "ws://localhost:8896"


async def main():
    print(f"🔗 Connecting to Agent Bridge at {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        for _ in range(20):
            try:
                await asyncio.wait_for(ws.recv(), timeout=1)
            except asyncio.TimeoutError:
                break

        for proj in PROJECTS:
            await ws.send(json.dumps({
                "command": "submit_project",
                "projectId": proj["projectId"],
                "configPath": proj["configPath"],
                "topic": proj["topic"],
            }))
            print(f"✅ 已提交: {proj['projectId']}")
            print(f"   Topic: {proj['topic'][:80]}...")
            print(f"   Config: {proj['configPath']}")

            for _ in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1)
                    d = json.loads(msg)
                    if d.get("type") == "log":
                        print(f"   → {d['payload']['message']}")
                except asyncio.TimeoutError:
                    break
            print()

        print("=" * 60)
        print(f"📋 已提交 {len(PROJECTS)} 个 FreeCustom 优化项目")
        print(f"   数据路径: /home/user/Claw-AI-Lab-share")
        print(f"   前端 UI:  http://localhost:5893/")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
