#!/usr/bin/env python3
"""Submit a single test project to verify OpenCode generates code based on the FreeCustom codebase."""

import asyncio
import json
import sys

import websockets

PROJECT_ID = "opencode-freecustom-verify"
TOPIC = (
    "Improving multi-reference self-attention (MRSA) in FreeCustom for better "
    "multi-concept image composition: extending the weighted mask strategy with "
    "adaptive attention scaling based on concept similarity in diffusion models"
)
CONFIG = "/home/user/PyramidResearchTeam/config_opencode_freecustom_test.yaml"
WS_URL = "ws://localhost:8766"


async def main():
    print("=" * 60)
    print("🧪 OpenCode + FreeCustom Codebase 测试")
    print("=" * 60)
    print(f"  Project ID: {PROJECT_ID}")
    print(f"  Config:     {CONFIG}")
    print(f"  Topic:      {TOPIC[:80]}...")
    print()

    try:
        async with websockets.connect(WS_URL) as ws:
            for _ in range(20):
                try:
                    await asyncio.wait_for(ws.recv(), timeout=1)
                except asyncio.TimeoutError:
                    break

            await ws.send(json.dumps({
                "command": "submit_project",
                "projectId": PROJECT_ID,
                "configPath": CONFIG,
                "topic": TOPIC,
            }))
            print(f"✅ 已提交项目: {PROJECT_ID}")

            for _ in range(10):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    d = json.loads(msg)
                    if d.get("type") == "log":
                        print(f"   → {d['payload']['message']}")
                    elif d.get("type") == "project_status":
                        print(f"   📋 状态: {d['payload'].get('status', 'unknown')}")
                except asyncio.TimeoutError:
                    break

            print()
            print("=" * 60)
            print("📋 项目已提交！请在前端查看进度: http://localhost:5173/")
            print(f"   运行目录: backend/runs/projects/{PROJECT_ID}/")
            print("   关注 stage-10 / stage-11 的 OpenCode 输出")
            print("=" * 60)

    except ConnectionRefusedError:
        print("❌ 无法连接到 Agent Bridge (ws://localhost:8766)")
        print("   请先运行 ./start.sh 启动服务")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
