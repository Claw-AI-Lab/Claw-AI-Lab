#!/usr/bin/env python3
"""Submit 4 GPU research projects to test system stability."""

import asyncio
import json
import sys

import websockets

PROJECTS = [
    {
        "projectId": "gpu-lora-finetune",
        "topic": "Parameter-efficient fine-tuning of LLaMA-7B using LoRA on GPU: comparing rank configurations and learning rate schedules on text classification benchmarks",
    },
    {
        "projectId": "gpu-attention-opt",
        "topic": "Optimizing multi-head attention memory usage with FlashAttention and KV-cache compression on GPU for long-context transformer inference",
    },
    {
        "projectId": "gpu-contrastive-learn",
        "topic": "Self-supervised contrastive learning with SimCLR on CIFAR-100 using GPU: impact of projection head architecture and temperature parameter on downstream accuracy",
    },
    {
        "projectId": "gpu-pruning-speedup",
        "topic": "Structured pruning of ResNet-50 on GPU: comparing magnitude pruning vs learned pruning for inference speedup with minimal accuracy loss on ImageNet subset",
    },
]

CONFIG = "/home/user/PyramidResearchTeam/backend/agent/config_gpu_project.yaml"
WS_URL = "ws://localhost:8766"


async def main():
    async with websockets.connect(WS_URL) as ws:
        # Drain initial agent list
        for _ in range(20):
            try:
                await asyncio.wait_for(ws.recv(), timeout=1)
            except asyncio.TimeoutError:
                break

        # Submit all 4 projects
        for proj in PROJECTS:
            await ws.send(json.dumps({
                "command": "submit_project",
                "projectId": proj["projectId"],
                "configPath": CONFIG,
                "topic": proj["topic"],
            }))
            print(f"✅ 已提交: {proj['projectId']}")
            print(f"   Topic: {proj['topic'][:60]}...")

            # Drain responses
            for _ in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1)
                    d = json.loads(msg)
                    if d["type"] == "log":
                        print(f"   → {d['payload']['message']}")
                except asyncio.TimeoutError:
                    break
            print()

        print("=" * 50)
        print(f"📋 已提交 {len(PROJECTS)} 个 GPU 项目")
        print("   打开 http://localhost:5173/ 查看进度")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
