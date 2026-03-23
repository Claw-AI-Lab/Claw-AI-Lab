# L3 代码生成层分析报告

> 生成日期: 2026-03-21
> 分析范围: 8 个 L3 项目（4 成功 + 4 失败）
> 研究主题: Training-free image generation using attention manipulation

---

## 一、生成代码内容总结

### 1.1 成功项目（通过 S11 验证）

| 项目 ID | 代码文件 | 总行数 | 框架 | 有 main() | 可运行 |
|---------|----------|--------|------|-----------|--------|
| idea-4aca66e0 | 5 files (main, models, metrics, data_loader, attention_manipulators) | 1,236 | NumPy | 否 | 否 |
| idea-ada8e461 | 4 files (main, models, methods, data) | 1,250 | NumPy | 否 | 否 |
| idea-db396b15 | 1 file (main) | **5** | 无 | 是(空) | 否 |
| idea-f182e55d | 5 files (main, models, metrics, diffusion, attention_manipulations) | 933 | PyTorch | 否 | 否 |

**共性问题**：
- **所有 4 个项目的 main.py 都不包含实验执行逻辑**（仅有 Config 类或空函数）
- 组件代码（模型、指标、注意力操控器）质量尚可，但没有"胶水代码"将组件连接运行
- 模型全部使用随机初始化权重，无预训练模型加载
- 3/4 个项目使用合成数据代替真实数据集（CelebA/CIFAR-10）

### 1.2 失败项目（S11 验证未通过）

| 项目 ID | 失败原因 | 代码行数 | 核心问题 |
|---------|---------|---------|---------|
| idea-6771ef03 | API 连接断开 | 719 | 指标完全伪造（ASCII 码哈希）|
| idea-ece84ddf | API 连接断开 | 472 | 代理指标不真实（L2距离到零向量）|
| idea-f2bb53e0 | 代码验证 BLOCKED | 218 | 纯配置文件，无任何可执行代码 |
| idea-f4c982d1 | 代码验证 BLOCKED | 127 | 纯 Config 类，无实验逻辑 |

### 1.3 代码质量典型案例

**案例 1：虚假指标（idea-6771ef03）**
```python
def deterministic_metric(seed, cond_name):
    base = sum(ord(c) for c in cond_name) % 10
    val = 0.7 + 0.05 * (seed % 5) + 0.01 * base
    return round(val, 4)
```
→ 用字符串 ASCII 码和种子号生成假 CLIP 分数，完全不基于图像质量。

**案例 2：参数覆盖 Bug（idea-ada8e461, idea-f182e55d）**
```python
def add_noise(self, x0, t, noise=None, rng=None):
    noise = None  # ← 传入的 noise 参数被立即覆盖！
```
→ 相同模式在 2 个项目中独立出现，是 LLM 代码生成的系统性缺陷。

**案例 3：空壳代码（idea-db396b15）**
```python
def main():
    pass

if __name__ == "__main__":
    main()
```
→ 5 行代码通过了原有验证系统。

---

## 二、问题根因分析

### 2.1 代码生成 Prompt 链路

```
exp_plan.yaml (S9)
    ↓
executor._execute_code_generation()
    ↓ 构建上下文: pkg_hint + compute_budget + extra_guidance
    ↓
CodeAgent.generate()
    ├─ Phase 1: Blueprint Planning (architecture_planning prompt)
    │   → YAML 格式蓝图，含文件列表、伪代码、张量形状
    │   ⚠ 蓝图经常被 LLM 包裹在 ```yaml 中导致解析失败
    │
    ├─ Phase 2: Code Generation
    │   ├─ 正常路径: 按蓝图逐文件生成 (generate_single_file prompt)
    │   └─ 回退路径: 蓝图解析失败 → 单次生成 (code_generation prompt)
    │   ⚠ code_generation prompt 有 347 行、20+ 个 CRITICAL 标记
    │
    ├─ Phase 2.5: Hard Validation (AST 静态检查)
    │   ⚠ 缺失 main() 入口点检测
    │   ⚠ 虚假指标检测正则太少（仅 3+2 条规则）
    │
    └─ Phase 4: Review Dialog (code_reviewer prompt)
        ⚠ reviewer 与 coder 使用同一 LLM
        ⚠ 修改后的代码可能引入新的语法错误
```

### 2.2 五大根因

| # | 根因 | 影响 | 证据 |
|---|------|------|------|
| R1 | **main() 入口检测缺失** | 空壳/配置文件通过验证 | 4/4 成功项目的 main.py 都缺少实验执行逻辑 |
| R2 | **Prompt 信息过载** | LLM 注意力分散，忽略核心要求 | code_generation prompt 347 行、20+ CRITICAL 章节涵盖 RL/KD/PPO 等不相关领域 |
| R3 | **虚假指标检测薄弱** | 随机数/哈希/公式化 metric 不被拦截 | 仅匹配 3 个字面字符串 + 2 个简单正则 |
| R4 | **蓝图解析频繁失败** | 回退到低质量 single-shot 路径 | 4/8 项目出现 "Blueprint YAML parse error" |
| R5 | **Review 循环破坏有效代码** | 通过验证的代码被改坏后无法恢复 | Phase 4 review score=1→3 但引入语法错误 |

### 2.3 为什么所有代码都缺少 main() 函数

1. `generate_single_file` prompt 说 "Output ONLY the Python code"，但没有明确要求 `if __name__ == '__main__': main()` 调用入口
2. Phase 2.5 的 `_hard_validate()` **完全没有检查** main.py 中是否存在 `main()` 函数
3. 蓝图的 pseudocode 虽然包含 `main()` 函数，但蓝图经常解析失败，回退到 single-shot 后信息丢失
4. Prompt 信息过载导致 LLM 将注意力放在 RL detach、维度检查等领域特定细节上，忽略了最基本的可执行性

---

## 三、修改意见

### 3.1 优先级排序

| 优先级 | 修复项 | 预期效果 | 修改文件 |
|--------|-------|---------|---------|
| **P0** | 添加 main() 入口点硬检测 | 拦截空壳/纯配置 main.py | `validator.py`, `executor.py` |
| **P0** | 精简 code_generation prompt | 提升核心代码质量 | `prompts.py` |
| **P0** | 增强虚假指标检测 | 拦截随机数/哈希/公式化 metric | `validator.py` |
| **P1** | Phase 4 review 安全回退 | 防止 review 破坏有效代码 | `code_agent.py` |
| **P1** | Executor 修复循环 fallback | 修复失败时回退到原始代码 | `executor.py` |

### 3.2 具体方案

**方案 A: main() 入口检测** — 新增 `check_main_entry_point()` 函数：
- 检测 `def main()` 是否存在
- 检测 `if __name__ == '__main__'` 入口守卫
- 检测 `main()` 是否 trivially empty（`def main(): pass`）

**方案 B: Prompt 重构** — 将 347 行 20+ CRITICAL 重构为三层优先级：
- TIER 1（5 条）: 必须满足 — main() 入口、真实算法、真实指标、多种子、无 CLI 参数
- TIER 2（5 条）: 应当满足 — 异常捕获、指标定义、条件注册、广度优先、统计汇总
- TIER 3（3 条）: 锦上添花 — Bootstrap CI、多难度测试、消融自检
- 加入正面代码范例（Minimum Viable main.py）

**方案 C: 虚假指标检测增强** — 新增 4 条检测模式：
- `np.random` / `random.` 赋值给 metric 变量
- `print()` 中包含随机值输出
- 循环变量线性公式生成 metric
- `sum(ord(c) for c in ...)` 哈希模式

---

## 四、已完成的代码修复

### 4.1 `validator.py` — 新增 main() 入口检测 + 虚假指标检测

**文件**: `backend/agent/researchclaw/experiment/validator.py`

新增函数 `check_main_entry_point(code: str) -> list[str]`:
- 使用 AST 解析检测 `def main()` 是否存在
- 检测 `if __name__` 入口守卫
- 递归计算 main() 内的有效语句数（Assignment, For, While, If, Call, Return, Try），< 3 则判定为 trivially empty

新增虚假指标检测模式（在 `check_code_complexity` 中）:
```python
fake_metric_patterns = [
    (r"(?:metric|score|accuracy|loss|fid|clip_score)\s*=\s*(?:np\.random|random\.)\w+",
     "random value assigned to metric variable"),
    (r"print\(.*(?:np\.random|random\.)\w+.*\)", "random value in print output"),
    (r"(?:metric|score|accuracy|loss)\s*=\s*[\d.]+\s*[+\-*/]\s*(?:seed|idx|i)\s*\*\s*[\d.]+",
     "formulaic metric from loop variable"),
    (r"sum\(ord\(c\)\s+for\s+c\s+in", "deterministic fake metric from string hash"),
]
```

### 4.2 `executor.py` — 集成入口检测 + 修复循环 fallback

**文件**: `backend/agent/researchclaw/pipeline/executor.py`

修改 1: 在 code quality check 阶段调用 `check_main_entry_point`:
```python
if "main.py" in files:
    _entry_warnings = check_main_entry_point(files["main.py"])
    for w in _entry_warnings:
        logger.warning("Stage 10 code quality: %s", w)
```

修改 2: 保存 code_agent 原始输出作为 fallback:
```python
files = _agent_result.files
_code_agent_original_files = {k: v for k, v in files.items()}
```

修改 3: 修复循环失败时自动回退到原始代码:
```python
if _has_critical and _code_agent_active and _code_agent_original_files:
    _fallback_ok = all(
        validate_code(fc).ok for fn, fc in _code_agent_original_files.items()
        if fn.endswith(".py")
    )
    if _fallback_ok:
        files = _code_agent_original_files
```

### 4.3 `code_agent.py` — Phase 4 review 安全回退

**文件**: `backend/agent/researchclaw/pipeline/code_agent.py`

在 Phase 4 review 之后新增安全检查：
```python
pre_review_files = dict(best.files)
best.files, review_rounds = self._phase4_review(...)
# Safety: if review broke previously-valid code, revert
from researchclaw.experiment.validator import validate_code as _vc
for fname, code in best.files.items():
    if fname.endswith(".py") and not _vc(code).ok:
        if fname in pre_review_files and _vc(pre_review_files[fname]).ok:
            best.files[fname] = pre_review_files[fname]
```

### 4.4 `prompts.py` — Prompt 重构

**文件**: `backend/agent/researchclaw/prompts.py`

将 code_generation prompt 从 347 行 → 约 80 行：
- 删除 RL detach、KD 稳定性、PPO 实现等不相关的领域特定指导（约 200 行）
- 将 20+ CRITICAL 重构为 TIER 1/2/3 三层优先级
- 新增正面代码范例（Minimum Viable main.py Structure）
- 保留核心要求：真实算法、多条件、多种子、指标定义

---

## 五、涉及的所有代码文件位置

| 文件 | 修改类型 | 行号范围 |
|------|---------|---------|
| `backend/agent/researchclaw/experiment/validator.py` | 新增函数 + 增强检测 | `check_main_entry_point()`, `check_code_complexity()` 中的 `fake_metric_patterns` |
| `backend/agent/researchclaw/pipeline/executor.py` | 集成检测 + fallback | `_execute_code_generation()` 中的 quality check 和 BLOCK 逻辑 |
| `backend/agent/researchclaw/pipeline/code_agent.py` | review 安全回退 | `generate()` 中 Phase 5 后的验证逻辑 |
| `backend/agent/researchclaw/prompts.py` | Prompt 重构 | `code_generation` 条目的 `user` 字段 |

---

## 六、预期改进效果

| 改进维度 | 修复前 | 修复后 |
|---------|--------|--------|
| main() 入口 | 4/4 成功项目缺失 | 检测并拒绝无入口代码 |
| 虚假指标 | 仅检测 5 种模式 | 检测 9+ 种模式（含随机数、哈希、公式化）|
| Prompt 质量 | 347 行信息过载，20+ CRITICAL | 80 行分层优先级 + 正面范例 |
| Review 安全性 | 可能破坏有效代码 | 自动回退到 pre-review 版本 |
| 修复循环 | 5 次失败即 BLOCK | 失败后回退到 code_agent 原始输出 |
