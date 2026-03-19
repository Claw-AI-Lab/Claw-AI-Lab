import { AgentLayer, STAGE_META, LAYER_META, RepoId } from './types';
import type { LobsterAgent, WSMessage, Artifact, RCStage as RCStageT } from './types';

let counter = 0;
const uid = () => `m-${++counter}-${Date.now()}`;
const pick = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];

function makeAgent(id: string, name: string, layer: AgentLayer, runId: string): LobsterAgent {
  const stages = LAYER_META[layer].stages;
  const progress: Record<number, 'pending'> = {};
  for (const s of stages) progress[s] = 'pending';
  return { id, name, layer, status: 'idle', currentStage: null, currentTask: '', stageProgress: progress, runId };
}

export const INITIAL_AGENTS: LobsterAgent[] = [
  makeAgent('L1-01', '🦞 调研长·Alpha',     AgentLayer.IDEA,       'run-001'),
  makeAgent('L1-02', '🦞 调研员·Beta',       AgentLayer.IDEA,       'run-002'),
  makeAgent('L2-01', '🦞 实验师·α',          AgentLayer.EXPERIMENT, 'run-001'),
  makeAgent('L2-02', '🦞 实验师·β',          AgentLayer.EXPERIMENT, 'run-002'),
  makeAgent('L3-01', '🦞 码农·甲',           AgentLayer.CODING,     'run-001'),
  makeAgent('L3-02', '🦞 码农·乙',           AgentLayer.CODING,     'run-002'),
  makeAgent('L3-03', '🦞 码农·丙',           AgentLayer.CODING,     'run-003'),
  makeAgent('L4-01', '🦞 执行者·壹',         AgentLayer.EXECUTION,  'run-001'),
  makeAgent('L4-02', '🦞 执行者·貳',         AgentLayer.EXECUTION,  'run-002'),
  makeAgent('L4-03', '🦞 执行者·叁',         AgentLayer.EXECUTION,  'run-003'),
  makeAgent('L4-04', '🦞 执行者·肆',         AgentLayer.EXECUTION,  'run-004'),
];

const TASK_DETAILS: Record<number, string[]> = {
  1:  ['解析研究课题，生成 SMART 目标...', '检测硬件环境 (GPU/CPU)...'],
  2:  ['将课题分解为子问题树...', '确定优先研究方向...'],
  3:  ['规划检索策略，确定数据源...', '生成关键词查询组合...'],
  4:  ['调用 OpenAlex API 检索论文...', '从 Semantic Scholar 收集引用...', '扫描 arXiv 最新预印本...'],
  5:  ['基于相关性和质量筛选文献...', '评估 shortlist 覆盖度... [GATE]'],
  6:  ['从筛选论文提取知识卡片...', '结构化关键发现...'],
  7:  ['聚类研究主题，识别研究空白...', '综合多论文结论...'],
  8:  ['生成可证伪假设...', '多 agent 辩论评估假设...'],
  9:  ['设计实验方案 (YAML)...', '确定 baseline 和评估指标... [GATE]', '规划消融实验矩阵...'],
  11: ['估算 GPU 需求和运行时间...', '生成调度计划 schedule.json...'],
  10: ['生成实验核心代码...', 'AST 验证代码正确性...', '适配硬件环境...', '编写评估脚本...'],
  12: ['在沙箱中执行实验代码...', '监控 NaN/Inf 检查...', '提交训练任务至集群...'],
  13: ['编辑-运行-评估循环...', 'LLM 修复失败用例...', '收敛性评估...'],
  14: ['分析实验指标...', '生成可视化图表...', '多 agent 结果评审...'],
  15: ['做出研究决策: PROCEED / PIVOT / REFINE...', '记录决策历史...'],
};

const LOG_TEMPLATES: Record<string, Array<{ msg: string; level: 'info' | 'success' | 'warning' | 'error' }>> = {
  [AgentLayer.IDEA]: [
    { msg: '发现 5 篇高相关度新论文 (OpenAlex)', level: 'info' },
    { msg: '文献筛选通过: shortlist 12 篇 → 8 篇', level: 'success' },
    { msg: '知识卡片提取完成: 8 cards', level: 'success' },
    { msg: 'Semantic Scholar API 速率限制，等待重试...', level: 'warning' },
    { msg: '综合分析发现 3 个研究空白', level: 'info' },
    { msg: '生成 2 个可证伪假设', level: 'success' },
    { msg: 'arXiv 连接超时', level: 'error' },
    { msg: '假设辩论: Agent 2/3 同意，1 反对 → 通过', level: 'info' },
  ],
  [AgentLayer.EXPERIMENT]: [
    { msg: '实验方案: 5 组对照 + 3 组消融', level: 'success' },
    { msg: '预计 GPU 需求: 4×A100, 约 12h', level: 'info' },
    { msg: '资源不足: 需要 8×A100 但只有 4×', level: 'warning' },
    { msg: '实验方案 GATE 审核通过', level: 'success' },
    { msg: 'schedule.json 已生成', level: 'info' },
    { msg: 'Baseline 确定: LLaMA-7B + LoRA', level: 'info' },
  ],
  [AgentLayer.CODING]: [
    { msg: '数据预处理 pipeline 编写完成', level: 'success' },
    { msg: '模型核心模块: AST 验证通过', level: 'success' },
    { msg: 'Lint 检查: 2 warnings, 0 errors', level: 'warning' },
    { msg: '训练循环 + WandB 日志集成完成', level: 'success' },
    { msg: '类型错误: Tensor shape mismatch [B,S,D]', level: 'error' },
    { msg: '评估脚本 (BLEU/ROUGE/F1) 编写完成', level: 'success' },
    { msg: '代码已提交至 experiment/ 目录', level: 'info' },
  ],
  [AgentLayer.EXECUTION]: [
    { msg: '训练任务已提交 (sandbox: docker)', level: 'info' },
    { msg: 'Epoch 3/10, loss=1.87, lr=2e-4', level: 'info' },
    { msg: 'CUDA OOM! 降低 batch_size 16→8 重试', level: 'error' },
    { msg: '迭代修复 #2: 梯度裁剪 max_norm=1.0', level: 'warning' },
    { msg: '评估完成: BLEU=32.5, ROUGE-L=41.2', level: 'success' },
    { msg: '检测到 loss 发散 → 回滚 checkpoint-ep5', level: 'warning' },
    { msg: '结果分析: 假设 H1 显著 (p<0.01), H2 不显著', level: 'success' },
    { msg: '决策: PROCEED → 实验达到预期', level: 'success' },
    { msg: '决策: PIVOT → 回退至假设生成重新探索', level: 'warning' },
  ],
};

function stageToRepo(stage: RCStageT): RepoId | null {
  if (stage <= 8) return RepoId.KNOWLEDGE;
  if (stage === 9 || stage === 11) return RepoId.EXP_DESIGN;
  if (stage === 10) return RepoId.CODEBASE;
  if (stage >= 12) return RepoId.RESULTS;
  return null;
}

export function createMockMessageGenerator(onMessage: (msg: WSMessage) => void): () => void {
  const intervals: number[] = [];
  const agents = [...INITIAL_AGENTS];
  const agentMap = new Map(agents.map((a) => [a.id, { ...a }]));

  const emitAgentActivity = () => {
    const agent = pick(agents);
    const state = agentMap.get(agent.id)!;
    const layerStages = LAYER_META[agent.layer].stages;
    const roll = Math.random();

    if (roll < 0.5 && state.status !== 'working') {
      const stage = pick(layerStages);
      state.status = 'working';
      state.currentStage = stage;
      state.currentTask = pick(TASK_DETAILS[stage] || ['处理中...']);
      state.stageProgress[stage] = 'running';
    } else if (roll < 0.75 && state.currentStage) {
      state.stageProgress[state.currentStage] = 'completed';
      state.status = 'done';
      state.currentTask = '';

      const repo = stageToRepo(state.currentStage);
      if (repo) {
        const outputs = STAGE_META[state.currentStage].outputs;
        const file = pick(outputs);
        const artifact: Artifact = {
          id: uid(),
          repoId: repo,
          projectId: agent.runId,
          filename: file,
          producedBy: agent.name,
          timestamp: Date.now(),
          size: `${(Math.random() * 100 + 1).toFixed(1)} KB`,
          status: 'fresh',
        };
        onMessage({ type: 'artifact_produced', payload: artifact });
      }

      const nextStage = state.currentStage;
      state.currentStage = null;
      onMessage({
        type: 'stage_update',
        payload: { agentId: agent.id, stage: nextStage, status: 'completed' },
      });
    } else if (roll < 0.85 && state.currentStage) {
      state.stageProgress[state.currentStage] = 'failed';
      state.status = 'error';
      state.currentTask = '执行失败，等待重试...';
    } else {
      state.status = 'idle';
      state.currentStage = null;
      state.currentTask = '';
    }

    agentMap.set(agent.id, { ...state });
    onMessage({ type: 'agent_update', payload: { ...state } });

    if (state.status !== 'idle') {
      const templates = LOG_TEMPLATES[agent.layer];
      const tmpl = state.status === 'error'
        ? templates.find((t) => t.level === 'error') || pick(templates)
        : pick(templates);
      onMessage({
        type: 'log',
        payload: {
          id: uid(),
          agentId: agent.id,
          agentName: agent.name,
          layer: agent.layer,
          stage: state.currentStage,
          message: tmpl.msg,
          level: tmpl.level,
          timestamp: Date.now(),
        },
      });
    }
  };

  intervals.push(
    window.setInterval(emitAgentActivity, 1500 + Math.random() * 2000),
  );

  setTimeout(emitAgentActivity, 300);
  setTimeout(emitAgentActivity, 800);

  return () => intervals.forEach(clearInterval);
}
