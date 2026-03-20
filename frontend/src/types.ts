// ===================== AutoResearchClaw Stage Definitions =====================

export const RCStage = {
  TOPIC_INIT: 1,
  PROBLEM_DECOMPOSE: 2,
  SEARCH_STRATEGY: 3,
  LITERATURE_COLLECT: 4,
  LITERATURE_SCREEN: 5,
  KNOWLEDGE_EXTRACT: 6,
  SYNTHESIS: 7,
  HYPOTHESIS_GEN: 8,
  EXPERIMENT_DESIGN: 9,
  CODEBASE_SEARCH: 10,
  CODE_GENERATION: 11,
  SANITY_CHECK: 12,
  RESOURCE_PLANNING: 13,
  EXPERIMENT_RUN: 14,
  ITERATIVE_REFINE: 15,
  RESULT_ANALYSIS: 16,
  RESEARCH_DECISION: 17,
} as const;

export type RCStage = (typeof RCStage)[keyof typeof RCStage];

export interface StageMeta {
  id: RCStage;
  name: string;
  key: string;
  outputs: string[];
}

export const STAGE_META: Record<RCStage, StageMeta> = {
  1:  { id: 1,  name: '课题初始化',   key: 'TOPIC_INIT',         outputs: ['goal.md', 'hardware_profile.json'] },
  2:  { id: 2,  name: '问题分解',     key: 'PROBLEM_DECOMPOSE',   outputs: ['problem_tree.md'] },
  3:  { id: 3,  name: '检索策略',     key: 'SEARCH_STRATEGY',     outputs: ['search_plan.yaml', 'sources.json', 'queries.json'] },
  4:  { id: 4,  name: '文献收集',     key: 'LITERATURE_COLLECT',  outputs: ['candidates.jsonl'] },
  5:  { id: 5,  name: '文献筛选 ⛩',  key: 'LITERATURE_SCREEN',   outputs: ['shortlist.jsonl'] },
  6:  { id: 6,  name: '知识提取',     key: 'KNOWLEDGE_EXTRACT',   outputs: ['cards/'] },
  7:  { id: 7,  name: '知识综合',     key: 'SYNTHESIS',           outputs: ['synthesis.md'] },
  8:  { id: 8,  name: '假设生成',     key: 'HYPOTHESIS_GEN',      outputs: ['hypotheses.md'] },
  9:  { id: 9,  name: '实验设计 ⛩',  key: 'EXPERIMENT_DESIGN',   outputs: ['exp_plan.yaml'] },
  10: { id: 10, name: '代码库检索',   key: 'CODEBASE_SEARCH',     outputs: ['codebase_candidates.json'] },
  11: { id: 11, name: '代码生成',     key: 'CODE_GENERATION',     outputs: ['experiment/', 'experiment_spec.md'] },
  12: { id: 12, name: '代码检验',     key: 'SANITY_CHECK',        outputs: ['sanity_report.json'] },
  13: { id: 13, name: '资源规划',     key: 'RESOURCE_PLANNING',   outputs: ['schedule.json'] },
  14: { id: 14, name: '实验执行',     key: 'EXPERIMENT_RUN',      outputs: ['runs/'] },
  15: { id: 15, name: '迭代优化',     key: 'ITERATIVE_REFINE',    outputs: ['refinement_log.json', 'experiment_final/'] },
  16: { id: 16, name: '结果分析',     key: 'RESULT_ANALYSIS',     outputs: ['analysis.md', 'experiment_summary.json', 'charts/'] },
  17: { id: 17, name: '研究决策',     key: 'RESEARCH_DECISION',   outputs: ['decision.md'] },
};

// ===================== Pyramid Layer Definitions =====================

export const AgentLayer = {
  IDEA: 'idea',
  EXPERIMENT: 'experiment',
  CODING: 'coding',
  EXECUTION: 'execution',
} as const;

export type AgentLayer = (typeof AgentLayer)[keyof typeof AgentLayer];

export interface LayerMeta {
  name: string;
  color: string;
  desc: string;
  stages: RCStage[];
}

export const LAYER_META: Record<AgentLayer, LayerMeta> = {
  [AgentLayer.IDEA]: {
    name: '第一层 · 调研与创意',
    color: '#f59e0b',
    desc: 'Phase A→C: 课题定义 → 文献调研 → 知识综合 → 假设生成',
    stages: [1, 2, 3, 4, 5, 6, 7, 8],
  },
  [AgentLayer.EXPERIMENT]: {
    name: '第二层 · 实验设计',
    color: '#3b82f6',
    desc: 'Phase D: 实验方案设计',
    stages: [9],
  },
  [AgentLayer.CODING]: {
    name: '第三层 · 代码与资源',
    color: '#10b981',
    desc: 'Phase D: 代码库检索 + 代码生成 + 资源规划',
    stages: [10, 11, 12, 13],
  },
  [AgentLayer.EXECUTION]: {
    name: '第四层 · 执行与修正',
    color: '#ef4444',
    desc: 'Phase E→F: 实验执行 → 迭代优化 → 结果分析 → 决策',
    stages: [14, 15, 16, 17],
  },
};

export const ALL_LAYERS: readonly AgentLayer[] = [
  AgentLayer.IDEA,
  AgentLayer.EXPERIMENT,
  AgentLayer.CODING,
  AgentLayer.EXECUTION,
];

// ===================== Shared Data Repositories =====================

export const RepoId = {
  KNOWLEDGE: 'knowledge',
  EXP_DESIGN: 'exp_design',
  CODEBASE: 'codebase',
  RESULTS: 'results',
} as const;

export type RepoId = (typeof RepoId)[keyof typeof RepoId];

export interface RepoMeta {
  name: string;
  icon: string;
  desc: string;
  fromLayer: AgentLayer;
  toLayer: AgentLayer | null;
  artifacts: string[];
}

export const REPO_META: Record<RepoId, RepoMeta> = {
  [RepoId.KNOWLEDGE]: {
    name: 'Idea & Knowledge 仓库',
    icon: '💡',
    desc: '文献卡片、知识综合、研究假设',
    fromLayer: AgentLayer.IDEA,
    toLayer: AgentLayer.EXPERIMENT,
    artifacts: ['goal.md', 'problem_tree.md', 'shortlist.jsonl', 'cards/', 'synthesis.md', 'hypotheses.md'],
  },
  [RepoId.EXP_DESIGN]: {
    name: '实验设计仓库',
    icon: '🧪',
    desc: '实验方案、资源调度计划',
    fromLayer: AgentLayer.EXPERIMENT,
    toLayer: AgentLayer.CODING,
    artifacts: ['exp_plan.yaml', 'schedule.json'],
  },
  [RepoId.CODEBASE]: {
    name: '代码仓库',
    icon: '💻',
    desc: '实验代码、规格说明',
    fromLayer: AgentLayer.CODING,
    toLayer: AgentLayer.EXECUTION,
    artifacts: ['experiment/', 'experiment_spec.md'],
  },
  [RepoId.RESULTS]: {
    name: '结果仓库',
    icon: '📊',
    desc: '实验结果、分析报告、决策',
    fromLayer: AgentLayer.EXECUTION,
    toLayer: null,
    artifacts: ['runs/', 'analysis.md', 'experiment_summary.json', 'charts/', 'decision.md'],
  },
};

export const ALL_REPOS: readonly RepoId[] = [
  RepoId.KNOWLEDGE,
  RepoId.EXP_DESIGN,
  RepoId.CODEBASE,
  RepoId.RESULTS,
];

// ===================== Agent & Runtime Types =====================

export type AgentStatus = 'idle' | 'working' | 'error' | 'done';
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface LobsterAgent {
  id: string;
  name: string;
  layer: AgentLayer;
  status: AgentStatus;
  currentStage: RCStage | null;
  currentTask: string;
  stageProgress: Record<number, StageStatus>;
  runId: string;
}

export interface Artifact {
  id: string;
  repoId: RepoId;
  projectId: string;
  filename: string;
  producedBy: string;
  timestamp: number;
  size: string;
  status: 'fresh' | 'stale' | 'error';
}

export interface LogEntry {
  id: string;
  agentId: string;
  agentName: string;
  layer: AgentLayer;
  stage: RCStage | null;
  message: string;
  level: 'info' | 'success' | 'warning' | 'error';
  timestamp: number;
}

// ===================== Resource Monitoring =====================

export interface GpuInfo {
  id: number;
  name: string;
  utilization: number;
  memUsed: number;
  memTotal: number;
  temperature: number;
}

export interface ResourceStats {
  cpuPercent: number;
  memUsed: number;
  memTotal: number;
  gpus: GpuInfo[];
  timestamp: number;
}

// ===================== Task Queues =====================

export interface QueueSummary {
  name: string;
  total: number;
  pending: number;
  assigned: number;
  completed: number;
}

export type QueueMap = Record<string, QueueSummary>;

// ===================== WebSocket Protocol =====================

export type WSMessage =
  | { type: 'agent_update'; payload: LobsterAgent }
  | { type: 'artifact_produced'; payload: Artifact }
  | { type: 'log'; payload: LogEntry }
  | { type: 'stage_update'; payload: { agentId: string; stage: RCStage; status: StageStatus } }
  | { type: 'resource_stats'; payload: ResourceStats }
  | { type: 'queue_update'; payload: QueueMap }
  | { type: 'system'; payload: { message: string } };

// ===================== App State =====================

export interface AppState {
  agents: LobsterAgent[];
  artifacts: Artifact[];
  logs: LogEntry[];
  queues: QueueMap;
  resources: ResourceStats | null;
  resConnected: boolean;
  connected: boolean;
  mockMode: boolean;
}
