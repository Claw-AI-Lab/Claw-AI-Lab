// ===================== AutoResearchClaw Stage Definitions =====================

export const RCStage = {
  TOPIC_INIT: 1,
  PROBLEM_DECOMPOSE: 2,
  SEARCH_STRATEGY: 3,
  LITERATURE_COLLECT: 4,
  LITERATURE_SCREEN: 5,
  KNOWLEDGE_EXTRACT: 6,
  SYNTHESIS: 7,
  DISCUSSION: 100,
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
  KNOWLEDGE_SUMMARY: 18,
  PAPER_OUTLINE: 19,
  PAPER_DRAFT: 20,
  PEER_REVIEW: 21,
  PAPER_REVISION: 22,
} as const;

export type RCStage = (typeof RCStage)[keyof typeof RCStage];

export interface StageMeta {
  id: RCStage;
  displayNumber: number;
  name: string;
  key: string;
  outputs: string[];
}

export const STAGE_META: Record<RCStage, StageMeta> = {
  1:  { id: 1,  displayNumber: 1,  name: '课题初始化',   key: 'TOPIC_INIT',         outputs: ['goal.md', 'hardware_profile.json'] },
  2:  { id: 2,  displayNumber: 2,  name: '问题分解',     key: 'PROBLEM_DECOMPOSE',   outputs: ['problem_tree.md'] },
  3:  { id: 3,  displayNumber: 3,  name: '检索策略',     key: 'SEARCH_STRATEGY',     outputs: ['search_plan.yaml', 'sources.json', 'queries.json'] },
  4:  { id: 4,  displayNumber: 4,  name: '文献收集',     key: 'LITERATURE_COLLECT',  outputs: ['candidates.jsonl'] },
  5:  { id: 5,  displayNumber: 5,  name: '文献筛选',  key: 'LITERATURE_SCREEN',   outputs: ['shortlist.jsonl'] },
  6:  { id: 6,  displayNumber: 6,  name: '知识提取',     key: 'KNOWLEDGE_EXTRACT',   outputs: ['cards/'] },
  7:  { id: 7,  displayNumber: 7,  name: '知识综合',     key: 'SYNTHESIS',           outputs: ['synthesis.md'] },
  100:{ id: 100,displayNumber: 0,  name: '沟通讨论',    key: 'DISCUSSION',          outputs: ['discussion_transcript.md', 'consensus_synthesis.md'] },
  8:  { id: 8,  displayNumber: 8,  name: '假设生成',     key: 'HYPOTHESIS_GEN',      outputs: ['hypotheses.md'] },
  9:  { id: 9,  displayNumber: 9,  name: '实验设计',  key: 'EXPERIMENT_DESIGN',   outputs: ['exp_plan.yaml'] },
  10: { id: 10, displayNumber: 10, name: '代码库检索',   key: 'CODEBASE_SEARCH',     outputs: ['codebase_candidates.json'] },
  11: { id: 11, displayNumber: 11, name: '代码生成',     key: 'CODE_GENERATION',     outputs: ['experiment/', 'experiment_spec.md'] },
  12: { id: 12, displayNumber: 12, name: '代码检验',     key: 'SANITY_CHECK',        outputs: ['sanity_report.json'] },
  13: { id: 13, displayNumber: 13, name: '资源规划',     key: 'RESOURCE_PLANNING',   outputs: ['schedule.json'] },
  14: { id: 14, displayNumber: 14, name: '实验执行',     key: 'EXPERIMENT_RUN',      outputs: ['runs/'] },
  15: { id: 15, displayNumber: 15, name: '迭代优化',     key: 'ITERATIVE_REFINE',    outputs: ['refinement_log.json', 'experiment_final/'] },
  16: { id: 16, displayNumber: 16, name: '结果分析',     key: 'RESULT_ANALYSIS',     outputs: ['analysis.md', 'experiment_summary.json', 'charts/'] },
  17: { id: 17, displayNumber: 17, name: '研究决策',     key: 'RESEARCH_DECISION',   outputs: ['decision.md'] },
  18: { id: 18, displayNumber: 18, name: '知识归纳',     key: 'KNOWLEDGE_SUMMARY',   outputs: ['knowledge_entry.json'] },
  19: { id: 19, displayNumber: 19, name: '论文大纲',     key: 'PAPER_OUTLINE',       outputs: ['outline.md'] },
  20: { id: 20, displayNumber: 20, name: '论文初稿',     key: 'PAPER_DRAFT',         outputs: ['paper_draft.md'] },
  21: { id: 21, displayNumber: 21, name: '同行评审',     key: 'PEER_REVIEW',         outputs: ['reviews.md'] },
  22: { id: 22, displayNumber: 22, name: '论文修订',     key: 'PAPER_REVISION',      outputs: ['paper_revised.md', 'latex_package.zip'] },
};

// ===================== Pyramid Layer Definitions =====================

export const AgentLayer = {
  IDEA: 'idea',
  EXPERIMENT: 'experiment',
  CODING: 'coding',
  EXECUTION: 'execution',
  WRITING: 'writing',
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
    stages: [1, 2, 3, 4, 5, 6, 7, 100, 8],
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
    stages: [14, 15, 16, 17, 18],
  },
  [AgentLayer.WRITING]: {
    name: '第五层 · 论文写作',
    color: '#a855f7',
    desc: 'Phase G: 论文大纲 → 初稿 → 同行评审 → 修订',
    stages: [19, 20, 21, 22],
  },
};

export const ALL_LAYERS: readonly AgentLayer[] = [
  AgentLayer.IDEA,
  AgentLayer.EXPERIMENT,
  AgentLayer.CODING,
  AgentLayer.EXECUTION,
  AgentLayer.WRITING,
];

// ===================== Shared Data Repositories =====================

export const RepoId = {
  KNOWLEDGE: 'knowledge',
  EXP_DESIGN: 'exp_design',
  CODEBASE: 'codebase',
  RESULTS: 'results',
  INSIGHTS: 'insights',
  PAPERS: 'papers',
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
    name: 'Idea 仓库',
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
  [RepoId.INSIGHTS]: {
    name: '知识库',
    icon: '🧠',
    desc: '跨项目研究结论、洞察、后续方向',
    fromLayer: AgentLayer.EXECUTION,
    toLayer: AgentLayer.IDEA,
    artifacts: ['knowledge_entry.json'],
  },
  [RepoId.PAPERS]: {
    name: '论文仓库',
    icon: '📝',
    desc: '论文大纲、初稿、评审、修订稿',
    fromLayer: AgentLayer.WRITING,
    toLayer: null,
    artifacts: ['outline.md', 'paper_draft.md', 'reviews.md', 'paper_revised.md'],
  },
};

export const ALL_REPOS: readonly RepoId[] = [
  RepoId.KNOWLEDGE,
  RepoId.INSIGHTS,
  RepoId.PAPERS,
];

// ===================== Agent & Runtime Types =====================

export type AgentStatus = 'idle' | 'working' | 'error' | 'done' | 'waiting_discussion' | 'discussing';
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'waiting' | 'discussing';

export interface LobsterAgent {
  id: string;
  name: string;
  layer: AgentLayer;
  status: AgentStatus;
  currentStage: RCStage | null;
  currentTask: string;
  stageProgress: Record<number, StageStatus>;
  runId: string;
  projectId?: string;
  roleTag?: string;
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
  content?: string;
  stage?: number;
}

export const ARTIFACT_LABELS: Record<string, { icon: string; zh: string; en: string }> = {
  'goal.md':               { icon: '🎯', zh: '研究目标', en: 'Research Goal' },
  'hardware_profile.json': { icon: '🖥️', zh: '硬件检测', en: 'Hardware Profile' },
  'problem_tree.md':       { icon: '🌳', zh: '问题分解树', en: 'Problem Tree' },
  'search_plan.yaml':      { icon: '🔍', zh: '检索策略', en: 'Search Strategy' },
  'sources.json':          { icon: '📡', zh: '数据源', en: 'Data Sources' },
  'queries.json':          { icon: '🔎', zh: '检索查询', en: 'Search Queries' },
  'candidates.jsonl':      { icon: '📚', zh: '候选文献', en: 'Candidate Papers' },
  'shortlist.jsonl':       { icon: '✅', zh: '精选文献', en: 'Shortlisted Papers' },
  'cards/':                { icon: '🗂️', zh: '知识卡片', en: 'Knowledge Cards' },
  'synthesis.md':          { icon: '🧬', zh: '知识综合报告', en: 'Synthesis Report' },
  'hypotheses.md':         { icon: '💡', zh: '研究假设', en: 'Research Hypotheses' },
  'exp_plan.yaml':         { icon: '🧪', zh: '实验方案', en: 'Experiment Plan' },
  'schedule.json':         { icon: '📅', zh: '资源调度', en: 'Resource Schedule' },
  'codebase_candidates.json': { icon: '🔗', zh: '参考代码库', en: 'Reference Codebases' },
  'experiment/':           { icon: '💻', zh: '实验代码', en: 'Experiment Code' },
  'experiment_spec.md':    { icon: '📋', zh: '实验规格', en: 'Experiment Spec' },
  'sanity_report.json':    { icon: '🔬', zh: '冒烟测试报告', en: 'Sanity Report' },
  'runs/':                 { icon: '▶️', zh: '运行结果', en: 'Run Results' },
  'refinement_log.json':   { icon: '🔄', zh: '迭代日志', en: 'Refinement Log' },
  'experiment_final/':     { icon: '🏁', zh: '最终实验', en: 'Final Experiment' },
  'analysis.md':           { icon: '📊', zh: '结果分析', en: 'Result Analysis' },
  'experiment_summary.json': { icon: '📈', zh: '实验摘要', en: 'Experiment Summary' },
  'charts/':               { icon: '📉', zh: '实验图表', en: 'Charts' },
  'decision.md':           { icon: '🧭', zh: '研究决策', en: 'Research Decision' },
  'knowledge_entry.json':  { icon: '🧠', zh: '知识条目', en: 'Knowledge Entry' },
  'outline.md':            { icon: '📝', zh: '论文大纲', en: 'Paper Outline' },
  'paper_draft.md':        { icon: '📄', zh: '论文初稿', en: 'Paper Draft' },
  'reviews.md':            { icon: '👁️', zh: '同行评审', en: 'Peer Reviews' },
  'paper_revised.md':      { icon: '✍️', zh: '论文终稿', en: 'Paper Revised' },
  'latex_package.zip':     { icon: '📦', zh: 'LaTeX 包', en: 'LaTeX Package' },
  'discussion_transcript.md': { icon: '💬', zh: '讨论记录', en: 'Discussion Transcript' },
  'consensus_synthesis.md':   { icon: '🤝', zh: '共识综合', en: 'Consensus Synthesis' },
  'pre_discussion_syntheses.md': { icon: '📋', zh: '讨论前综合', en: 'Pre-discussion Syntheses' },
};

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
  acceleratorLabel?: string;
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

// ===================== Human Feedback =====================

export interface ChatMessage {
  id: string;
  role: 'user' | 'system';
  content: string;
  targetLayer?: string;
  timestamp: number;
}

// ===================== Project Management =====================

export type ProjectStatus = 'running' | 'queued' | 'completed' | 'interrupted' | 'new';

export interface ProjectInfo {
  projectId: string;
  status: ProjectStatus;
  lastCompletedStage: number;
  lastCompletedName: string;
  firstStage: number;
  totalStages: number;
  timestamp: string;
  topic: string;
  configPath: string;
  intervention?: string;
}

// ===================== WebSocket Protocol =====================

export type WSMessage =
  | { type: 'agent_update'; payload: LobsterAgent }
  | { type: 'artifact_produced'; payload: Artifact }
  | { type: 'log'; payload: LogEntry }
  | { type: 'stage_update'; payload: { agentId: string; stage: RCStage; status: StageStatus } }
  | { type: 'resource_stats'; payload: ResourceStats }
  | { type: 'queue_update'; payload: QueueMap }
  | { type: 'chat_message'; payload: ChatMessage }
  | { type: 'project_list'; payload: ProjectInfo[] }
  | { type: 'system'; payload: { message: string } };

// ===================== App State =====================

export interface AppState {
  agents: LobsterAgent[];
  artifacts: Artifact[];
  logs: LogEntry[];
  queues: QueueMap;
  chatMessages: ChatMessage[];
  projects: ProjectInfo[];
  selectedProjectId: string | null;
  resources: ResourceStats | null;
  resConnected: boolean;
  connected: boolean;
  mockMode: boolean;
}
