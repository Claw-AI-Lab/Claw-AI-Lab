import { useState } from 'react';
import type { ProjectInfo } from '../types';
import { STAGE_META, RCStage } from '../types';

const STATUS_CFG: Record<string, { label: string; color: string; icon: string }> = {
  running:     { label: '运行中', color: '#22c55e', icon: '▶' },
  queued:      { label: '排队中', color: '#f59e0b', icon: '⏳' },
  completed:   { label: '已完成', color: '#3b82f6', icon: '✓' },
  interrupted: { label: '已中断', color: '#ef4444', icon: '⏸' },
  new:         { label: '新建',   color: '#94a3b8', icon: '○' },
};

type SubmitMode = 'lab' | 'reproduce';

const MODE_INFO: Record<SubmitMode, { label: string; icon: string; placeholder: string; desc: string }> = {
  lab: {
    label: 'Lab 探索',
    icon: '🔬',
    placeholder: '研究具身智能中 video action model 的最新进展',
    desc: '多方向并行调研 → 跨领域讨论 → 统一假设',
  },
  reproduce: {
    label: '论文复现',
    icon: '📄',
    placeholder: '复现 SwitchCraft (arXiv:2602.23956) 的注意力控制方法',
    desc: '单 Agent 全流程复现',
  },
};

function stageName(n: number): string {
  const meta = STAGE_META[n as RCStage];
  return meta ? meta.name : `S${n}`;
}

interface Props {
  projects: ProjectInfo[];
  connected: boolean;
  selectedProjectId: string | null;
  onSelect: (projectId: string) => void;
  onResume: (projectId: string) => void;
  onDelete: (projectId: string) => void;
  onQuickSubmit: (topic: string, mode: SubmitMode, researchAngles: string[]) => void;
}

export default function ProjectPanel({ projects, connected, selectedProjectId, onSelect, onResume, onDelete, onQuickSubmit }: Props) {
  const [expanded, setExpanded] = useState(true);
  const [mode, setMode] = useState<SubmitMode>('lab');
  const [topicInput, setTopicInput] = useState('');
  const [anglesInput, setAnglesInput] = useState('');

  const info = MODE_INFO[mode];

  const submit = () => {
    const text = topicInput.trim();
    if (!text) return;
    const angles = mode === 'lab'
      ? anglesInput.split(/[,，、;；]/).map(s => s.trim()).filter(Boolean)
      : [];
    onQuickSubmit(text, mode, angles);
    setTopicInput('');
    setAnglesInput('');
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  const sorted = [...projects].sort((a, b) => {
    const o: Record<string, number> = { running: 0, queued: 1, interrupted: 2, new: 3, completed: 4 };
    return (o[a.status] ?? 5) - (o[b.status] ?? 5);
  });

  const running = projects.filter(p => p.status === 'running').length;
  const interrupted = projects.filter(p => p.status === 'interrupted').length;

  const anglesCount = anglesInput.split(/[,，、;；]/).filter(s => s.trim()).length;
  const goLabel = mode === 'lab'
    ? (anglesInput.trim() ? `${anglesCount} 方向并行研究` : '3 方向自动探索')
    : '开始复现';

  return (
    <div className="project-panel">
      {/* Header */}
      <div className="project-panel-header" onClick={() => setExpanded(!expanded)}>
        <h3>
          <span className="panel-icon">📋</span> 项目管理
          {projects.length > 0 && <span className="project-count">{projects.length}</span>}
          {running > 0 && <span className="project-count running-count">{running} 运行</span>}
          {interrupted > 0 && <span className="project-count interrupted-count">{interrupted} 中断</span>}
        </h3>
        <span className="expand-arrow">{expanded ? '▾' : '▸'}</span>
      </div>

      {expanded && (
        <div className="project-panel-body">
          {/* --- Submit Section --- */}
          <div className="submit-section">
            <div className="mode-selector">
              {(['lab', 'reproduce'] as SubmitMode[]).map(m => (
                <button
                  key={m}
                  className={`mode-btn ${mode === m ? 'active' : ''}`}
                  onClick={() => setMode(m)}
                >
                  {MODE_INFO[m].icon} {MODE_INFO[m].label}
                </button>
              ))}
            </div>
            <div className="mode-desc">{info.desc}</div>

            <div className="project-input-stack">
              <textarea
                className="project-topic-input"
                placeholder={info.placeholder}
                value={topicInput}
                onChange={e => setTopicInput(e.target.value)}
                onKeyDown={onKey}
                rows={2}
                disabled={!connected}
              />
              {mode === 'lab' && (
                <div className="project-angles-row">
                  <input
                    className="project-angles-input"
                    placeholder="研究方向 (可选, 逗号分隔, 如: VLM, World Model, VLA)"
                    value={anglesInput}
                    onChange={e => setAnglesInput(e.target.value)}
                    onKeyDown={onKey}
                    disabled={!connected}
                  />
                </div>
              )}
              <button
                className={`project-go-btn ${mode === 'lab' ? 'lab-mode' : 'reproduce-mode'}`}
                onClick={submit}
                disabled={!connected || !topicInput.trim()}
              >
                {goLabel}
              </button>
            </div>
          </div>

          {/* --- Divider --- */}
          {sorted.length > 0 && <div className="panel-divider" />}

          {/* --- Project List --- */}
          {sorted.length === 0 ? (
            <div className="project-empty">
              {mode === 'lab'
                ? '输入研究主题，多 Agent 并行调研'
                : '输入论文信息，一键复现'}
            </div>
          ) : (
            <div className="project-list-inner">
              {sorted.map(proj => {
                const cfg = STATUS_CFG[proj.status] || STATUS_CFG.new;
                const pct = proj.totalStages > 0
                  ? Math.round((proj.lastCompletedStage / proj.totalStages) * 100)
                  : 0;
                const isSelected = selectedProjectId === proj.projectId;

                return (
                  <div
                    key={proj.projectId}
                    className={`project-card project-${proj.status}${isSelected ? ' selected' : ''}`}
                    onClick={() => onSelect(proj.projectId)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="project-card-top">
                      <span className="project-status-badge" style={{ background: cfg.color }}>
                        {cfg.icon} {cfg.label}
                      </span>
                      <span className="project-id" title={proj.projectId}>
                        {proj.projectId}
                      </span>
                      <button
                        className="project-delete-btn"
                        title="删除项目"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm(`确认删除项目 "${proj.projectId}"？\n此操作将删除所有阶段数据，不可恢复。`)) {
                            onDelete(proj.projectId);
                          }
                        }}
                      >
                        ✕
                      </button>
                    </div>

                    {proj.topic && (
                      <div className="project-topic" title={proj.topic}>{proj.topic}</div>
                    )}

                    <div className="project-progress-row">
                      <div className="project-progress-bar">
                        <div
                          className="project-progress-fill"
                          style={{ width: `${pct}%`, background: cfg.color }}
                        />
                      </div>
                      <span className="project-progress-text">
                        {proj.lastCompletedStage}/{proj.totalStages}
                      </span>
                    </div>

                    {proj.lastCompletedStage > 0 && (
                      <div className="project-stage-info">
                        {stageName(proj.lastCompletedStage)}
                        {proj.lastCompletedStage < proj.totalStages && (
                          <span className="project-next-stage">
                            {' → '}{stageName(proj.lastCompletedStage + 1)}
                          </span>
                        )}
                      </div>
                    )}

                    {proj.timestamp && (
                      <div className="project-timestamp">
                        {new Date(proj.timestamp).toLocaleString('zh-CN')}
                      </div>
                    )}

                    <div className="project-card-actions">
                      {proj.status === 'interrupted' && connected && (
                        <button className="project-resume-btn" onClick={(e) => { e.stopPropagation(); onResume(proj.projectId); }}>
                          ▶ 断点续跑
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
