import { useState } from 'react';
import type { ProjectInfo, Artifact } from '../types';
import { STAGE_META, RCStage, REPO_META } from '../types';

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
  artifactsByProject: Record<string, Artifact[]>;
  onSelect: (projectId: string) => void;
  onResume: (projectId: string) => void;
  onDelete: (projectId: string) => void;
  onQuickSubmit: (topic: string, mode: SubmitMode, researchAngles: string[]) => void;
}

export default function ProjectPanel({ projects, connected, selectedProjectId, artifactsByProject, onSelect, onResume, onDelete, onQuickSubmit }: Props) {
  const [panelOpen, setPanelOpen] = useState(true);
  const [mode, setMode] = useState<SubmitMode>('lab');
  const [topicInput, setTopicInput] = useState('');
  const [anglesInput, setAnglesInput] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

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

  const toggleExpand = (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedId(prev => prev === projectId ? null : projectId);
  };

  return (
    <div className="project-panel">
      <div className="project-panel-header" onClick={() => setPanelOpen(!panelOpen)}>
        <h3>
          <span className="panel-icon">📋</span> 项目管理
          {projects.length > 0 && <span className="project-count">{projects.length}</span>}
          {running > 0 && <span className="project-count running-count">{running} 运行</span>}
          {interrupted > 0 && <span className="project-count interrupted-count">{interrupted} 中断</span>}
        </h3>
        <span className="expand-arrow">{panelOpen ? '▾' : '▸'}</span>
      </div>

      {panelOpen && (
        <div className="project-panel-body">
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
                    placeholder="研究方向 (可选, 逗号分隔, 默认: VLM, World Model, VLA)"
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

          {sorted.length > 0 && <div className="panel-divider" />}

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
                const isExpanded = expandedId === proj.projectId;
                const arts = artifactsByProject[proj.projectId] || [];

                return (
                  <div
                    key={proj.projectId}
                    className={`project-card project-${proj.status}${isSelected ? ' selected' : ''}${isExpanded ? ' expanded' : ''}`}
                  >
                    {/* Collapsed summary row — always visible */}
                    <div className="project-summary" onClick={(e) => toggleExpand(proj.projectId, e)}>
                      <span className="project-status-dot" style={{ background: cfg.color }} title={cfg.label}>{cfg.icon}</span>
                      <span className="project-summary-name" title={proj.topic || proj.projectId}>
                        {proj.topic || proj.projectId}
                      </span>
                      <span className="project-summary-progress">{pct}%</span>
                      {arts.length > 0 && <span className="project-summary-arts">📦{arts.length}</span>}
                      <span className={`project-expand-arrow${isExpanded ? ' open' : ''}`}>▸</span>
                    </div>

                    {/* Mini progress bar */}
                    <div className="project-mini-bar" onClick={(e) => toggleExpand(proj.projectId, e)}>
                      <div className="project-mini-fill" style={{ width: `${pct}%`, background: cfg.color }} />
                    </div>

                    {/* Expanded detail */}
                    {isExpanded && (
                      <div className="project-detail">
                        <div className="project-detail-meta">
                          <span className="project-id-label" title={proj.projectId}>{proj.projectId}</span>
                          <span className="project-status-badge" style={{ background: cfg.color }}>
                            {cfg.icon} {cfg.label}
                          </span>
                        </div>

                        <div className="project-detail-stage">
                          {proj.lastCompletedStage}/{proj.totalStages} 阶段
                          {proj.lastCompletedStage > 0 && (
                            <> · {stageName(proj.lastCompletedStage)}
                              {proj.lastCompletedStage < proj.totalStages && (
                                <span className="project-next-stage"> → {stageName(proj.lastCompletedStage + 1)}</span>
                              )}
                            </>
                          )}
                        </div>

                        {proj.timestamp && (
                          <div className="project-detail-time">
                            {new Date(proj.timestamp).toLocaleString('zh-CN')}
                          </div>
                        )}

                        {/* Artifacts by repo */}
                        {arts.length > 0 && (
                          <div className="project-detail-artifacts">
                            {(() => {
                              const grouped: Record<string, Artifact[]> = {};
                              for (const a of arts) {
                                if (!grouped[a.repoId]) grouped[a.repoId] = [];
                                grouped[a.repoId].push(a);
                              }
                              return Object.entries(grouped).map(([repoId, items]) => {
                                const meta = REPO_META[repoId as keyof typeof REPO_META];
                                return (
                                  <div key={repoId} className="project-art-group">
                                    <div className="project-art-header">
                                      {meta?.icon || '📦'} {meta?.name || repoId}
                                      <span className="project-art-count">{items.length}</span>
                                    </div>
                                    <div className="project-art-files">
                                      {items.map(a => (
                                        <span key={a.id} className={`project-art-file status-${a.status}`} title={`${a.filename} (${a.size})`}>
                                          {a.filename}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                );
                              });
                            })()}
                          </div>
                        )}

                        {/* Actions */}
                        <div className="project-detail-actions">
                          {proj.status === 'interrupted' && connected && (
                            <button className="project-resume-btn" onClick={(e) => { e.stopPropagation(); onResume(proj.projectId); }}>
                              ▶ 断点续跑
                            </button>
                          )}
                          <button className="project-select-btn" onClick={(e) => { e.stopPropagation(); onSelect(proj.projectId); }}>
                            {isSelected ? '✓ 已聚焦' : '🔍 聚焦'}
                          </button>
                          <button
                            className="project-delete-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              if (window.confirm(`确认删除项目 "${proj.projectId}"？\n此操作将删除所有阶段数据，不可恢复。`)) {
                                onDelete(proj.projectId);
                              }
                            }}
                          >
                            🗑 删除
                          </button>
                        </div>
                      </div>
                    )}
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
