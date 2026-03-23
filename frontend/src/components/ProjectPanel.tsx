import { useState } from 'react';
import type { ProjectInfo, Artifact } from '../types';
import { STAGE_META, RCStage, REPO_META } from '../types';
import { useLocale } from '../i18n';

const STATUS_ICONS: Record<string, { color: string; icon: string }> = {
  running:     { color: '#22c55e', icon: '▶' },
  queued:      { color: '#f59e0b', icon: '⏳' },
  completed:   { color: '#3b82f6', icon: '✓' },
  interrupted: { color: '#ef4444', icon: '⏸' },
  new:         { color: '#94a3b8', icon: '○' },
};

type SubmitMode = 'lab' | 'reproduce';

function stageName(n: number, t: (k: string) => string): string {
  const key = `stage.${n}`;
  const translated = t(key);
  return translated !== key ? translated : `S${n}`;
}

interface Props {
  projects: ProjectInfo[];
  connected: boolean;
  selectedProjectId: string | null;
  artifactsByProject: Record<string, Artifact[]>;
  discussionMode: boolean;
  onToggleDiscussion: () => void;
  onShowDiscussionInfo: () => void;
  onSelect: (projectId: string) => void;
  onResume: (projectId: string) => void;
  onDelete: (projectId: string) => void;
  onQuickSubmit: (topic: string, mode: SubmitMode, researchAngles: string[], referencePapers: string) => void;
}

export default function ProjectPanel({ projects, connected, selectedProjectId, artifactsByProject, discussionMode, onToggleDiscussion, onShowDiscussionInfo, onSelect, onResume, onDelete, onQuickSubmit }: Props) {
  const [panelOpen, setPanelOpen] = useState(true);
  const [mode, setMode] = useState<SubmitMode>('lab');
  const [topicInput, setTopicInput] = useState('');
  const [anglesInput, setAnglesInput] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [refPapersInput, setRefPapersInput] = useState('');
  const [showRefPapers, setShowRefPapers] = useState(false);
  const { t, locale } = useLocale();

  const modeInfo: Record<SubmitMode, { label: string; icon: string; placeholder: string; desc: string }> = {
    lab: {
      label: t('project.mode.lab'),
      icon: '🔬',
      placeholder: t('project.placeholder.lab'),
      desc: t('project.mode.lab_desc'),
    },
    reproduce: {
      label: t('project.mode.reproduce'),
      icon: '📄',
      placeholder: t('project.placeholder.reproduce'),
      desc: t('project.mode.reproduce_desc'),
    },
  };

  const info = modeInfo[mode];

  const submit = () => {
    const text = topicInput.trim();
    if (!text) return;
    const angles = mode === 'lab'
      ? anglesInput.split(/[,，、;；]/).map(s => s.trim()).filter(Boolean)
      : [];
    onQuickSubmit(text, mode, angles, refPapersInput.trim());
    setTopicInput('');
    setAnglesInput('');
    setRefPapersInput('');
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
    ? (anglesInput.trim() ? t('project.go_lab_n', { n: anglesCount }) : t('project.go_lab_auto'))
    : t('project.go_reproduce');

  const toggleExpand = (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedId(prev => prev === projectId ? null : projectId);
  };

  return (
    <div className="project-panel">
      <div className="project-panel-header" onClick={() => setPanelOpen(!panelOpen)}>
        <h3>
          <span className="panel-icon">📋</span> {t('project.title')}
          {projects.length > 0 && <span className="project-count">{projects.length}</span>}
          {running > 0 && <span className="project-count running-count">{t('project.count_running', { n: running })}</span>}
          {interrupted > 0 && <span className="project-count interrupted-count">{t('project.count_interrupted', { n: interrupted })}</span>}
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
                  {modeInfo[m].icon} {modeInfo[m].label}
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
                <>
                  <div className="project-angles-row">
                    <input
                      className="project-angles-input"
                      placeholder={t('project.angles_placeholder')}
                      value={anglesInput}
                      onChange={e => setAnglesInput(e.target.value)}
                      onKeyDown={onKey}
                      disabled={!connected}
                    />
                  </div>
                  <div className="discussion-mode-row">
                    <button
                      className={`btn-sm discussion-toggle${discussionMode ? ' active' : ''}`}
                      onClick={onToggleDiscussion}
                      type="button"
                    >
                      {discussionMode ? t('discussion.on') : t('discussion.off')}
                    </button>
                    <button
                      className="btn-sm discussion-info-btn"
                      onClick={onShowDiscussionInfo}
                      type="button"
                      title={t('discussion.dialog_title')}
                    >
                      ?
                    </button>
                    <span className="discussion-hint">
                      {discussionMode ? t('discussion.hint_on') : t('discussion.hint_off')}
                    </span>
                  </div>
                </>
              )}
              <div className="project-ref-papers-toggle">
                <button
                  className="ref-papers-toggle-btn"
                  type="button"
                  onClick={() => setShowRefPapers(!showRefPapers)}
                >
                  {showRefPapers ? '▾' : '▸'} {t('project.ref_papers_toggle')}
                  {refPapersInput.trim() && <span className="ref-badge">
                    {refPapersInput.split(/[\n,]/).filter(s => s.trim()).length}
                  </span>}
                </button>
              </div>
              {showRefPapers && (
                <textarea
                  className="project-ref-papers-input"
                  placeholder={t('project.ref_papers_placeholder')}
                  value={refPapersInput}
                  onChange={e => setRefPapersInput(e.target.value)}
                  rows={3}
                  disabled={!connected}
                />
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
              {mode === 'lab' ? t('project.empty_lab') : t('project.empty_reproduce')}
            </div>
          ) : (
            <div className="project-list-inner">
              {sorted.map(proj => {
                const cfg = STATUS_ICONS[proj.status] || STATUS_ICONS.new;
                const statusLabel = t(`project.status.${proj.status}`);
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
                    <div className="project-summary" onClick={(e) => { onSelect(proj.projectId); toggleExpand(proj.projectId, e); }}>
                      <span className="project-status-dot" style={{ background: cfg.color }} title={statusLabel}>{cfg.icon}</span>
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

                    {isExpanded && (
                      <div className="project-detail">
                        <div className="project-detail-meta">
                          <span className="project-id-label" title={proj.projectId}>{proj.projectId}</span>
                          <span className="project-status-badge" style={{ background: cfg.color }}>
                            {cfg.icon} {statusLabel}
                          </span>
                        </div>

                        <div className="project-detail-stage">
                          {proj.lastCompletedStage}/{proj.totalStages} {t('project.stages_label')}
                          {proj.lastCompletedStage > 0 && (
                            <> · {stageName(proj.lastCompletedStage, t)}
                              {proj.lastCompletedStage < proj.totalStages && (
                                <span className="project-next-stage"> → {stageName(proj.lastCompletedStage + 1, t)}</span>
                              )}
                            </>
                          )}
                        </div>

                        {proj.timestamp && (
                          <div className="project-detail-time">
                            {new Date(proj.timestamp).toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US')}
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
                              {t('project.resume')}
                            </button>
                          )}
                          <button
                            className="project-delete-btn"
                            title={t('project.delete_title')}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (window.confirm(t('project.delete_confirm', { id: proj.projectId }))) {
                                onDelete(proj.projectId);
                              }
                            }}
                          >
                            {t('project.delete')}
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
