import { memo, useState, useMemo } from 'react';
import { REPO_META, LAYER_META, ARTIFACT_LABELS, STAGE_META } from '../types';
import type { Artifact, RepoId, RCStage } from '../types';
import { useLocale } from '../i18n';

interface Props {
  repoId: RepoId;
  artifacts: Artifact[];
}

function getArtifactLabel(filename: string, locale: string): { icon: string; label: string } {
  const info = ARTIFACT_LABELS[filename];
  if (info) return { icon: info.icon, label: locale === 'zh' ? info.zh : info.en };
  if (filename.endsWith('/')) return { icon: '📁', label: filename.replace(/\/$/, '') };
  if (filename.endsWith('.md')) return { icon: '📄', label: filename.replace('.md', '') };
  if (filename.endsWith('.json') || filename.endsWith('.jsonl')) return { icon: '📋', label: filename };
  if (filename.endsWith('.yaml') || filename.endsWith('.yml')) return { icon: '⚙️', label: filename };
  return { icon: '📄', label: filename };
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '…' : s;
}

function ArtifactItem({ a, locale }: { a: Artifact; locale: string }) {
  const [expanded, setExpanded] = useState(false);
  const hasContent = !!a.content;
  const { icon, label } = getArtifactLabel(a.filename, locale);
  const isDownloadable = a.filename.endsWith('.zip');

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    const url = `/download/${a.projectId}/${a.filename}`;
    window.open(url, '_blank');
  };

  return (
    <div className={`artifact-card status-${a.status}`}>
      <div
        className={`artifact-card-header ${hasContent ? 'clickable' : ''}`}
        onClick={() => hasContent && setExpanded(!expanded)}
      >
        <span className="artifact-card-icon">{icon}</span>
        <span className="artifact-card-label">{label}</span>
        <span className="artifact-card-size">{a.size}</span>
        {isDownloadable && (
          <button className="artifact-download-btn" onClick={handleDownload} title={locale === 'zh' ? '下载' : 'Download'}>
            ⬇
          </button>
        )}
        {hasContent && <span className={`artifact-card-chevron ${expanded ? 'open' : ''}`}>▶</span>}
      </div>
      {hasContent && !expanded && (
        <div className="artifact-card-summary">{truncate(a.content!, 200)}</div>
      )}
      {expanded && a.content && (
        <div className="artifact-card-detail">
          <pre>{a.content}</pre>
        </div>
      )}
    </div>
  );
}

interface StageGroup {
  stage: number;
  stageName: string;
  artifacts: Artifact[];
}

function groupByStage(files: Artifact[], locale: string): StageGroup[] {
  const stageMap = new Map<number, Artifact[]>();
  const noStage: Artifact[] = [];

  for (const a of files) {
    const s = a.stage ?? 0;
    if (s > 0) {
      if (!stageMap.has(s)) stageMap.set(s, []);
      stageMap.get(s)!.push(a);
    } else {
      noStage.push(a);
    }
  }

  const groups: StageGroup[] = [];
  const sortedStages = [...stageMap.keys()].sort((a, b) => a - b);
  for (const s of sortedStages) {
    const meta = STAGE_META[s as RCStage];
    const stageName = meta
      ? `S${meta.displayNumber || s} ${locale === 'zh' ? meta.name : meta.key.replace(/_/g, ' ').toLowerCase()}`
      : `S${s}`;
    groups.push({ stage: s, stageName, artifacts: stageMap.get(s)! });
  }
  if (noStage.length > 0) {
    groups.push({ stage: 0, stageName: locale === 'zh' ? '其他' : 'Other', artifacts: noStage });
  }
  return groups;
}

function ProjectFolder({ pid, files, locale }: { pid: string; files: Artifact[]; locale: string }) {
  const [open, setOpen] = useState(true);
  const stageGroups = useMemo(() => groupByStage(files, locale), [files, locale]);

  return (
    <div className="shelf-project">
      <div className="shelf-project-name" onClick={() => setOpen(!open)}>
        <span className="folder-icon">{open ? '📂' : '📁'}</span>
        <span className="folder-name">{pid}</span>
        <span className="shelf-count">{files.length}</span>
        <span className={`folder-arrow ${open ? 'open' : ''}`}>▶</span>
      </div>
      {open && (
        <div className="shelf-artifacts">
          {stageGroups.map((g) => (
            <div key={g.stage} className="stage-group">
              <div className="stage-group-label">{g.stageName}</div>
              {g.artifacts.map((a) => (
                <ArtifactItem key={a.id} a={a} locale={locale} />
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default memo(function DataShelf({ repoId, artifacts }: Props) {
  const [open, setOpen] = useState(false);
  const { t, locale } = useLocale();
  const meta = REPO_META[repoId];
  const fromColor = LAYER_META[meta.fromLayer].color;
  const repoName = t(`repo.${repoId}.name`);
  const repoDesc = t(`repo.${repoId}.desc`);

  const byProject = useMemo(() => {
    const map = new Map<string, Artifact[]>();
    for (const a of artifacts) {
      const pid = a.projectId || t('shelf.unknown_project');
      if (!map.has(pid)) map.set(pid, []);
      map.get(pid)!.push(a);
    }
    return map;
  }, [artifacts, t]);

  return (
    <div
      className="data-shelf"
      style={{ '--shelf-color': fromColor } as React.CSSProperties}
    >
      <div className="shelf-header" onClick={() => setOpen(!open)}>
        <span className="shelf-icon">{meta.icon}</span>
        <span className="shelf-name">{repoName}</span>
        {artifacts.length > 0 && <span className="shelf-count">{artifacts.length}</span>}
        <span className={`shelf-toggle ${open ? 'open' : ''}`}>▶</span>
      </div>
      {!open && artifacts.length > 0 && (
        <div className="shelf-preview">{repoDesc}</div>
      )}
      {open && (
        <div className="shelf-body">
          <div className="shelf-flow">
            <span style={{ color: fromColor }}>{t(`layer.${meta.fromLayer}.name`).split('·')[1]?.trim()}</span>
            <span className="shelf-arrow-h">→</span>
            <span style={{ color: meta.toLayer ? LAYER_META[meta.toLayer].color : '#6366f1' }}>
              {meta.toLayer ? t(`layer.${meta.toLayer}.name`).split('·')[1]?.trim() : t('shelf.feedback_l1')}
            </span>
          </div>
          {byProject.size === 0 && <div className="shelf-empty">{t('shelf.no_artifacts')}</div>}
          {repoId === 'papers'
            ? [...byProject.entries()].map(([pid, files]) => (
                <ProjectFolder key={pid} pid={pid} files={files} locale={locale} />
              ))
            : [...byProject.entries()].map(([pid, files]) => (
                <ProjectFolder key={pid} pid={pid} files={files} locale={locale} />
              ))
          }
        </div>
      )}
    </div>
  );
});
