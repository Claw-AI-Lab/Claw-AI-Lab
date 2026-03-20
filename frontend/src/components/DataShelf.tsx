import { memo, useState } from 'react';
import { REPO_META, LAYER_META } from '../types';
import type { Artifact, RepoId } from '../types';

interface Props {
  repoId: RepoId;
  artifacts: Artifact[];
}

function ProjectFolder({ pid, files }: { pid: string; files: Artifact[] }) {
  const [open, setOpen] = useState(false);
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
          {files.map((a) => (
            <div key={a.id} className={`artifact-row status-${a.status}`}>
              <span className="artifact-icon">{a.filename.endsWith('/') ? '📁' : '📄'}</span>
              <span className="artifact-name">{a.filename}</span>
              <span className="artifact-size">{a.size}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default memo(function DataShelf({ repoId, artifacts }: Props) {
  const [open, setOpen] = useState(false);
  const meta = REPO_META[repoId];
  const fromColor = LAYER_META[meta.fromLayer].color;

  const byProject = new Map<string, Artifact[]>();
  for (const a of artifacts) {
    const pid = a.projectId || '未知项目';
    if (!byProject.has(pid)) byProject.set(pid, []);
    byProject.get(pid)!.push(a);
  }

  return (
    <div
      className="data-shelf"
      style={{ '--shelf-color': fromColor } as React.CSSProperties}
    >
      <div className="shelf-header" onClick={() => setOpen(!open)}>
        <span className="shelf-icon">{meta.icon}</span>
        <span className="shelf-name">{meta.name}</span>
        <span className="shelf-count">{artifacts.length}</span>
      </div>
      {open && (
        <div className="shelf-body">
          <div className="shelf-flow">
            <span style={{ color: fromColor }}>{LAYER_META[meta.fromLayer].name.split('·')[1]?.trim()}</span>
            <span className="shelf-arrow-h">→</span>
            <span style={{ color: meta.toLayer ? LAYER_META[meta.toLayer].color : '#6366f1' }}>
              {meta.toLayer ? LAYER_META[meta.toLayer].name.split('·')[1]?.trim() : '反馈 L1'}
            </span>
          </div>
          {byProject.size === 0 && <div className="shelf-empty">暂无产物</div>}
          {[...byProject.entries()].map(([pid, files]) => (
            <ProjectFolder key={pid} pid={pid} files={files} />
          ))}
        </div>
      )}
    </div>
  );
});
