import { memo, useState, useMemo } from 'react';
import type { QueueMap } from '../types';
import { useLocale } from '../i18n';

interface Props {
  queues: QueueMap;
}

const PIPELINE: { key: string; labelKey: string; color: string }[] = [
  { key: 'init_to_idea',          labelKey: 'queue.stage_idea',       color: '#f59e0b' },
  { key: 'idea_to_experiment',    labelKey: 'queue.stage_experiment', color: '#3b82f6' },
  { key: 'experiment_to_coding',  labelKey: 'queue.stage_coding',    color: '#10b981' },
  { key: 'coding_to_execution',   labelKey: 'queue.stage_execution', color: '#ef4444' },
  { key: 'execution_to_writing',  labelKey: 'queue.stage_writing',   color: '#a855f7' },
];

export default memo(function QueuePanel({ queues }: Props) {
  const [expanded, setExpanded] = useState(true);
  const { t } = useLocale();
  const entries = useMemo(
    () => PIPELINE.map((p) => ({ ...p, q: queues[p.key] })).filter((p) => p.q),
    [queues],
  );

  const totals = useMemo(() => {
    let pending = 0, assigned = 0, completed = 0;
    for (const e of entries) {
      pending += e.q.pending || 0;
      assigned += e.q.assigned || 0;
      completed += e.q.completed || 0;
    }
    const total = pending + assigned + completed;
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
    return { pending, assigned, completed, total, pct };
  }, [entries]);

  return (
    <div className="queue-panel">
      <div className="queue-header" onClick={() => setExpanded(!expanded)}>
        <span className="queue-title">{t('queue.title')}</span>

        <div className="queue-header-stats">
          {totals.total > 0 ? (
            <>
              <div className="queue-header-bar">
                {totals.completed > 0 && <div className="qh-seg done" style={{ flex: totals.completed }} />}
                {totals.assigned > 0 && <div className="qh-seg running" style={{ flex: totals.assigned }} />}
                {totals.pending > 0 && <div className="qh-seg waiting" style={{ flex: totals.pending }} />}
              </div>
              <span className="queue-header-pct">{totals.pct}%</span>
            </>
          ) : (
            <span className="qhn idle">{t('queue.idle')}</span>
          )}
        </div>

        <span className={`queue-expand ${expanded ? 'open' : ''}`}>▸</span>
      </div>

      {expanded && entries.length > 0 && (
        <div className="queue-pipeline">
          {entries.map(({ key, labelKey, color, q }) => {
            const total = q.total || 0;
            const done = q.completed || 0;
            const running = q.assigned || 0;
            const pct = total > 0 ? Math.round((done / total) * 100) : 0;
            const active = q.pending > 0 || running > 0;
            return (
              <div key={key} className={`qp-row${active ? ' active' : ''}${done === total && total > 0 ? ' completed' : ''}`}>
                <span className="qp-label" style={{ color: active ? color : undefined }}>{t(labelKey)}</span>
                <div className="qp-bar-area">
                  <div className="qp-bar-track">
                    {total > 0 && (
                      <>
                        {done > 0 && (
                          <div className="qp-bar-fill done" style={{ width: `${(done / total) * 100}%`, background: color }} />
                        )}
                        {running > 0 && (
                          <div
                            className={`qp-bar-fill running${active ? ' pulse' : ''}`}
                            style={{ width: `${(running / total) * 100}%`, background: color, opacity: 0.55 }}
                          />
                        )}
                      </>
                    )}
                  </div>
                </div>
                <span className="qp-progress">{done}/{total}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});
