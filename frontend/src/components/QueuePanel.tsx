import { memo, useState, useMemo } from 'react';
import type { QueueMap } from '../types';
import { useLocale } from '../i18n';

interface Props {
  queues: QueueMap;
}

const PIPELINE: { key: string; from: string; to: string; color: string }[] = [
  { key: 'init_to_idea',          from: 'L0',   to: 'L1', color: '#f59e0b' },
  { key: 'idea_to_experiment',    from: 'L1',   to: 'L2', color: '#3b82f6' },
  { key: 'experiment_to_coding',  from: 'L2',   to: 'L3', color: '#10b981' },
  { key: 'coding_to_execution',   from: 'L3',   to: 'L4', color: '#ef4444' },
  { key: 'execution_to_writing',  from: 'L4',   to: 'L5', color: '#a855f7' },
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
    return { pending, assigned, completed, total: pending + assigned + completed };
  }, [entries]);

  const hasActivity = totals.pending > 0 || totals.assigned > 0;

  return (
    <div className="queue-panel">
      <div className="queue-header" onClick={() => setExpanded(!expanded)}>
        <span className="queue-title">
          {t('queue.title')}
        </span>

        <div className="queue-header-stats">
          {totals.total > 0 && (
            <div className="queue-header-bar">
              {totals.completed > 0 && (
                <div className="qh-seg done" style={{ flex: totals.completed }} />
              )}
              {totals.assigned > 0 && (
                <div className="qh-seg running" style={{ flex: totals.assigned }} />
              )}
              {totals.pending > 0 && (
                <div className="qh-seg waiting" style={{ flex: totals.pending }} />
              )}
            </div>
          )}
          {hasActivity ? (
            <span className="queue-header-nums">
              {totals.assigned > 0 && <span className="qhn running">{totals.assigned}▶</span>}
              {totals.pending > 0 && <span className="qhn waiting">{totals.pending}⏳</span>}
            </span>
          ) : (
            <span className="qhn idle">{t('queue.idle')}</span>
          )}
        </div>

        <span className={`queue-expand ${expanded ? 'open' : ''}`}>▸</span>
      </div>

      {expanded && entries.length > 0 && (
        <div className="queue-pipeline">
          {entries.map(({ key, from, to, color, q }) => {
            const active = q.pending > 0 || q.assigned > 0;
            const pct = q.total > 0 ? Math.round(((q.completed) / q.total) * 100) : 0;
            return (
              <div key={key} className={`qp-row${active ? ' active' : ''}`}>
                <div className="qp-route">
                  <span className="qp-node" style={{ borderColor: color }}>{from}</span>
                  <span className="qp-arrow" style={{ color }}>→</span>
                  <span className="qp-node" style={{ borderColor: color }}>{to}</span>
                </div>

                <div className="qp-bar-area">
                  <div className="qp-bar-track">
                    {q.total > 0 ? (
                      <>
                        {q.completed > 0 && (
                          <div
                            className="qp-bar-fill done"
                            style={{ width: `${(q.completed / q.total) * 100}%`, background: color }}
                          />
                        )}
                        {q.assigned > 0 && (
                          <div
                            className={`qp-bar-fill running${active ? ' pulse' : ''}`}
                            style={{
                              width: `${(q.assigned / q.total) * 100}%`,
                              background: color,
                              opacity: 0.55,
                            }}
                          />
                        )}
                      </>
                    ) : null}
                  </div>
                  {q.total > 0 && <span className="qp-pct">{pct}%</span>}
                </div>

                <div className="qp-counts">
                  {q.assigned > 0 && <span className="qp-badge running" title={t('queue.badge_running')}>{q.assigned}</span>}
                  {q.pending > 0 && <span className="qp-badge waiting" title={t('queue.badge_waiting')}>{q.pending}</span>}
                  {q.pending === 0 && q.assigned === 0 && <span className="qp-badge idle">—</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});
