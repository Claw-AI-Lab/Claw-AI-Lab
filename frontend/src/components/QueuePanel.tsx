import { memo, useState } from 'react';
import type { QueueMap } from '../types';
import { useLocale } from '../i18n';

interface Props {
  queues: QueueMap;
}

const QUEUE_ICONS: Record<string, string> = {
  init_to_idea: '🌱',
  idea_to_experiment: '💡',
  experiment_to_coding: '🧪',
  coding_to_execution: '💻',
  execution_to_writing: '📝',
  execution_feedback: '🔄',
};

export default memo(function QueuePanel({ queues }: Props) {
  const [expanded, setExpanded] = useState(true);
  const entries = Object.entries(queues);
  const { t } = useLocale();
  const totalPending = entries.reduce((s, [, q]) => s + (q.pending || 0), 0);
  const totalAssigned = entries.reduce((s, [, q]) => s + (q.assigned || 0), 0);

  return (
    <div className="queue-panel">
      <div className="queue-header" onClick={() => setExpanded(!expanded)}>
        <span className="queue-title">{t('queue.title')}</span>
        <span className="queue-summary">
          {totalPending > 0 && <span className="qc pending">{totalPending}</span>}
          {totalAssigned > 0 && <span className="qc assigned">{totalAssigned}</span>}
          {totalPending === 0 && totalAssigned === 0 && <span className="qc empty">{t('queue.empty')}</span>}
        </span>
        <span className={`queue-expand ${expanded ? 'open' : ''}`}>▸</span>
      </div>

      {expanded && entries.length > 0 && (
        <div className="queue-list">
          {entries.map(([name, q]) => {
            const icon = QUEUE_ICONS[name] || '📦';
            const label = t(`queue.${name}`);
            const displayLabel = label !== `queue.${name}` ? label : name;
            const hasTasks = q.pending > 0 || q.assigned > 0;
            return (
              <div key={name} className={`queue-row${hasTasks ? ' active' : ''}`}>
                <span className="queue-icon">{icon}</span>
                <span className="queue-label">{displayLabel}</span>
                <div className="queue-bar-wrap">
                  {q.total > 0 ? (
                    <div className="queue-bar">
                      {q.completed > 0 && (
                        <div className="queue-bar-seg completed" style={{ flex: q.completed }} title={t('queue.completed', { n: q.completed })} />
                      )}
                      {q.assigned > 0 && (
                        <div className="queue-bar-seg assigned" style={{ flex: q.assigned }} title={t('queue.assigned', { n: q.assigned })} />
                      )}
                      {q.pending > 0 && (
                        <div className="queue-bar-seg pending" style={{ flex: q.pending }} title={t('queue.pending', { n: q.pending })} />
                      )}
                    </div>
                  ) : (
                    <div className="queue-bar"><div className="queue-bar-seg empty" style={{ flex: 1 }} /></div>
                  )}
                </div>
                <span className="queue-num">{q.pending > 0 ? q.pending : '—'}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});
