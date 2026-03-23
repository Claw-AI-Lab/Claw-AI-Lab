import { memo, useState } from 'react';
import type { QueueMap } from '../types';

interface Props {
  queues: QueueMap;
}

const QUEUE_LABELS: Record<string, { label: string; from: string; to: string }> = {
  init_to_idea:          { label: '新项目 → L1', from: 'init', to: 'L1' },
  idea_to_experiment:    { label: 'L1 → L2', from: 'L1', to: 'L2' },
  experiment_to_coding:  { label: 'L2 → L3', from: 'L2', to: 'L3' },
  coding_to_execution:   { label: 'L3 → L4', from: 'L3', to: 'L4' },
  execution_to_writing:  { label: 'L4 → L5', from: 'L4', to: 'L5' },
  execution_feedback:    { label: 'L4 → 反馈', from: 'L4', to: 'L1' },
};

export default memo(function QueuePanel({ queues }: Props) {
  const [expanded, setExpanded] = useState(true);
  const entries = Object.entries(queues);
  const totalPending = entries.reduce((s, [, q]) => s + (q.pending || 0), 0);
  const totalAssigned = entries.reduce((s, [, q]) => s + (q.assigned || 0), 0);

  return (
    <div className="queue-panel">
      <div className="queue-header" onClick={() => setExpanded(!expanded)}>
        <span className="queue-title">📋 任务队列</span>
        <span className="queue-summary">
          {totalPending > 0 && <span className="qc pending">{totalPending}</span>}
          {totalAssigned > 0 && <span className="qc assigned">{totalAssigned}</span>}
          {totalPending === 0 && totalAssigned === 0 && <span className="qc empty">空闲</span>}
        </span>
        <span className={`queue-expand ${expanded ? 'open' : ''}`}>▸</span>
      </div>

      {expanded && entries.length > 0 && (
        <div className="queue-list">
          {entries.map(([name, q]) => {
            const meta = QUEUE_LABELS[name] || { label: name, from: '?', to: '?' };
            const hasTasks = q.pending > 0 || q.assigned > 0;
            return (
              <div key={name} className={`queue-row${hasTasks ? ' active' : ''}`}>
                <span className="queue-label">{meta.label}</span>
                <div className="queue-bar-wrap">
                  {q.total > 0 ? (
                    <div className="queue-bar">
                      {q.completed > 0 && (
                        <div className="queue-bar-seg completed" style={{ flex: q.completed }} title={`${q.completed} 完成`} />
                      )}
                      {q.assigned > 0 && (
                        <div className="queue-bar-seg assigned" style={{ flex: q.assigned }} title={`${q.assigned} 执行中`} />
                      )}
                      {q.pending > 0 && (
                        <div className="queue-bar-seg pending" style={{ flex: q.pending }} title={`${q.pending} 等待`} />
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
