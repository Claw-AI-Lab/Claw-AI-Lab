import { memo } from 'react';
import type { QueueMap } from '../types';

interface Props {
  queues: QueueMap;
}

const QUEUE_LABELS: Record<string, { icon: string; label: string }> = {
  init_to_idea:          { icon: '🌱', label: '新项目 → 调研' },
  idea_to_experiment:    { icon: '💡', label: '调研 → 实验设计' },
  experiment_to_coding:  { icon: '🧪', label: '实验 → 代码' },
  coding_to_execution:   { icon: '💻', label: '代码 → 执行' },
  execution_feedback:    { icon: '🔄', label: '执行 → 反馈' },
};

export default memo(function QueuePanel({ queues }: Props) {
  const entries = Object.entries(queues);
  if (entries.length === 0) return null;

  return (
    <div className="queue-panel">
      <div className="queue-title">📋 任务队列</div>
      {entries.map(([name, q]) => {
        const meta = QUEUE_LABELS[name] || { icon: '📦', label: name };
        return (
          <div key={name} className="queue-row">
            <span className="queue-icon">{meta.icon}</span>
            <span className="queue-label">{meta.label}</span>
            <span className="queue-counts">
              {q.pending > 0 && <span className="qc pending">{q.pending}待</span>}
              {q.assigned > 0 && <span className="qc assigned">{q.assigned}执行</span>}
              {q.completed > 0 && <span className="qc completed">{q.completed}✓</span>}
              {q.total === 0 && <span className="qc empty">空</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
});
