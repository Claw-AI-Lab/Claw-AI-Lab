import { memo, useState } from 'react';
import { LAYER_META, STAGE_META } from '../types';
import type { LobsterAgent, LogEntry, AgentLayer } from '../types';

interface Props {
  layer: AgentLayer;
  agents: LobsterAgent[];
  logs: LogEntry[];
  tierIndex: number;
  selectedProjectId?: string | null;
}

const STATUS_LABEL: Record<string, string> = {
  idle: '空闲', working: '工作中', error: '异常', done: '完成',
  waiting_discussion: '等待讨论', discussing: '讨论中',
};
const STAGE_ST: Record<string, string> = {
  pending: '⬜', running: '🔄', completed: '✅', failed: '❌', skipped: '⏭',
  waiting: '⏳', discussing: '💬',
};
const DISCUSSION_STAGE = 100;

export default memo(function LayerPanel({ layer, agents, logs, tierIndex, selectedProjectId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const meta = LAYER_META[layer];
  const recentLogs = logs.slice(-30);
  const widthPercent = 50 + tierIndex * 14;
  const workingCount = agents.filter((a) => ['working', 'waiting_discussion', 'discussing'].includes(a.status)).length;

  return (
    <div
      className="layer-panel"
      style={{ '--layer-color': meta.color, width: `${widthPercent}%`, maxWidth: '100%' } as React.CSSProperties}
    >
      <div className="layer-header" onClick={() => setExpanded(!expanded)}>
        <div className="layer-title-row">
          <span className="layer-dot" />
          <h3>{meta.name}</h3>
          <span className="layer-agent-count">
            {agents.length} 🦞 · {workingCount} 活跃
          </span>
          <span className={`expand-icon ${expanded ? 'open' : ''}`}>▼</span>
        </div>
        <div className="layer-stages-bar">
          {meta.stages.map((s) => {
            const sm = STAGE_META[s];
            if (!sm) return null;
            const isDisc = s === DISCUSSION_STAGE;
            const anyRunning = isDisc
              ? agents.some((a) => a.status === 'waiting_discussion' || a.status === 'discussing')
              : agents.some((a) => a.currentStage === s);
            const anyDone = isDisc
              ? agents.some((a) => a.stageProgress[DISCUSSION_STAGE] === 'completed')
              : agents.some((a) => a.stageProgress[s] === 'completed');
            const cls = anyRunning ? 'stage-running' : anyDone ? 'stage-done' : 'stage-idle';
            const dn = sm.displayNumber;
            const label = isDisc ? `💬 ${sm.name}` : `S${dn} ${sm.name.replace(/ ⛩$/, '')}`;
            return (
              <span key={s} className={`stage-chip ${cls}${isDisc ? ' stage-discussion' : ''}`} title={sm.key}>
                {label}
              </span>
            );
          })}
        </div>
      </div>

      <div className="agent-row">
        {agents.map((agent) => {
          const dimmed = selectedProjectId && agent.projectId && agent.projectId !== selectedProjectId;
          const highlighted = selectedProjectId && agent.projectId === selectedProjectId;
          return (
          <div key={agent.id} className={`agent-card status-${agent.status}${highlighted ? ' agent-highlighted' : ''}${dimmed ? ' agent-dimmed' : ''}`}>
            <div className="agent-card-top">
              <span className="agent-name">{agent.name}</span>
              <span className="agent-run-id">{agent.runId}</span>
            </div>
            <div className="agent-status">
              <span className={`status-dot ${agent.status}`} />
              {STATUS_LABEL[agent.status]}
              {agent.currentStage && (
                <span className={`agent-stage-badge${agent.currentStage === DISCUSSION_STAGE ? ' discussion-badge' : ''}`}>
                  {agent.currentStage === DISCUSSION_STAGE
                    ? '💬讨论'
                    : `S${STAGE_META[agent.currentStage]?.displayNumber ?? agent.currentStage}`}
                </span>
              )}
              {!agent.currentStage && (agent.status === 'waiting_discussion' || agent.status === 'discussing') && (
                <span className="agent-stage-badge discussion-badge">💬讨论</span>
              )}
            </div>
            {agent.currentTask && <div className="agent-task">{agent.currentTask}</div>}
            <div className="agent-progress">
              {LAYER_META[layer].stages.map((s) => {
                const isDisc = s === DISCUSSION_STAGE;
                const status = isDisc
                  ? (agent.status === 'discussing' ? 'discussing'
                    : agent.status === 'waiting_discussion' ? 'waiting'
                    : agent.stageProgress[s] || 'pending')
                  : (agent.stageProgress[s] || 'pending');
                const dn2 = STAGE_META[s]?.displayNumber ?? s;
                const label = isDisc ? `💬 沟通讨论: ${status}` : `S${dn2}: ${status}`;
                return (
                  <span key={s} className="stage-pip" title={label}>
                    {STAGE_ST[status] || '⬜'}
                  </span>
                );
              })}
            </div>
          </div>
          );
        })}
      </div>

      {expanded && (
        <div className="layer-logs">
          <div className="log-title">📋 层级日志 ({recentLogs.length})</div>
          <div className="log-list">
            {recentLogs.length === 0 && <div className="log-empty">暂无日志</div>}
            {recentLogs.map((log) => (
              <div key={log.id} className={`log-item level-${log.level}`}>
                <span className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
                <span className="log-agent">{log.agentName.slice(0, 12)}</span>
                {log.stage && <span className={`log-stage${log.stage === DISCUSSION_STAGE ? ' log-stage-discussion' : ''}`}>
                  {log.stage === DISCUSSION_STAGE
                    ? '💬讨论'
                    : `S${STAGE_META[log.stage]?.displayNumber ?? log.stage}`}
                </span>}
                <span className="log-msg">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
