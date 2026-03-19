import { memo, useState } from 'react';
import { LAYER_META, STAGE_META } from '../types';
import type { LobsterAgent, LogEntry, AgentLayer } from '../types';

interface Props {
  layer: AgentLayer;
  agents: LobsterAgent[];
  logs: LogEntry[];
  tierIndex: number;
}

const STATUS_LABEL: Record<string, string> = { idle: '空闲', working: '工作中', error: '异常', done: '完成' };
const STAGE_ST: Record<string, string> = { pending: '⬜', running: '🔄', completed: '✅', failed: '❌', skipped: '⏭' };

export default memo(function LayerPanel({ layer, agents, logs, tierIndex }: Props) {
  const [expanded, setExpanded] = useState(false);
  const meta = LAYER_META[layer];
  const recentLogs = logs.slice(-30);
  const widthPercent = 50 + tierIndex * 14;
  const workingCount = agents.filter((a) => a.status === 'working').length;

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
            const anyRunning = agents.some((a) => a.currentStage === s);
            const anyDone = agents.some((a) => a.stageProgress[s] === 'completed');
            const cls = anyRunning ? 'stage-running' : anyDone ? 'stage-done' : 'stage-idle';
            return (
              <span key={s} className={`stage-chip ${cls}`} title={sm.key}>
                S{s} {sm.name.replace(/ ⛩$/, '')}
              </span>
            );
          })}
        </div>
      </div>

      <div className="agent-row">
        {agents.map((agent) => (
          <div key={agent.id} className={`agent-card status-${agent.status}`}>
            <div className="agent-card-top">
              <span className="agent-name">{agent.name}</span>
              <span className="agent-run-id">{agent.runId}</span>
            </div>
            <div className="agent-status">
              <span className={`status-dot ${agent.status}`} />
              {STATUS_LABEL[agent.status]}
              {agent.currentStage && (
                <span className="agent-stage-badge">S{agent.currentStage}</span>
              )}
            </div>
            {agent.currentTask && <div className="agent-task">{agent.currentTask}</div>}
            <div className="agent-progress">
              {LAYER_META[layer].stages.map((s) => (
                <span key={s} className="stage-pip" title={`S${s}: ${agent.stageProgress[s] || 'pending'}`}>
                  {STAGE_ST[agent.stageProgress[s] || 'pending']}
                </span>
              ))}
            </div>
          </div>
        ))}
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
                {log.stage && <span className="log-stage">S{log.stage}</span>}
                <span className="log-msg">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});
