import { memo, useEffect, useRef, useState } from 'react';
import { LAYER_META, ALL_LAYERS, STAGE_META } from '../types';
import type { LogEntry, AgentLayer } from '../types';

interface Props {
  logs: LogEntry[];
}

export default memo(function LogPanel({ logs }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState<AgentLayer | 'all'>('all');
  const autoScroll = useRef(true);

  const filtered = filter === 'all' ? logs : logs.filter((l) => l.layer === filter);
  const display = filtered.slice(-120);

  const handleScroll = () => {
    const el = listRef.current;
    if (!el) return;
    autoScroll.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  };

  useEffect(() => {
    const el = listRef.current;
    if (el && autoScroll.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [display.length]);

  return (
    <div className="side-panel log-panel">
      <h2>📊 事件日志 <span className="count-badge">{logs.length}</span></h2>
      <div className="log-filters">
        <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>全部</button>
        {ALL_LAYERS.map((l) => (
          <button
            key={l}
            className={filter === l ? 'active' : ''}
            onClick={() => setFilter(l)}
            style={{ '--btn-color': LAYER_META[l].color } as React.CSSProperties}
          >
            {LAYER_META[l].name.split('·')[1]?.trim() || l}
          </button>
        ))}
      </div>
      <div className="global-log-list" ref={listRef} onScroll={handleScroll}>
        {display.map((log) => (
          <div key={log.id} className={`glog-item level-${log.level}`}>
            <span className="glog-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span className="glog-layer" style={{ color: LAYER_META[log.layer].color }}>
              [{LAYER_META[log.layer].name.split('·')[0].trim()}]
            </span>
            {log.stage && (
              <span className={`glog-stage${log.stage === 100 ? ' glog-stage-discussion' : ''}`} title={STAGE_META[log.stage]?.key}>
                {log.stage === 100
                  ? '💬讨论'
                  : `S${STAGE_META[log.stage]?.displayNumber ?? log.stage}`}
              </span>
            )}
            <span className="glog-msg">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
});
