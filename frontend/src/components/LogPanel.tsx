import { useEffect, useRef, useState } from 'react';
import { LAYER_META, ALL_LAYERS, STAGE_META } from '../types';
import type { LogEntry, AgentLayer } from '../types';
import { useLocale } from '../i18n';

function getLayerForStage(stage: number | null | undefined): AgentLayer | null {
  if (!stage) return null;
  if (stage >= 1 && stage <= 8) return 'idea';
  if (stage === 100) return 'idea';
  if (stage === 9) return 'experiment';
  if (stage >= 10 && stage <= 13) return 'coding';
  if (stage >= 14 && stage <= 18) return 'execution';
  if (stage >= 19 && stage <= 22) return 'writing';
  return null;
}

function effectiveLayer(log: LogEntry): AgentLayer {
  return getLayerForStage(log.stage) ?? log.layer;
}

interface Props {
  logs: LogEntry[];
}

export default function LogPanel({ logs }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState<AgentLayer | 'all'>('all');
  const autoScroll = useRef(true);
  const { t } = useLocale();
  const discussionLabel = `💬${t('stage.100')}`;

  const tail = logs.slice(-120);

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
  }, [tail.length]);

  return (
    <div className="log-panel-inner">
      <h2>{t('log.title')} <span className="count-badge">{logs.length}</span></h2>
      <div className="log-filters">
        <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>{t('log.all')}</button>
        {ALL_LAYERS.map((l) => (
          <button
            key={l}
            className={filter === l ? 'active' : ''}
            onClick={() => setFilter(l)}
            style={{ '--btn-color': LAYER_META[l].color } as React.CSSProperties}
          >
            {t(`layer.${l}.name`).split('·')[1]?.trim() || l}
          </button>
        ))}
      </div>
      <div className="global-log-list" ref={listRef} onScroll={handleScroll}>
        {tail.map((log) => {
          const layer = effectiveLayer(log);
          const hidden = filter !== 'all' && layer !== filter;
          return (
            <div
              key={log.id}
              className={`glog-item level-${log.level}`}
              style={hidden ? { display: 'none' } : undefined}
            >
              <span className="glog-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
              <span className="glog-layer" style={{ color: LAYER_META[layer].color }}>
                [{t(`layer.${layer}.name`).split('·')[0].trim()}]
              </span>
              {log.stage && (
                <span className={`glog-stage${log.stage === 100 ? ' glog-stage-discussion' : ''}`} title={STAGE_META[log.stage]?.key}>
                  {log.stage === 100
                    ? discussionLabel
                    : `S${STAGE_META[log.stage]?.displayNumber ?? log.stage}`}
                </span>
              )}
              <span className="glog-msg">{log.message}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
