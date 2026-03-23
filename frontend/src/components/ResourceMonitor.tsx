import { memo } from 'react';
import type { ResourceStats } from '../types';
import { useLocale } from '../i18n';

interface Props {
  stats: ResourceStats | null;
  connected: boolean;
}

function pctColor(v: number) {
  return v > 85 ? '#ef4444' : v > 60 ? '#f59e0b' : '#6ee7b7';
}

export default memo(function ResourceMonitor({ stats, connected }: Props) {
  const { t } = useLocale();

  if (!stats) {
    return (
      <div className="res-bar">
        <span className="res-tag">{t('resource.tag')}</span>
        <span className="res-offline-hint">{connected ? t('resource.waiting') : t('resource.disconnected')}</span>
      </div>
    );
  }

  const memPct = stats.memTotal > 0 ? (stats.memUsed / stats.memTotal * 100) : 0;
  const gpuName = stats.gpus.length > 0 ? stats.gpus[0].name : '';

  return (
    <div className="res-bar">
      <span className="res-tag">📈</span>
      <span className="res-item">
        CPU <b style={{ color: pctColor(stats.cpuPercent) }}>{stats.cpuPercent.toFixed(0)}%</b>
      </span>
      <span className="res-item">
        MEM <b style={{ color: pctColor(memPct) }}>{stats.memUsed.toFixed(0)}</b>/{stats.memTotal.toFixed(0)}G
      </span>
      {gpuName && <span className="res-item res-gpu-label">{stats.gpus.length}×{gpuName}</span>}
      {stats.gpus.map((gpu) => {
        const memP = gpu.memTotal > 0 ? (gpu.memUsed / gpu.memTotal * 100) : 0;
        return (
          <span key={gpu.id} className="res-item res-gpu">
            {gpu.id}:
            <b style={{ color: pctColor(gpu.utilization) }}>{gpu.utilization.toFixed(0)}%</b>
            <span className="res-mem">{gpu.memUsed.toFixed(0)}<span style={{ color: pctColor(memP) }}>/{gpu.memTotal.toFixed(0)}G</span></span>
            <span className="res-temp">{gpu.temperature}°</span>
          </span>
        );
      })}
    </div>
  );
});
