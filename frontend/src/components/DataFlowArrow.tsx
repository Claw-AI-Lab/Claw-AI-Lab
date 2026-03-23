import { memo } from 'react';

interface Props {
  active: boolean;
  direction?: 'down' | 'up';
}

export default memo(function DataFlowArrow({ active, direction = 'down' }: Props) {
  return (
    <div className={`flow-arrow ${active ? 'active' : ''} flow-${direction}`}>
      <div className="flow-line">
        <div className="flow-pulse" />
      </div>
      <div className="flow-tip" />
    </div>
  );
});
