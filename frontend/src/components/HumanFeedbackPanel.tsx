import { memo, useEffect, useRef, useState } from 'react';
import { AgentLayer, ALL_LAYERS, LAYER_META } from '../types';
import type { ChatMessage } from '../types';

interface Props {
  messages: ChatMessage[];
  onSend: (content: string, targetLayer?: string) => void;
  connected: boolean;
}

const TARGET_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'all', label: '全局' },
  ...ALL_LAYERS.map((l) => ({
    value: l,
    label: LAYER_META[l].name.split('·')[1]?.trim() || l,
  })),
];

export default memo(function HumanFeedbackPanel({ messages, onSend, connected }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [input, setInput] = useState('');
  const [target, setTarget] = useState('all');
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (listRef.current && expanded) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages.length, expanded]);

  useEffect(() => {
    if (expanded && inputRef.current) {
      inputRef.current.focus();
    }
  }, [expanded]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    onSend(trimmed, target);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const unreadCount = messages.filter((m) => m.sender === 'system').length;

  return (
    <div className={`feedback-panel ${expanded ? 'expanded' : ''}`}>
      <div className="feedback-header" onClick={() => setExpanded(!expanded)}>
        <div className="feedback-header-left">
          <span className="feedback-icon">💬</span>
          <span className="feedback-title">人工反馈</span>
          {messages.length > 0 && (
            <span className="feedback-badge">{messages.length}</span>
          )}
          {!expanded && unreadCount > 0 && (
            <span className="feedback-hint">
              {messages[messages.length - 1]?.content.slice(0, 40)}...
            </span>
          )}
        </div>
        <div className="feedback-header-right">
          {connected
            ? <span className="feedback-conn on">已连接</span>
            : <span className="feedback-conn off">离线</span>
          }
          <span className={`feedback-toggle ${expanded ? 'open' : ''}`}>▲</span>
        </div>
      </div>

      {expanded && (
        <div className="feedback-body">
          <div className="feedback-messages" ref={listRef}>
            {messages.length === 0 && (
              <div className="feedback-empty">
                在 pipeline 运行过程中，你可以在这里提供反馈来调整研究计划。
              </div>
            )}
            {messages.map((msg) => (
              <div key={msg.id} className={`feedback-msg ${msg.sender}`}>
                <div className="feedback-msg-header">
                  <span className="feedback-sender">
                    {msg.sender === 'human' ? '👤 你' : '🤖 系统'}
                  </span>
                  {msg.targetLayer && msg.targetLayer !== 'all' && (
                    <span
                      className="feedback-target-tag"
                      style={{ color: LAYER_META[msg.targetLayer as AgentLayer]?.color }}
                    >
                      @{LAYER_META[msg.targetLayer as AgentLayer]?.name.split('·')[1]?.trim()}
                    </span>
                  )}
                  <span className="feedback-time">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="feedback-msg-content">{msg.content}</div>
                {msg.planUpdate && (
                  <div className="feedback-plan-update">
                    <span className="plan-update-tag">📋 计划更新</span>
                    <span>{msg.planUpdate}</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="feedback-input-area">
            <select
              className="feedback-target-select"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            >
              {TARGET_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <textarea
              ref={inputRef}
              className="feedback-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的反馈或建议... (Enter 发送, Shift+Enter 换行)"
              rows={1}
            />
            <button
              className="feedback-send-btn"
              onClick={handleSend}
              disabled={!input.trim()}
            >
              发送
            </button>
          </div>
        </div>
      )}
    </div>
  );
});
