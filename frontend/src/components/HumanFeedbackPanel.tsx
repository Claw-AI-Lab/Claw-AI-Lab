import { memo, useEffect, useRef, useState } from 'react';
import { AgentLayer, ALL_LAYERS, LAYER_META } from '../types';
import type { ChatMessage } from '../types';
import { useLocale } from '../i18n';

interface Props {
  messages: ChatMessage[];
  onSend: (content: string, targetLayer?: string) => void;
  connected: boolean;
}

export default memo(function HumanFeedbackPanel({ messages, onSend, connected }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [input, setInput] = useState('');
  const [target, setTarget] = useState('all');
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const { t } = useLocale();

  const targetOptions: Array<{ value: string; label: string }> = [
    { value: 'all', label: t('feedback.target_all') },
    ...ALL_LAYERS.map((l) => ({
      value: l,
      label: t(`layer.${l}.name`).split('·')[1]?.trim() || l,
    })),
  ];

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

  const unreadCount = messages.filter((m) => m.role === 'system').length;

  return (
    <div className={`feedback-panel ${expanded ? 'expanded' : ''}`}>
      <div className="feedback-header" onClick={() => setExpanded(!expanded)}>
        <div className="feedback-header-left">
          <span className="feedback-icon">💬</span>
          <span className="feedback-title">{t('feedback.title')}</span>
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
            ? <span className="feedback-conn on">{t('feedback.connected')}</span>
            : <span className="feedback-conn off">{t('feedback.offline')}</span>
          }
          <span className={`feedback-toggle ${expanded ? 'open' : ''}`}>▲</span>
        </div>
      </div>

      {expanded && (
        <div className="feedback-body">
          <div className="feedback-messages" ref={listRef}>
            {messages.length === 0 && (
              <div className="feedback-empty">
                {t('feedback.empty')}
              </div>
            )}
            {messages.map((msg) => (
              <div key={msg.id} className={`feedback-msg ${msg.role === 'user' ? 'human' : 'system'}`}>
                <div className="feedback-msg-header">
                  <span className="feedback-sender">
                    {msg.role === 'user' ? t('feedback.you') : t('feedback.system')}
                  </span>
                  {msg.targetLayer && msg.targetLayer !== 'all' && (
                    <span
                      className="feedback-target-tag"
                      style={{ color: LAYER_META[msg.targetLayer as AgentLayer]?.color }}
                    >
                      @{t(`layer.${msg.targetLayer}.name`).split('·')[1]?.trim()}
                    </span>
                  )}
                  <span className="feedback-time">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="feedback-msg-content">{msg.content}</div>
              </div>
            ))}
          </div>

          <div className="feedback-input-area">
            <select
              className="feedback-target-select"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            >
              {targetOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <textarea
              ref={inputRef}
              className="feedback-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('feedback.placeholder')}
              rows={1}
            />
            <button
              className="feedback-send-btn"
              onClick={handleSend}
              disabled={!input.trim()}
            >
              {t('feedback.send')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
});
