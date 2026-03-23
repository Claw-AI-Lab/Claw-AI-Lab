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

  const targetOptions: Array<{ value: string; label: string; color?: string }> = [
    { value: 'all', label: t('feedback.target_all') },
    ...ALL_LAYERS.map((l) => ({
      value: l,
      label: t(`layer.${l}.name`).split('·')[1]?.trim() || l,
      color: LAYER_META[l]?.color,
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

  const lastSystemMsg = messages.filter((m) => m.role === 'system').pop();

  return (
    <div className={`fb ${expanded ? 'fb--open' : ''}`}>
      {/* accent stripe */}
      <div className="fb-stripe" />

      <div className="fb-bar" onClick={() => setExpanded(!expanded)}>
        <div className="fb-bar-left">
          <span className="fb-icon">💬</span>
          <span className="fb-label">{t('feedback.title')}</span>
          {messages.length > 0 && <span className="fb-count">{messages.length}</span>}
        </div>

        {!expanded && lastSystemMsg && (
          <span className="fb-preview">{lastSystemMsg.content.slice(0, 50)}</span>
        )}

        <div className="fb-bar-right">
          <span className={`fb-dot ${connected ? 'on' : 'off'}`} />
          <span className={`fb-arrow ${expanded ? 'up' : ''}`}>▾</span>
        </div>
      </div>

      {expanded && (
        <div className="fb-body">
          <div className="fb-msgs" ref={listRef}>
            {messages.length === 0 && (
              <div className="fb-empty">
                <span className="fb-empty-icon">🤖</span>
                <span>{t('feedback.empty')}</span>
              </div>
            )}
            {messages.map((msg) => (
              <div key={msg.id} className={`fb-msg ${msg.role === 'user' ? 'me' : 'sys'}`}>
                <div className="fb-msg-top">
                  <span className="fb-sender">
                    {msg.role === 'user' ? t('feedback.you') : t('feedback.system')}
                  </span>
                  {msg.targetLayer && msg.targetLayer !== 'all' && (
                    <span
                      className="fb-tag"
                      style={{ color: LAYER_META[msg.targetLayer as AgentLayer]?.color }}
                    >
                      @{t(`layer.${msg.targetLayer}.name`).split('·')[1]?.trim()}
                    </span>
                  )}
                  <span className="fb-ts">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="fb-text">{msg.content}</div>
              </div>
            ))}
          </div>

          <div className="fb-input">
            <select
              className="fb-target"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            >
              {targetOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <textarea
              ref={inputRef}
              className="fb-text-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('feedback.placeholder')}
              rows={1}
            />
            <button
              className="fb-send"
              onClick={handleSend}
              disabled={!input.trim()}
              title={t('feedback.send')}
            >
              ↑
            </button>
          </div>
        </div>
      )}
    </div>
  );
});
