import { useReducer, useEffect, useCallback, useRef, useState, useMemo } from 'react';
import { AgentLayer, ALL_LAYERS, ALL_REPOS } from './types';
import type { AppState, WSMessage, ResourceStats, LobsterAgent, Artifact } from './types';
import { INITIAL_AGENTS, createMockMessageGenerator } from './mock';
import LayerPanel from './components/LayerPanel';
import ResourceMonitor from './components/ResourceMonitor';
import LogPanel from './components/LogPanel';
import HumanFeedbackPanel from './components/HumanFeedbackPanel';
import DataFlowArrow from './components/DataFlowArrow';
import DataShelf from './components/DataShelf';
import ProjectPanel from './components/ProjectPanel';
import { LocaleContext, makeT } from './i18n';
import type { Locale } from './i18n';
import './App.css';

type ReferencePdfUpload = { name: string; contentBase64: string };

type Action =
  | WSMessage
  | { type: 'set_connected'; payload: boolean }
  | { type: 'set_mock'; payload: boolean }
  | { type: 'set_res_connected'; payload: boolean }
  | { type: 'clear_agents' }
  | { type: 'select_project'; payload: string | null };

function upsertAgent(agents: LobsterAgent[], payload: LobsterAgent): LobsterAgent[] {
  const idx = agents.findIndex((a) => a.id === payload.id);
  if (idx >= 0) {
    const prev = agents[idx];
    if (
      prev.status === payload.status &&
      prev.currentStage === payload.currentStage &&
      prev.currentTask === payload.currentTask
    ) {
      return agents;
    }
    const updated = [...agents];
    updated[idx] = { ...updated[idx], ...payload };
    return updated;
  }
  return [...agents, payload];
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'agent_update':
      return { ...state, agents: upsertAgent(state.agents, action.payload) };
    case 'stage_update':
      return {
        ...state,
        agents: state.agents.map((a) => {
          if (a.id !== action.payload.agentId) return a;
          return {
            ...a,
            stageProgress: { ...a.stageProgress, [action.payload.stage]: action.payload.status },
          };
        }),
      };
    case 'artifact_produced': {
      const a = action.payload;
      const key = `${a.projectId}:${a.stage ?? 0}:${a.filename}`;
      const exists = state.artifacts.some(
        (x) => `${x.projectId}:${x.stage ?? 0}:${x.filename}` === key,
      );
      if (exists) return state;
      return { ...state, artifacts: [a, ...state.artifacts] };
    }
    case 'resource_stats':
      return { ...state, resources: action.payload, resConnected: true };
    case 'log':
      return { ...state, logs: [...state.logs, action.payload] };
    case 'chat_message':
      return { ...state, chatMessages: [...state.chatMessages, action.payload] };
    case 'set_connected':
      return { ...state, connected: action.payload };
    case 'set_res_connected':
      return { ...state, resConnected: action.payload };
    case 'queue_update': {
      const prev = JSON.stringify(state.queues);
      const next = JSON.stringify(action.payload);
      return prev === next ? state : { ...state, queues: action.payload };
    }
    case 'project_list':
      return { ...state, projects: action.payload };
    case 'select_project':
      return { ...state, selectedProjectId: action.payload };
    case 'set_mock':
      return { ...state, mockMode: action.payload };
    case 'clear_agents':
      return { ...state, agents: [], artifacts: [], logs: [], queues: {} };
    case 'system':
      return state;
    default:
      return state;
  }
}

const WS_PROTO = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

const INITIAL_STATE: AppState = {
  agents: [],
  artifacts: [],
  logs: [],
  queues: {},
  chatMessages: [],
  projects: [],
  selectedProjectId: null,
  resources: null,
  resConnected: false,
  connected: false,
  mockMode: false,
};

export default function App() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const [agentWsUrl] = useState(`${WS_PROTO}//${window.location.host}/ws/agents`);
  const [resWsUrl] = useState(`${WS_PROTO}//${window.location.host}/ws/resources`);
  const [discussionMode, setDiscussionMode] = useState(true);
  const [locale, setLocale] = useState<Locale>(() =>
    (localStorage.getItem('claw-locale') as Locale) || 'en'
  );
  const [theme, setTheme] = useState<'dark' | 'light'>(() =>
    (localStorage.getItem('claw-theme') as 'dark' | 'light') || 'dark'
  );
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('claw-theme', theme);
  }, [theme]);
  const t = useMemo(() => makeT(locale), [locale]);
  const localeCtx = useMemo(() => ({
    locale, setLocale: (l: Locale) => { setLocale(l); localStorage.setItem('claw-locale', l); }, t,
  }), [locale, t]);
  const agentWsRef = useRef<WebSocket | null>(null);
  const resWsRef = useRef<WebSocket | null>(null);
  const mockCleanup = useRef<(() => void) | null>(null);
  const firstListRef = useRef(false);
  const mockModeRef = useRef(false);

  mockModeRef.current = state.mockMode;

  // All WS/mock logic uses refs to avoid useCallback dependency cycles
  const dispatchMsg = useCallback((msg: WSMessage) => {
    if (firstListRef.current && msg.type === 'agent_update') {
      dispatch({ type: 'clear_agents' });
      firstListRef.current = false;
    }
    dispatch(msg);
  }, []);

  const doStopMock = useCallback(() => {
    if (mockCleanup.current) { mockCleanup.current(); mockCleanup.current = null; }
  }, []);

  const doStartMock = useCallback(() => {
    doStopMock();
    dispatch({ type: 'set_mock', payload: true });
    dispatch({ type: 'clear_agents' });
    INITIAL_AGENTS.forEach((a) => dispatch({ type: 'agent_update', payload: a }));
    mockCleanup.current = createMockMessageGenerator(dispatchMsg);
    dispatch({ type: 'set_connected', payload: true });
  }, [dispatchMsg, doStopMock]);

  // ── Agent Bridge WS ──
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnects = 0;
    let timer: ReturnType<typeof setTimeout>;

    function connect() {
      if (ws) { ws.close(); ws = null; }
      try {
        ws = new WebSocket(agentWsUrl);
        agentWsRef.current = ws;
        ws.onopen = () => {
          doStopMock();
          dispatch({ type: 'set_connected', payload: true });
          dispatch({ type: 'set_mock', payload: false });
          reconnects = 0;
          firstListRef.current = true;
          ws!.send(JSON.stringify({ command: 'list_agents' }));
        };
        ws.onmessage = (e) => {
          try { dispatchMsg(JSON.parse(e.data)); } catch { /* ignore */ }
        };
        ws.onclose = () => {
          dispatch({ type: 'set_connected', payload: false });
          reconnects++;
          const delay = Math.min(3000 * Math.pow(1.5, reconnects), 20000);
          timer = setTimeout(connect, delay);
        };
        ws.onerror = () => ws?.close();
      } catch {
        dispatch({ type: 'set_connected', payload: false });
      }
    }
    connect();
    return () => { clearTimeout(timer); ws?.close(); ws = null; doStopMock(); };
  }, [agentWsUrl, dispatchMsg, doStopMock, doStartMock]);

  // ── Resource Monitor WS ──
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnects = 0;
    let timer: ReturnType<typeof setTimeout>;

    function connect() {
      if (ws) { ws.close(); ws = null; }
      try {
        ws = new WebSocket(resWsUrl);
        resWsRef.current = ws;
        ws.onopen = () => { dispatch({ type: 'set_res_connected', payload: true }); reconnects = 0; };
        ws.onmessage = (e) => {
          try {
            const msg = JSON.parse(e.data) as { type: string; payload: ResourceStats };
            if (msg.type === 'resource_stats') dispatch({ type: 'resource_stats', payload: msg.payload });
          } catch { /* ignore */ }
        };
        ws.onclose = () => {
          dispatch({ type: 'set_res_connected', payload: false });
          reconnects++;
          timer = setTimeout(connect, Math.min(2000 * Math.pow(1.5, reconnects), 15000));
        };
        ws.onerror = () => ws?.close();
      } catch {
        dispatch({ type: 'set_res_connected', payload: false });
      }
    }
    connect();
    return () => { clearTimeout(timer); ws?.close(); ws = null; };
  }, [resWsUrl]);

  const toggleDiscussionMode = () => {
    const next = !discussionMode;
    setDiscussionMode(next);
    const ws = agentWsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: 'set_discussion_mode', enabled: next }));
    }
  };

  // ── Memoized derived state ──
  const ideaAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.IDEA), [state.agents]);
  const expAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.EXPERIMENT), [state.agents]);
  const codeAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.CODING), [state.agents]);
  const execAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.EXECUTION), [state.agents]);
  const writeAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.WRITING), [state.agents]);
  const agentMap = useMemo(() => ({
    [AgentLayer.IDEA]: ideaAgents,
    [AgentLayer.EXPERIMENT]: expAgents,
    [AgentLayer.CODING]: codeAgents,
    [AgentLayer.EXECUTION]: execAgents,
    [AgentLayer.WRITING]: writeAgents,
  }), [ideaAgents, expAgents, codeAgents, execAgents, writeAgents]);

  const ideaLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.IDEA), [state.logs]);
  const expLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.EXPERIMENT), [state.logs]);
  const codeLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.CODING), [state.logs]);
  const execLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.EXECUTION), [state.logs]);
  const writeLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.WRITING), [state.logs]);
  const logMap = useMemo(() => ({
    [AgentLayer.IDEA]: ideaLogs,
    [AgentLayer.EXPERIMENT]: expLogs,
    [AgentLayer.CODING]: codeLogs,
    [AgentLayer.EXECUTION]: execLogs,
    [AgentLayer.WRITING]: writeLogs,
  }), [ideaLogs, expLogs, codeLogs, execLogs, writeLogs]);

  const artifactsByProject = useMemo(() => {
    const map: Record<string, Artifact[]> = {};
    for (const a of state.artifacts) {
      const pid = a.projectId || '_global';
      if (!map[pid]) map[pid] = [];
      map[pid].push(a);
    }
    return map;
  }, [state.artifacts]);

  const artifactsByRepo = useMemo(() => {
    const map: Record<string, Artifact[]> = {};
    for (const a of state.artifacts) {
      if (!map[a.repoId]) map[a.repoId] = [];
      map[a.repoId].push(a);
    }
    return map;
  }, [state.artifacts]);

  const workingCount = useMemo(() => state.agents.filter((a) => ['working', 'waiting_discussion', 'discussing'].includes(a.status)).length, [state.agents]);
  const errorCount = useMemo(() => state.agents.filter((a) => a.status === 'error').length, [state.agents]);

  return (
    <LocaleContext.Provider value={localeCtx}>
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>🦞 {t('header.title')}</h1>
          <span className="header-subtitle">1.0.0</span>
        </div>
        <div className="header-stats">
          <span className="stat">{t('header.stat_agents')} <b>{state.agents.length}</b></span>
          <span className="stat">{t('header.stat_active')} <b>{workingCount}</b></span>
          <span className="stat">{t('header.stat_error')} <b>{errorCount}</b></span>
          <span className="stat" dangerouslySetInnerHTML={{ __html: t('header.stat_artifacts', { n: `<b>${state.artifacts.length}</b>` }) }} />
        </div>
        <div className="header-right">
          <button
            className="btn-sm theme-toggle-btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={theme === 'dark' ? t('header.theme_light') : t('header.theme_dark')}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <button
            className="btn-sm lang-toggle-btn"
            onClick={() => localeCtx.setLocale(locale === 'zh' ? 'en' : 'zh')}
            title={locale === 'zh' ? 'Switch to English' : '切换到中文'}
          >
            {locale === 'zh' ? 'EN' : '中文'}
          </button>
        </div>
      </header>

      <div className="main-layout">
        <div className="side-panel repo-panel">
          <ProjectPanel
            projects={state.projects}
            connected={state.connected}
            selectedProjectId={state.selectedProjectId}
            artifactsByProject={artifactsByProject}
            discussionMode={discussionMode}
            onToggleDiscussion={toggleDiscussionMode}
            onSelect={(projectId) => dispatch({ type: 'select_project', payload: projectId })}
            onResume={(projectId) => {
              const ws = agentWsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'resume_project', projectId }));
              }
            }}
            onPause={(projectId) => {
              const ws = agentWsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'pause_project', projectId }));
              }
            }}
            onRestart={(projectId) => {
              const ws = agentWsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'restart_project', projectId }));
              }
            }}
            onDelete={(projectId) => {
              const ws = agentWsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'delete_project', projectId }));
                if (state.selectedProjectId === projectId) {
                  dispatch({ type: 'select_project', payload: null });
                }
              }
            }}
            onQuickSubmit={(topic, mode, researchAngles, referencePapers, referenceFiles, paths) => {
              const ws = agentWsRef.current;
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                  command: 'quick_submit',
                  topic,
                  mode,
                  researchAngles: researchAngles.length > 0 ? researchAngles : undefined,
                  referencePapers: referencePapers || undefined,
                  referenceFiles: referenceFiles.length > 0 ? (referenceFiles as ReferencePdfUpload[]) : undefined,
                  codebasesDir: paths?.codebases || undefined,
                  datasetsDir: paths?.datasets || undefined,
                  checkpointsDir: paths?.checkpoints || undefined,
                }));
              }
            }}
          />
        </div>

        <div className="pyramid-container">
          <ResourceMonitor stats={state.resources} connected={state.resConnected} />
          <div className="pyramid-wrapper">
            <div className="pyramid">
              {ALL_LAYERS.map((layer, idx) => {
                const hasWorking = agentMap[layer].some((a) => ['working', 'waiting_discussion', 'discussing'].includes(a.status));
                return (
                  <div key={layer} className="pyramid-tier">
                    <LayerPanel
                      layer={layer}
                      agents={agentMap[layer]}
                      logs={logMap[layer]}
                      tierIndex={idx}
                      selectedProjectId={state.selectedProjectId}
                    />
                    {idx < ALL_LAYERS.length - 1 && (
                      <DataFlowArrow active={hasWorking} />
                    )}
                  </div>
                );
              })}
            </div>
            <div className={`feedback-loop ${agentMap[AgentLayer.EXECUTION].some((a) => a.status === 'done') ? 'active' : ''}`}>
              <div className="fb-line fb-bottom" />
              <div className="fb-line fb-side"><div className="fb-pulse" /></div>
              <div className="fb-line fb-top" />
              <div className="fb-tip" />
            </div>
          </div>
        </div>

        <div className="side-panel log-panel">
          <LogPanel logs={state.logs} />
          {/* <QueuePanel queues={state.queues} /> */}
          <div className="shelf-section">
            <div className="shelf-section-header">
              <span className="shelf-section-title">{t('header.shared_repo')}</span>
            </div>
            <div className="shelf-list">
              {ALL_REPOS.map((repoId) => (
                <DataShelf
                  key={repoId}
                  repoId={repoId}
                  artifacts={artifactsByRepo[repoId] || []}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      <HumanFeedbackPanel
        messages={state.chatMessages}
        connected={state.connected}
        onSend={(content, targetLayer) => {
          const ws = agentWsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              command: 'chat_input',
              content,
              targetLayer: targetLayer || 'all',
            }));
            dispatch({
              type: 'chat_message',
              payload: {
                id: `user-${Date.now()}`,
                role: 'user',
                content,
                targetLayer: targetLayer || 'all',
                timestamp: Date.now(),
              },
            });
          }
        }}
      />
    </div>
    </LocaleContext.Provider>
  );
}
