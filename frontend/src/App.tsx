import { useReducer, useEffect, useCallback, useRef, useState, useMemo } from 'react';
import { AgentLayer, ALL_LAYERS, ALL_REPOS } from './types';
import type { AppState, WSMessage, ResourceStats, LobsterAgent, QueueMap } from './types';
import { INITIAL_AGENTS, createMockMessageGenerator } from './mock';
import LayerPanel from './components/LayerPanel';
import DataShelf from './components/DataShelf';
import ResourceMonitor from './components/ResourceMonitor';
import LogPanel from './components/LogPanel';
import QueuePanel from './components/QueuePanel';
import './App.css';

type Action =
  | WSMessage
  | { type: 'set_connected'; payload: boolean }
  | { type: 'set_mock'; payload: boolean }
  | { type: 'set_res_connected'; payload: boolean }
  | { type: 'clear_agents' };

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
    case 'artifact_produced':
      return { ...state, artifacts: [action.payload, ...state.artifacts] };
    case 'resource_stats':
      return { ...state, resources: action.payload, resConnected: true };
    case 'log':
      return { ...state, logs: [...state.logs, action.payload] };
    case 'set_connected':
      return { ...state, connected: action.payload };
    case 'set_res_connected':
      return { ...state, resConnected: action.payload };
    case 'queue_update': {
      const prev = JSON.stringify(state.queues);
      const next = JSON.stringify(action.payload);
      return prev === next ? state : { ...state, queues: action.payload };
    }
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
  resources: null,
  resConnected: false,
  connected: false,
  mockMode: false,
};

export default function App() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const [agentWsUrl, setAgentWsUrl] = useState(`${WS_PROTO}//${window.location.host}/ws/agents`);
  const [resWsUrl, setResWsUrl] = useState(`${WS_PROTO}//${window.location.host}/ws/resources`);
  const [showSettings, setShowSettings] = useState(false);
  const agentWsRef = useRef<WebSocket | null>(null);
  const agentReconnRef = useRef<number>(0);
  const resWsRef = useRef<WebSocket | null>(null);
  const resReconnRef = useRef<number>(0);
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
          if (reconnects >= 4 && !mockModeRef.current) {
            doStartMock();
          }
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

  const toggleMode = () => {
    if (state.mockMode) {
      doStopMock();
      dispatch({ type: 'set_mock', payload: false });
      dispatch({ type: 'clear_agents' });
    } else {
      if (agentWsRef.current) { agentWsRef.current.close(); agentWsRef.current = null; }
      doStartMock();
    }
  };

  // ── Memoized derived state ──
  const ideaAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.IDEA), [state.agents]);
  const expAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.EXPERIMENT), [state.agents]);
  const codeAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.CODING), [state.agents]);
  const execAgents = useMemo(() => state.agents.filter((a) => a.layer === AgentLayer.EXECUTION), [state.agents]);
  const agentMap = useMemo(() => ({
    [AgentLayer.IDEA]: ideaAgents,
    [AgentLayer.EXPERIMENT]: expAgents,
    [AgentLayer.CODING]: codeAgents,
    [AgentLayer.EXECUTION]: execAgents,
  }), [ideaAgents, expAgents, codeAgents, execAgents]);

  const ideaLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.IDEA), [state.logs]);
  const expLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.EXPERIMENT), [state.logs]);
  const codeLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.CODING), [state.logs]);
  const execLogs = useMemo(() => state.logs.filter((l) => l.layer === AgentLayer.EXECUTION), [state.logs]);
  const logMap = useMemo(() => ({
    [AgentLayer.IDEA]: ideaLogs,
    [AgentLayer.EXPERIMENT]: expLogs,
    [AgentLayer.CODING]: codeLogs,
    [AgentLayer.EXECUTION]: execLogs,
  }), [ideaLogs, expLogs, codeLogs, execLogs]);

  const knowledgeArt = useMemo(() => state.artifacts.filter((a) => a.repoId === 'knowledge'), [state.artifacts]);
  const expDesignArt = useMemo(() => state.artifacts.filter((a) => a.repoId === 'exp_design'), [state.artifacts]);
  const codebaseArt = useMemo(() => state.artifacts.filter((a) => a.repoId === 'codebase'), [state.artifacts]);
  const resultsArt = useMemo(() => state.artifacts.filter((a) => a.repoId === 'results'), [state.artifacts]);
  const artMap = useMemo(() => ({
    knowledge: knowledgeArt, exp_design: expDesignArt,
    codebase: codebaseArt, results: resultsArt,
  }), [knowledgeArt, expDesignArt, codebaseArt, resultsArt]);

  const workingCount = useMemo(() => state.agents.filter((a) => a.status === 'working').length, [state.agents]);
  const errorCount = useMemo(() => state.agents.filter((a) => a.status === 'error').length, [state.agents]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>🦞 Pyramid Research Team</h1>
          <span className="header-subtitle">1.0.0</span>
        </div>
        <div className="header-stats">
          <span className="stat">🦞 <b>{state.agents.length}</b></span>
          <span className="stat">🔧 <b>{workingCount}</b></span>
          <span className="stat">❌ <b>{errorCount}</b></span>
          <span className="stat">📦 <b>{state.artifacts.length}</b></span>
        </div>
        <div className="header-right">
          <button className="btn-sm" onClick={() => setShowSettings(!showSettings)}>⚙️</button>
        </div>
      </header>

      {showSettings && (
        <div className="settings-bar">
          <label>
            <button className="btn-sm mode-btn" onClick={toggleMode}>
              {state.mockMode ? '🎭 模拟 → 切换真实' : '🔌 真实 → 切换模拟'}
            </button>
          </label>
          <span className="settings-sep">|</span>
          <label>Agent: <input value={agentWsUrl} onChange={(e) => setAgentWsUrl(e.target.value)} /></label>
          <label>资源: <input value={resWsUrl} onChange={(e) => setResWsUrl(e.target.value)} /></label>
        </div>
      )}

      <div className="main-layout">
        <div className="side-panel repo-panel">
          <h2>📚 共享数据仓库</h2>
          <div className="repo-list">
            {ALL_REPOS.map((repoId) => (
              <DataShelf key={repoId} repoId={repoId} artifacts={artMap[repoId]} />
            ))}
          </div>
          <QueuePanel queues={state.queues} />
        </div>

        <div className="pyramid-container">
          <ResourceMonitor stats={state.resources} connected={state.resConnected} />
          <div className="pyramid">
            {ALL_LAYERS.map((layer, idx) => (
              <LayerPanel
                key={layer}
                layer={layer}
                agents={agentMap[layer]}
                logs={logMap[layer]}
                tierIndex={idx}
              />
            ))}
          </div>
        </div>

        <LogPanel logs={state.logs} />
      </div>
    </div>
  );
}
