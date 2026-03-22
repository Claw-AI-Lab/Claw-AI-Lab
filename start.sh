#!/bin/bash
# 🦞 龙虾 Agent 军团 — 一键启动/停止
# Usage: ./start.sh [start|stop|restart|status]

BASE="$(cd "$(dirname "$0")" && pwd)"
PY="/home/user/miniforge3/bin/python3"
FE="$BASE/frontend"
LOG="$BASE/logs"
PIDF="$BASE/.pids"

export PATH="/home/user/.local/share/fnm:$PATH"
eval "$(/home/user/.local/share/fnm/fnm env 2>/dev/null)" 2>/dev/null

mkdir -p "$LOG" "$PIDF"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[0;33m'; N='\033[0m'

do_start() {
    echo "🦞 启动龙虾 Agent 军团..."
    echo ""

    # 1) Resource Monitor
    if ss -tlnp 2>/dev/null | grep -q ":8785 "; then
        echo -e "  ${Y}⏭ resource_monitor 已在运行${N}"
    else
        nohup $PY -u "$BASE/backend/services/resource_monitor.py" --port 8785 \
            > "$LOG/resource_monitor.log" 2>&1 &
        echo $! > "$PIDF/resource_monitor.pid"
        sleep 1
        echo -e "  ${G}✅ resource_monitor (PID=$!)${N}"
    fi

    # 2) Agent Bridge
    if ss -tlnp 2>/dev/null | grep -q ":8786 "; then
        echo -e "  ${Y}⏭ agent_bridge 已在运行${N}"
    else
        LLM_CFG=""
        for _cfg in "$BASE/backend/agent/config_gpu_project.yaml" \
                     "$BASE/backend/agent/config.researchclaw.yaml"; do
            [ -f "$_cfg" ] && LLM_CFG="--llm-config $_cfg" && break
        done
        nohup $PY -u "$BASE/backend/services/agent_bridge.py" \
            --port 8786 --python "$PY" \
            --agent-dir "$BASE/backend/agent" \
            --runs-dir "$BASE/backend/runs" \
            --pool-idea 2 --pool-exp 2 --pool-code 3 --pool-exec 4 \
            $LLM_CFG \
            > "$LOG/agent_bridge.log" 2>&1 &
        echo $! > "$PIDF/agent_bridge.pid"
        sleep 1
        echo -e "  ${G}✅ agent_bridge (PID=$!)${N}"
    fi

    # 3) Frontend Vite
    if ss -tlnp 2>/dev/null | grep -q ":5190 "; then
        echo -e "  ${Y}⏭ frontend 已在运行${N}"
    else
        cd "$FE"
        nohup npx vite --host 0.0.0.0 --port 5190 \
            > "$LOG/frontend.log" 2>&1 &
        echo $! > "$PIDF/frontend.pid"
        sleep 2
        echo -e "  ${G}✅ frontend (PID=$!)${N}"
        cd "$BASE"
    fi

    echo ""
    echo "📍 服务地址:"
    echo -e "   ${G}前端 UI:      http://localhost:5190/${N}"
    echo "   资源监控 WS:  ws://localhost:8785"
    echo "   Agent Bridge: ws://localhost:8786"
    echo ""
}

do_stop() {
    echo "🛑 停止所有服务..."
    for svc in frontend agent_bridge resource_monitor; do
        f="$PIDF/$svc.pid"
        if [ -f "$f" ]; then
            pid=$(cat "$f")
            # Verify the PID actually belongs to our service before killing
            if kill -0 "$pid" 2>/dev/null; then
                proc_cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
                case "$proc_cmd" in
                    *resource_monitor*|*agent_bridge*|*vite*|*npm*)
                        kill "$pid" 2>/dev/null && sleep 0.5
                        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
                        echo -e "  ${G}⏹ $svc (PID=$pid)${N}"
                        ;;
                    *)
                        echo -e "  ${Y}⏭ $svc PID=$pid belongs to another process, skipping${N}"
                        ;;
                esac
            fi
            rm -f "$f"
        fi
    done
    # Fallback: kill only LISTENING server processes on our ports (not clients/SSH)
    for port in 5190 8785 8786; do
        ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | while read pid; do
            proc_cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
            case "$proc_cmd" in
                *resource_monitor*|*agent_bridge*|*vite*|*node*)
                    kill "$pid" 2>/dev/null
                    echo -e "  ${G}⏹ port $port (PID=$pid)${N}"
                    ;;
            esac
        done
    done
    echo ""
}

do_status() {
    echo "📊 服务状态:"
    for pair in "resource_monitor:8785" "agent_bridge:8786" "frontend:5190"; do
        svc="${pair%%:*}"; port="${pair##*:}"
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            echo -e "  ${G}● $svc${N} (port $port)"
        else
            echo -e "  ${R}○ $svc${N} (port $port)"
        fi
    done
    echo ""
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 1; do_start ;;
    status)  do_status ;;
    *)       echo "Usage: $0 {start|stop|restart|status}" ;;
esac
