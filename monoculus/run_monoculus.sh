#!/bin/bash

# Monoculus Unified Runner
# Usage: ./run_monoculus.sh [--nogui]

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/static/images"

MONITOR_LOG="${LOG_DIR}/monitor.log"
BACKEND_LOG="${LOG_DIR}/backend.log"

NOGUI_FLAG=""
HEADLESS_MODE=false

# Check for --nogui argument
for arg in "$@"; do
    if [ "$arg" == "--nogui" ]; then
        NOGUI_FLAG="--nogui"
        HEADLESS_MODE=true
        echo "🌐 Running in HEADLESS mode (No Camera GUI)"
    fi
done

# Cleanup Function
cleanup() {
    echo ""
    echo "🛑 Stopping Monoculus system..."
    
    # Kill captured PIDs
    if [ -n "$BACKEND_PID" ]; then
        echo "   - Killing Backend (PID $BACKEND_PID)"
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ -n "$MONITOR_PID" ]; then
        echo "   - Killing Monitor (PID $MONITOR_PID)"
        kill $MONITOR_PID 2>/dev/null
    fi
    
    # Fallback cleanup
    pkill -f integrated_monitor_2.py 2>/dev/null
    pkill -f app.py 2>/dev/null
    
    exit 0
}

# Trap Signals
trap cleanup SIGINT SIGTERM EXIT

# 1. Cleanup existing processes
echo "🧹 Cleaning up existing Monoculus processes..."
pkill -9 -f main.py 2>/dev/null
pkill -9 -f integrated_monitor_2.py 2>/dev/null
pkill -9 -f app.py 2>/dev/null
sleep 1

# 2. Start Backend Server
echo "🚀 Starting Backend Server (Port: 5000)..."
cd "${PROJECT_DIR}"
nohup python3 app.py > "${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
sleep 2

if ! ps -p $BACKEND_PID > /dev/null; then
    echo "❌ ERROR: Backend failed to start. Check logs at ${BACKEND_LOG}"
    exit 1
fi
echo "✅ Backend started (PID: $BACKEND_PID)"

# 3. Start Monitoring System
echo "🚀 Starting Monitoring System..."
echo "💡 To see all logs: tail -f logs/*.log"
echo "------------------------------------------------"

# Run Monitor in Background & Wait
# This allows the shell script to receive signals and run the trap
if [ "$HEADLESS_MODE" == "true" ]; then
    export QT_QPA_PLATFORM=offscreen
    # Pipe output to log and shell
    python3 main.py --nogui > >(tee -a "${MONITOR_LOG}") 2>&1 &
    MONITOR_PID=$!
else
    if [ -z "$DISPLAY" ]; then
        export DISPLAY=:0
    fi
    echo "📺 Camera GUI will be displayed on $DISPLAY"
    python3 main.py > >(tee -a "${MONITOR_LOG}") 2>&1 &
    MONITOR_PID=$!
fi

# Wait for process
wait $MONITOR_PID
