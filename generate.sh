#!/bin/bash

# Pipeline runner for manage_detections.py
# Usage:
#   ./generate.sh start "python manage_detections.py build --limit 100"
#   ./generate.sh stop
#   ./generate.sh status

PIDFILE=".pipeline.pid"

start() {
    if [ -f "$PIDFILE" ]; then
        OLD_PID=$(cat "$PIDFILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Pipeline already running with PID: $OLD_PID"
            echo "Use './generate.sh stop' to stop it first"
            exit 1
        else
            rm -f "$PIDFILE"
        fi
    fi

    if [ -z "$1" ]; then
        echo "Usage: $0 start \"command args...\""
        exit 1
    fi

    CMD="$1"

    # Get current timestamp for log filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOGFILE="run_${TIMESTAMP}.log"

    # Run the command with nohup and log output
    # PYTHONUNBUFFERED=1 forces immediate output flushing
    nohup bash -c "PYTHONUNBUFFERED=1 $CMD" > "$LOGFILE" 2>&1 < /dev/null &
    PID=$!

    # Save PID to file
    echo "$PID" > "$PIDFILE"

    echo "Started pipeline with PID: $PID"
    echo "Logging to: $LOGFILE"
    echo "Use './generate.sh stop' to stop pipeline"
    echo "Use './generate.sh status' to check status"
}

stop() {
    if [ ! -f "$PIDFILE" ]; then
        echo "No pipeline PID file found"
        exit 0
    fi

    MAIN_PID=$(cat "$PIDFILE")
    if kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "Stopping pipeline process: $MAIN_PID"
        kill -TERM "$MAIN_PID" 2>/dev/null
        sleep 1
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            kill -9 "$MAIN_PID" 2>/dev/null
        fi
    fi
    rm -f "$PIDFILE"
    echo "Pipeline stopped"
}

status() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Pipeline is RUNNING (PID: $PID)"
            echo ""
            echo "Process info:"
            ps -p "$PID" -o pid,ppid,etime,command
            return 0
        else
            echo "Pipeline process $PID is not running (stale PID file)"
            rm -f "$PIDFILE"
        fi
    else
        echo "No pipeline running"
    fi
}

# Main command handler
case "${1:-}" in
    start)
        shift
        start "$*"
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo ""
        echo "Commands:"
        echo "  start \"command\"  - Start pipeline in background"
        echo "  stop             - Stop pipeline process"
        echo "  status           - Check if pipeline is running"
        echo ""
        echo "Examples:"
        echo "  $0 start \"python manage_detections.py build\""
        echo "  $0 start \"python manage_detections.py build --limit 100\""
        echo "  $0 stop"
        echo "  $0 status"
        exit 1
        ;;
esac
