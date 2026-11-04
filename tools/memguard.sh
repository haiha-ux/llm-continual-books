#!/bin/bash
# Kill this project's heavy jobs if system memory gets low (unified memory: GPU allocations are not covered by cgroup limits).
# Usage: memguard.sh [min_avail_GB=30] ; logs to logs/memguard.log
MIN=${1:-30}
LOG="$(dirname "$0")/../logs/memguard.log"
PAT='llm-continual-books|gen_teacher|run_method|eval_ckpt|gen_replay|build_data|vllm'
while true; do
  AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$AV" -lt "$MIN" ]; then
    PIDS=$(pgrep -u "$USER" -f "python.*($PAT)")
    echo "$(date '+%F %T') LOW avail=${AV}G < ${MIN}G -> SIGKILL $PIDS" >> "$LOG"
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null
    sleep 10
  fi
  sleep 2
done
