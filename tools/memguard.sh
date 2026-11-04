#!/bin/bash
# Safety guard for this project's heavy jobs on the GB10 box.
# Kills our python jobs if MemAvailable < MIN_GB, or hottest ACPI zone >= TZ_MAX_C, or GPU temp >= GPU_MAX_C.
# GB10 boxes hard-freeze (EC power cut, no kernel log) under sustained load when they run hot, so stop early.
MIN=${1:-30}; TZ_MAX=${2:-85}; GPU_MAX=${3:-80}
LOG="$(dirname "$0")/../logs/memguard.log"
PAT='llm-continual-books|gen_teacher|run_method|eval_ckpt|gen_replay|build_data|stress_test'
while true; do
  AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  TZ=$(( $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1) / 1000 ))
  GT=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1); GT=${GT:-0}
  WHY=""
  [ "$AV" -lt "$MIN" ] && WHY="avail=${AV}G<${MIN}G"
  [ "$TZ" -ge "$TZ_MAX" ] && WHY="$WHY acpi=${TZ}C>=${TZ_MAX}C"
  [ "$GT" -ge "$GPU_MAX" ] && WHY="$WHY gpu=${GT}C>=${GPU_MAX}C"
  if [ -n "$WHY" ]; then
    PIDS=$(pgrep -u "$USER" -f "python.*($PAT)")
    echo "$(date '+%F %T') STOP $WHY -> SIGKILL $PIDS" >> "$LOG"
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null
    sleep 10
  fi
  sleep 1
done
