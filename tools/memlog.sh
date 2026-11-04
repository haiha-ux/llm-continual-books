#!/bin/bash
# Log available RAM, load and GPU memory every second to logs/memlog.log (on disk, survives reboot).
LOG="$(dirname "$0")/../logs/memlog.log"
while true; do
  echo "$(date '+%F %T') avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G load=$(cut -d' ' -f1 /proc/loadavg) gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')" >> "$LOG"
  sync -f "$LOG" 2>/dev/null
  sleep 1
done
