#!/bin/bash
# Log RAM, load, GPU temp/power/util/clocks, GPU processes and hottest thermal zone every second (on disk, survives reboot).
LOG="$(dirname "$0")/../logs/memlog.log"
while true; do
  TZ_MAX=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  G=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm,clocks_event_reasons.active --format=csv,noheader 2>/dev/null)
  P=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')
  echo "$(date '+%F %T') avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G load=$(cut -d' ' -f1 /proc/loadavg) tz_max=$((TZ_MAX/1000))C gpu=[$G] procs=$P" >> "$LOG"
  sync -f "$LOG" 2>/dev/null
  sleep 1
done
