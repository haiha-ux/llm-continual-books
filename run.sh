#!/bin/bash
# Reproduce one stage. All GPU work runs one job at a time, niced, <= 4 threads, behind the shared GPU locks,
# and pauses itself when the box gets hot (clb.thermal). One master seed: SEED (default 0).
#   ./run.sh gen 1,2,3                 teacher data via Ollama (resumable)
#   ./run.sh build                     filter + split + 8-gram overlap check -> data/final
#   ./run.sh eval NAME [--stages ckpt/m1 ...] [--experts ckpt/expert_b2]
#   ./run.sh train configs/naive_b2.json
#   ./run.sh replay m1 --stages ckpt/m1
#   ./run.sh book 2 OUT_DIR            full book-2 experiment: naive / KL self-replay / insulated expert (+ evals) into OUT_DIR
#   ./run.sh summary
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
SEED=${SEED:-0}

heavy() {
  local avail hot
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 70 ] || { echo "only ${avail} GB available, refusing to start"; exit 1; }
  while :; do
    hot=$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) / 1000 ))
    [ "$hot" -le 74 ] && break
    echo "box at ${hot} C, waiting"; sleep 20
  done
  nice -n 19 ionice -c3 flock /tmp/geoe_gpu.lock flock /tmp/geoe_gpu2.lock "$@"
}
train() { local cfg=$1; shift; heavy python3 scripts/run_method.py "$cfg" seed="$SEED" "$@"; }
evalm() { local name=$1; shift; heavy python3 scripts/eval_ckpt.py "$name" "$@"; }

case "${1:-}" in
  gen)     nice -n 19 python3 scripts/gen_teacher.py --books "${2:-1,2,3}" ;;
  build)   python3 scripts/build_data.py ;;
  eval)    shift; evalm "$@" ;;
  train)   shift; train "$@" ;;
  replay)  shift; heavy python3 scripts/gen_replay.py "$@" --seed "$SEED" ;;
  summary) python3 scripts/summarize.py ;;
  book)
    B=$2; OUT=${3:-ckpt}
    for m in naive kl expert; do
      train "configs/${m}_b${B}.json" "out=\"$OUT/${m}_b${B}\"" 
      if [ "$m" = expert ]; then evalm "${m}_b${B}" --stages ckpt/m1 --experts "$OUT/${m}_b${B}"
      else evalm "${m}_b${B}" --stages ckpt/m1 "$OUT/${m}_b${B}"; fi
    done ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
