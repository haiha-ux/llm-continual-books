#!/bin/bash
# Whole book-2 experiment, unattended and restartable: every step is skipped when its output exists and retried
# (after the box cools) when the guard killed it.   tools/pipeline.sh   (log: logs/pipeline.log)
cd "$(dirname "$0")/.."
ok() { [ -e "$1" ] || return 1; case $1 in results/raw/*) grep -q '"complete": true' "$1" ;; esac; }
step() {  # step OUTPUT_FILE cmd...   retry up to 30 times, 4 min apart
  local out=$1; shift
  for i in $(seq 30); do ok "$out" && return 0; echo "$(date '+%F %T') run: $*"; "$@"; ok "$out" && return 0; sleep 240; done
  echo "$(date '+%F %T') GAVE UP: $*"; exit 1
}
until [ -f data/gen/book2_eval_rw.ok ]; do sleep 60; done          # teacher stage (launched separately) must be done
step data/final/report.json python3 scripts/build_data.py --books 1,2
step results/raw/m0.json ./run.sh eval m0
step ckpt/m1/lora.pt ./run.sh train configs/m1_book1.json
step results/raw/m1.json ./run.sh eval m1 --stages ckpt/m1
step data/replay/m1.jsonl ./run.sh replay m1 --stages ckpt/m1
for m in naive kl expert; do
  step ckpt/${m}_b2/$([ $m = expert ] && echo gate.pt || echo lora.pt) ./run.sh train configs/${m}_b2.json
  if [ $m = expert ]; then step results/raw/${m}_b2.json ./run.sh eval ${m}_b2 --stages ckpt/m1 --experts ckpt/${m}_b2
  else step results/raw/${m}_b2.json ./run.sh eval ${m}_b2 --stages ckpt/m1 ckpt/${m}_b2; fi
done
python3 scripts/summarize.py
echo "$(date '+%F %T') PIPELINE DONE"
