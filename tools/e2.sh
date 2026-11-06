#!/bin/bash
# E2 chain (one GPU job at a time, resumable; the thermal guard may SIGKILL a job, so each step retries): oracle gate, then the
# hard router that is usable without book-1 text (A_rw), then the chat-format general-question check.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
X="--stages ckpt/m1 --experts ckpt/expert_b2"
ok() { grep -q '"complete": true' "results/raw/$1.json" 2>/dev/null; }
step() { for i in 1 2 3 4; do ok "$1" && return 0; ./run.sh eval "$@"; ok "$1" && return 0; sleep 240; done; }
step e2_oracle $X --oracle
step e2_router_A_rw $X --router results/router_e1_A_rw.pt
[ -f results/e2_chat_general_A_rw.json ] || nice -n 19 python3 scripts/e2_chat_general.py --router results/router_e1_A_rw.pt --out results/e2_chat_general_A_rw.json
