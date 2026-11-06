#!/bin/bash
# E2 chain (one GPU job at a time, resumable): oracle gate, then the hard router from E1 (style-matched B_teacher and realistic A_rw).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
X="--stages ckpt/m1 --experts ckpt/expert_b2"
./run.sh eval e2_oracle $X --oracle
./run.sh eval e2_router_B_teacher $X --router results/router_e1_B_teacher.pt
./run.sh eval e2_router_A_rw $X --router results/router_e1_A_rw.pt
nice -n 19 python3 scripts/e2_chat_general.py --router results/router_e1_B_teacher.pt --out results/e2_chat_general_B_teacher.json
nice -n 19 python3 scripts/e2_chat_general.py --router results/router_e1_A_rw.pt --out results/e2_chat_general_A_rw.json
