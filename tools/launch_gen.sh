#!/bin/bash
cd /home/sw-haiha/Hai/portfolio/llm-continual-books
export PYTHONPATH=src
until [ -f data/gen/book2_eval_rw.ok ]; do
  while [ $(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) / 1000 )) -gt 74 ]; do sleep 10; done
  nice -n 19 ionice -c3 flock /tmp/geoe_gpu.lock flock /tmp/geoe_gpu2.lock python3 scripts/gen_teacher.py --books 1,2
  sleep 20
done
