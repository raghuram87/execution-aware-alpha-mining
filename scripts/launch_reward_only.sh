#!/bin/bash
# Reward-only arm (execution-aware reward, baseline prompt and critique).
# Each GPU starts once launch_sharadar_reruns.sh reports that GPU done.
cd "$(dirname "$0")/.."
PY=/home/raghuram/anaconda3/envs/reccoai/bin/python3
GRID=results/local_llm_runs/full_grid_sharadar_2006_2026
STATUS="$GRID/launcher_status.txt"
run_after() {  # gpu, alphas...
  local gpu=$1; shift
  until grep -q "GPU$gpu DONE" "$STATUS" 2>/dev/null; do sleep 300; done
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/run_full_grid.py --seeds 1001 2002 3003 --modes reward_only --tag gpu$gpu --alphas "$@" > "$GRID/run_reward_only_gpu$gpu.log" 2>&1
  echo "REWARD_ONLY GPU$gpu DONE $(date)" >> "$STATUS"
}
run_after 0 reversal volume &
run_after 1 volatility momentum &
wait
