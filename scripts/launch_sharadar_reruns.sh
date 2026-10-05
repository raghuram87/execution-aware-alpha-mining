#!/bin/bash
# Re-runs the full grid and the gamma sweep on Sharadar panels, one model per GPU:
#   GPU 0: grid baseline arm, then gamma settings 0-4
#   GPU 1: grid execution-aware arm, then gamma settings 5-8
cd "$(dirname "$0")/.."
PY=/home/raghuram/anaconda3/envs/reccoai/bin/python3
GRID=results/local_llm_runs/full_grid_sharadar_2006_2026
GAMMA=results/local_llm_runs/gamma_sweep_sharadar_volatility_seed1001
mkdir -p "$GRID" "$GAMMA"
(
  CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_full_grid.py --seeds 1001 2002 3003 --modes baseline > "$GRID/run_baseline.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_gamma_sweep.py --settings 0 1 2 3 4 --tag gpu0 > "$GAMMA/run_gpu0.log" 2>&1
  echo "GPU0 DONE $(date)" >> "$GRID/launcher_status.txt"
) &
(
  CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_full_grid.py --seeds 1001 2002 3003 --modes execution_aware > "$GRID/run_execution_aware.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_gamma_sweep.py --settings 5 6 7 8 --tag gpu1 > "$GAMMA/run_gpu1.log" 2>&1
  echo "GPU1 DONE $(date)" >> "$GRID/launcher_status.txt"
) &
wait
