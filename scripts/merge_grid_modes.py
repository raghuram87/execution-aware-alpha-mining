"""Merge a grid run whose two arms ran as separate processes
(run_full_grid.py --modes baseline / --modes execution_aware) into the
standard combined files the analysis and figure scripts read:
grid_summary.csv, and per combination wfo_folds.csv and table1_wfo.csv.

Usage: python scripts/merge_grid_modes.py [--grid-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config

GRID_DIR = config.RESULTS / "local_llm_runs" / "full_grid_sharadar_2006_2026"
MODES = ("baseline", "execution_aware")
OPTIONAL_MODES = ("reward_only",)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", default=str(GRID_DIR))
    grid = Path(ap.parse_args().grid_dir)

    parts = [pd.read_csv(f) for m in MODES + OPTIONAL_MODES for f in sorted(grid.glob(f"grid_summary_{m}*.csv"))]
    summary = pd.concat(parts, ignore_index=True).sort_values(["seed_alpha", "llm_seed", "mode"])
    summary.to_csv(grid / "grid_summary.csv", index=False)
    print(f"grid_summary.csv: {len(summary)} rows")

    incomplete = []
    for combo in sorted(grid.glob("*/seed_*")):
        folds = [combo / f"wfo_folds_{m}.csv" for m in MODES]
        tables = [combo / f"table1_wfo_{m}.csv" for m in MODES]
        if not all(p.exists() for p in folds + tables):
            incomplete.append(combo.relative_to(grid))
            continue
        folds += [combo / f"wfo_folds_{m}.csv" for m in OPTIONAL_MODES if (combo / f"wfo_folds_{m}.csv").exists()]
        tables += [combo / f"table1_wfo_{m}.csv" for m in OPTIONAL_MODES if (combo / f"table1_wfo_{m}.csv").exists()]
        pd.concat([pd.read_csv(p) for p in folds], ignore_index=True).to_csv(combo / "wfo_folds.csv", index=False)
        pd.concat([pd.read_csv(p, index_col=0) for p in tables]).to_csv(combo / "table1_wfo.csv")
    print(f"merged {len(list(grid.glob('*/seed_*'))) - len(incomplete)} combinations; incomplete: {incomplete or 'none'}")
