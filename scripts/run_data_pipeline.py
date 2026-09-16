"""Week 1: download and cache the S&P 100 daily OHLCV panels.

Usage: python scripts/run_data_pipeline.py [--refresh]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papers.execution_aware_alpha_mining.src.data_pipeline import get_panels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="force re-download even if cached")
    args = parser.parse_args()

    panels = get_panels(force_refresh=args.refresh)
    print("Panels:", {k: v.shape for k, v in panels.items()})
