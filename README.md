# Execution-Aware Alpha Mining

Code, data pipeline, and result logs accompanying the Finance Research
Letters submission "Execution-Aware Alpha Mining: Teaching LLM Factor
Agents to Price Their Own Trading Costs" (`paper/manuscript.md` /
`paper/manuscript.docx`). A locally-hosted LLM (Qwen2.5-Coder-14B-Instruct)
proposes daily cross-sectional equity factors; each candidate is backtested
on a true **point-in-time** S&P universe and priced for realistic execution
cost (Corwin-Schultz spread + square-root market impact); the resulting
diagnostics are fed back to the agent every round. We compare an agent
rewarded on **raw returns only** ("baseline") against one rewarded on
**Gross_IR net of a turnover/cost penalty** ("execution-aware"), holding the
search process, universe, and cost model fixed.

**Headline result** (21-fold walk-forward, 2006-2026, averaged across 3 LLM
sampling seeds x 4 seed-alpha categories — see `results/local_llm_runs/full_grid_2006_2026/grid_summary.csv`):
the cost-blind baseline converges to 49-101%/day turnover and is wiped out
(net Sharpe -4.7 to -11.2, cumulative return -100%) in every category
despite genuine gross signal (avg. gross Sharpe 0.67); the execution-aware
agent cuts turnover 22-38x in every category and, for the low-volatility-
anomaly seed, is net *profitable* (net Sharpe +0.35, cumulative return
+117%). Full detail: `paper/manuscript.md` Section 3.

## Methodology summary

- **Point-in-time universe** (`src/pit_universe.py`): true historical S&P
  constituent membership (1999-2026, not a survivorship-biased current-day
  snapshot), reconstructed from a public historical-components record.
  Daily OHLCV comes from a local multi-vendor price archive (not
  redistributable; not required to reproduce — see below) supplemented by
  a `yfinance` gap-fill. Each walk-forward fold trades its own
  liquidity-capped ~150-220-ticker subset, reconstituted **annually within
  the fold** using only data available as of each reconstitution date — a
  single global or train-start-only liquidity ranking would let an early
  fold trade names that only became liquid decades later, or leave a test
  year trading a stale universe (see `pit_universe.build_fold_universe`).
- **Cost engine** (`src/cost_engine.py`): Corwin & Schultz (2012) high-low
  spread estimator + a Grinold-Kahn-style square-root impact model.
- **Factor DSL + guardrails** (`src/factor_eval.py`): LLM output restricted
  to an AST-whitelisted expression grammar; six hard anti-pattern rules
  (`check_anti_patterns`) reject redundant ranking, no-op scalar rescaling,
  cascading smoothers, unprotected division, unbounded powers, and
  out-of-range lookback windows before a candidate is ever backtested.
- **Agent + reward** (`src/agent_loop.py`, `src/llm_client.py`): local
  Qwen2.5-Coder-14B-Instruct via llama.cpp, temperature 0.7 (validated
  against temp 0/0.3, both of which collapse to a fixed point after 1-2
  rounds). `llm_client.make_matched_pair` gives baseline and
  execution-aware independently-constructed, identically-seeded clients
  bootstrapped from the same seed alpha, for an apples-to-apples
  comparison differing only in reward.
- **Turnover** (`factor_eval._pre_rebalance_weights`): drift-adjusted
  weight-based definition — today's target weight is compared against how
  yesterday's weight would have organically drifted given that day's
  return, not against yesterday's raw target, so passive price drift isn't
  double-counted as a trade.
- **Walk-forward optimization** (`src/walk_forward.py`): 21 folds, 5-year
  train / 1-year test, rolling 1 year, test windows 2006 through mid-2026.
  Each fold re-runs the full 50-round search from scratch on its own train
  window; the winning factor is frozen and scored once on the held-out
  test window; per-fold OOS streams are stitched into one continuous curve
  per (seed alpha, sampling seed, mode).
- **Robustness grid** (`scripts/run_full_grid.py`): 4 seed alphas from
  distinct economic categories (reversal, volume, volatility, momentum) x
  3 independent LLM sampling seeds (1001/2002/3003) = 12 independently
  re-optimized (seed-alpha, sampling-seed) combinations, ~38 hours total
  on a single local GPU pair.

## Environment

Developed and run in the `reccoai` conda environment.

```bash
conda create -n reccoai python=3.12   # if it doesn't already exist
conda run -n reccoai pip install -r requirements.txt
```

Requires internet access for the yfinance gap-fill and, for a real LLM
run, either an `ANTHROPIC_API_KEY` environment variable (`--llm claude`)
or a local GPU (`--llm local`, see below) — no key needed for `--llm mock`.

## Local LLM backend

The Generator & Refiner runs entirely on-box via
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llm_client.LocalQwenClient`),
using **Qwen2.5-Coder-14B-Instruct, Q4_K_M GGUF** (~9GB) — no API key, no
per-token cost, no network calls after the one-time download.

```bash
# 1. Install a CUDA-enabled llama-cpp-python build (the plain PyPI wheel is
#    CPU-only). Pick the extra-index matching your system CUDA version, e.g.
#    for CUDA 12.2 (check with `nvcc --version`):
conda run -n reccoai pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# 2. Download the model weights (one-time, ~9GB -> models/)
conda run -n reccoai python scripts/download_local_model.py

# 3. Run with --llm local (see "Running it" below)
```

Notes:
- **Hardware**: verified against 2x NVIDIA Titan X (Pascal, 12GB, compute
  capability 6.1). `make_matched_pair` pins baseline/execution-aware to
  separate GPUs (`main_gpu=0`/`main_gpu=1`) so two ~9GB model copies don't
  compete for one card.
- **Installing `llama-cpp-python` does not touch the NVIDIA driver, kernel
  modules, or system CUDA install** — it's a self-contained wheel.
- **Sampling**: `temperature=0.7`, `top_p=0.9`
  (`config.LOCAL_LLM_TEMPERATURE`/`LOCAL_LLM_TOP_P`). This is not an
  arbitrary default — full-trajectory inspection (not just tail sampling)
  showed both temperature 0 and temperature 0.3 collapse to a fixed point
  after round 1-2 and make no further progress for the rest of a 25-50
  round run; 0.7 was the lowest setting found to sustain genuine
  multi-round search. Consequently, sampling variance is characterized via
  three independent seeds at temperature 0.7 (the robustness grid) rather
  than relying on a lower, more deterministic temperature.
- **Seeding**: `LocalQwenClient` generates a fresh random seed per instance
  unless you pass `seed=<int>` explicitly. Do not rely on llama.cpp's usual
  "-1 = randomize" convention here — passing `seed=-1` to this build
  produced byte-identical output across separate process runs (verified),
  so it is **not** treated as "randomize from system time."
- Model loads once per `LocalQwenClient` instance and is reused for every
  round. `run_wfo_experiment`/`run_full_grid.py` construct the matched pair
  **once** before the fold loop and reuse it across all folds — constructing
  a fresh pair per fold reloads the ~9GB model repeatedly for no benefit and
  can crash with a CUDA OOM (see `walk_forward.py`'s docstring).

## Project layout

```
src/
  config.py         sample period, WFO_*, PIT_*, cost-model params, gamma1/gamma2, SEED_ALPHAS, paths
  pit_universe.py   point-in-time S&P membership + local-DB/yfinance price resolution + per-fold causal
                     liquidity reconstitution (select_fold_tickers, build_fold_universe, get_pit_panels)
  data_pipeline.py  legacy static-snapshot universe path (superseded by pit_universe for the paper's results)
  cost_engine.py    Corwin-Schultz spread estimator + sqrt market-impact model
  factor_eval.py    safe factor DSL (AST-whitelisted) + anti-pattern guardrails + evaluate_factor(...)
  llm_client.py     LLMClient interface: ClaudeClient / LocalQwenClient (llama.cpp) / MockLLMClient; make_matched_pair
  agent_loop.py     system prompt + generate->backtest->feedback loop, baseline vs execution-aware reward
  walk_forward.py   rolling WFO: generate_wfo_windows, run_wfo_experiment (matched-pair, per-fold liquidity
                     reconstitution, stitched OOS), evaluate_fixed_factor_across_folds (no-LLM control)
  experiment.py     single-split (70/30) path: cheaper sanity check, not the paper's primary methodology
  visualize.py      Figure 1 (architecture), Figure 2/3 small-multiples (cumulative return / bps sensitivity)
scripts/
  run_full_grid.py            PRIMARY: 4 seed alphas x 3 LLM seeds, full 21-fold WFO each, checkpointed per combo
  generate_paper_figures.py   builds paper Figures 2-3 from a completed full_grid run (no rerun needed)
  run_walk_forward.py         single (seed alpha, LLM seed) WFO run
  run_experiment.py           single-split experiment (cheaper sanity check)
  run_seed_sweep.py           earlier ad hoc multi-seed sweep (superseded by run_full_grid.py)
  download_local_model.py     one-time ~9GB GGUF download for --llm local
data/raw/                 point-in-time S&P membership CSV (committed, ~5MB)
data/processed_pit*/      cached point-in-time price panels (gitignored, regenerable via pit_universe.get_pit_panels)
models/                   local GGUF model weights (gitignored, not committed)
results/local_llm_runs/full_grid_2006_2026/   PRIMARY result set: grid_summary.csv + per-combo tables/logs/figures
results/figures/          paper's final Figures 1-3
paper/manuscript.md       FRL manuscript (source of truth; .docx generated from this)
paper/manuscript.docx     submission-ready Word export
paper/cover_letter.md     cover letter + suggested reviewers
```

## Running it

```bash
# 1. Build/cache the point-in-time panels (local DB if available + yfinance
#    gap-fill; ~1.5 min). Cached under data/processed_pit*/ after first run.
conda run -n reccoai python -c "from src.pit_universe import get_pit_panels; get_pit_panels()"

# 2. PRIMARY: the full robustness grid (4 seed alphas x 3 LLM seeds, 21-fold
#    WFO each, ~38 hours total on one local GPU pair; checkpointed per combo
#    to grid_summary.csv so a crash/interrupt loses at most one combo).
conda run -n reccoai python scripts/run_full_grid.py --seeds 1001 2002 3003 \
    --n-iterations 50 --alphas reversal volume volatility momentum

# 3. Regenerate the paper's Figures 2-3 from a completed grid run (fast,
#    reads already-saved CSVs, no rerun / GPU needed):
conda run -n reccoai python scripts/generate_paper_figures.py

# Cheaper checks:
# - one (seed alpha, LLM seed) WFO run instead of the full 12-combo grid:
conda run -n reccoai python scripts/run_walk_forward.py --llm local --n-iterations 50
# - single 70/30 split sanity check (not the paper's methodology):
conda run -n reccoai python scripts/run_experiment.py --llm mock --n-iterations 50
```

Primary outputs: `results/local_llm_runs/full_grid_2006_2026/grid_summary.csv`
(one row per mode x seed-alpha x LLM-seed — the numbers behind Table 1),
and per-combo `<alpha>/seed_<n>/table1_wfo.csv`, `wfo_folds.csv` (one row
per fold: winning factor + its own OOS stats), `net_returns_*.csv`,
`daily_turnover_*.csv`, and per-combo `fig2`/`fig3` PNGs/PDFs.

## Known limitations (see manuscript Section 4 for the full discussion)

- **Point-in-time price coverage is incomplete pre-2010** (roughly 75-90%
  of true constituents) — a data-availability constraint of free/quasi-free
  sources. Affects both reward conditions identically within each fold, so
  it should not bias the baseline-vs-execution-aware comparison, though it
  may affect absolute return levels in early folds.
- **Liquidity cap** (~150-220 tickers/fold) trades away some universe
  breadth for compute tractability (the full ~500-name point-in-time index
  makes `evaluate_factor` ~30x slower per call). A single full-universe
  confirmatory run is a natural follow-up robustness check, not yet run.
- **Cost model calibration** (`Y=0.5` impact coefficient) is not fit to
  realized execution data — the flat 5/10/20bps sensitivity columns exist
  because a single point estimate shouldn't be over-read.
- **Portfolio construction is deliberately unconstrained** (daily
  rebalancing, no no-trade band), so the factor expression itself (via
  `decay_linear`) is the only turnover-reduction lever available to the
  search.
