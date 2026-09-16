# Execution-Aware Alpha Mining

Code accompanying a 2,500-word FRL empirical note on execution-aware LLM
factor mining: an LLM agent proposes daily cross-sectional equity factors,
each candidate is backtested and priced for realistic transaction costs
(Corwin-Schultz spread + square-root market impact), and the resulting
diagnostics are fed back to the agent each round. We compare an agent
rewarded on **raw returns only** ("baseline") against one rewarded on
**Gross_IR net of a turnover/cost penalty** ("execution-aware").

The primary methodology is **rolling walk-forward optimization (WFO)**
(`src/walk_forward.py`, `scripts/run_walk_forward.py`): rather than a single
train/test split, the (train, test) window rolls forward across the sample.
On each fold the agent re-runs its *entire* 50-round search from scratch,
seeing only that fold's in-sample diagnostics; the fold's winning factor is
then frozen and scored, unmodified, on that fold's own held-out test window.
The per-fold OOS return streams are concatenated in date order into one
continuous, stitched out-of-sample equity curve per mode — this is what
Table 1 (`table1_wfo.csv`) and Figure 2 report. A lighter single-split
70/30 path also exists (`src/experiment.py`, `scripts/run_experiment.py`,
`config.TRAIN_FRACTION`) as a cheaper sanity check, but WFO is the
institutional-grade result: picking the best of many candidates scored
against the same data used to report performance is a textbook source of
backtest-overfitting bias, and a single static split is still only one draw
of "how favorable was this particular out-of-sample window" — walk-forward
averages that away across several independent re-searches and holdouts.
See Section 2.4 of `paper/manuscript.md`.

## Environment

Developed and run in the `reccoai` conda environment.

```bash
conda create -n reccoai python=3.12   # if it doesn't already exist
conda run -n reccoai pip install -r requirements.txt
```

Requires internet access for the first data pull (yfinance) and, for a real
LLM run, either an `ANTHROPIC_API_KEY` environment variable (`--llm claude`)
or a local GPU (`--llm local`, see below) — no key needed for `--llm mock`.

## Local LLM backend

The Generator & Refiner can run entirely on-box via
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llm_client.LocalQwenClient`),
using **Qwen2.5-Coder-14B-Instruct, Q4_K_M GGUF** (~9GB) — no API key, no
per-token cost, no network calls after the one-time download. This is the
recommended way to get genuine LLM-generated (not mock, not paid-API)
results for the paper.

```bash
# 1. Install a CUDA-enabled llama-cpp-python build (the plain PyPI wheel is
#    CPU-only). Pick the extra-index matching your system CUDA version, e.g.
#    for CUDA 12.2 (check with `nvcc --version`):
conda run -n reccoai pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# 2. Download the model weights (one-time, ~9GB -> models/)
conda run -n reccoai python scripts/download_local_model.py

# 3. Run with --llm local instead of --llm mock / --llm claude
conda run -n reccoai python scripts/run_walk_forward.py --llm local --n-iterations 50
conda run -n reccoai python scripts/run_experiment.py --llm local --n-iterations 50
```

Notes:
- **Hardware**: this was set up and verified against 2x NVIDIA Titan X
  (Pascal, 12GB, compute capability 6.1) — llama.cpp's CUDA backend supports
  Pascal well (no tensor cores needed for GGUF's quantized GEMV/GEMM
  kernels), unlike AWQ/GPTQ fast-kernel paths (marlin/exllama), which
  generally target Ampere+ and were a worse fit for this hardware — hence
  GGUF/llama.cpp over the AWQ alternative. The Q4_K_M weights need ~9GB VRAM
  and fit on a single 12GB card with headroom (`config.LOCAL_LLM_N_GPU_LAYERS
  = -1` offloads every layer to GPU 0 by default); pass
  `tensor_split=[0.5, 0.5]` to `LocalQwenClient`/`make_client("local", ...)`
  to split across both GPUs instead if you'd rather balance load or run
  something else on GPU 0 concurrently — not required for VRAM headroom at
  this model size, just an option.
- **Installing `llama-cpp-python` does not touch the NVIDIA driver, kernel
  modules, or system CUDA install** — it's a self-contained wheel in the
  conda env's site-packages, no root required, verified against
  `/usr/local/cuda` (12.2) and driver 570.195 in this environment.
- **Chat template**: `LocalQwenClient` leaves `chat_format` unset so
  llama.cpp uses the ChatML template embedded in the GGUF's own metadata
  (`tokenizer.chat_template`), rather than a hardcoded template that could
  drift from what the checkpoint actually expects.
- **Speed (measured on the 2x Titan X Pascal setup above)**: ~2.7s one-time
  model load, ~4.4s/round thereafter for a single-turn factor proposal
  (short prompts, `LOCAL_LLM_MAX_TOKENS=400` response cap). Extrapolates to
  roughly 7 minutes for a full single-split run (50 iters x 2 modes = 100
  calls) and roughly 45 minutes for a full 6-fold walk-forward run (600
  search calls; the OOS scoring pass afterward is pure backtest, not LLM
  calls, so adds negligible time). Pascal has no tensor cores, so this is
  well below what the same quant would get on Ampere+ — measure your own
  hardware with a short smoke test (a handful of iterations) if different.
- **Sampling**: `temperature=0.7`, `top_p=0.9` by default
  (`config.LOCAL_LLM_TEMPERATURE`/`LOCAL_LLM_TOP_P`) so 50 rounds actually
  explore rather than repeatedly regenerating one proposal; lower for more
  reproducible (but less exploratory) runs.
- **Seeding**: `LocalQwenClient` generates a fresh random seed per instance
  (printed at construction, e.g. `[LocalQwenClient] seed=...`) unless you
  pass `seed=<int>` explicitly. Do not rely on llama.cpp's usual "-1 =
  randomize" convention here — passing `seed=-1` to this build/binding
  produced byte-identical output across two separate process runs (verified
  empirically), so it was **not** treated as "randomize from system time" as
  it is in some other llama.cpp builds. Pass an explicit `seed` for
  reproducible runs; omit it for genuine run-to-run variation.
- Model loads once per `LocalQwenClient` instance and is reused for every
  round — instantiate one client per process (as `make_client("local")`
  does) rather than one per call.
- **Qualitatively different search behavior than the mock**: in a smoke
  test, Qwen tended to iteratively *extend* its own previous factor
  expression each round (e.g. multiplying in an additional term) rather
  than the mock's templated mutate-or-explore, and did not monotonically
  improve reward round over round — genuinely LLM-driven, not template
  hill-climbing. Worth keeping in mind when comparing local-model results
  against the mock/Claude runs reported in the manuscript.
- **High run-to-run variance in single-split local-LLM runs — not yet
  resolved, flagged for follow-up**: two genuinely different-seed 50-round
  single-split runs (`--llm local`) gave *opposite* conclusions. One seed's
  execution-aware agent got stuck nesting `ts_rank` around an
  already-`decay_linear`-smoothed signal (`ts_rank(decay_linear(close -
  vwap, 20), 20)`) — which reintroduces the churn `decay_linear` was
  supposed to remove — and lost badly to the baseline (turnover 25,535%
  vs. 12,046%/yr, net Sharpe −5.18 vs. −0.67). A second seed's
  execution-aware agent instead found a genuinely low-turnover factor
  (`rank(decay_linear((high - low) / ts_mean(volume, 20), 5))`) and beat
  the baseline decisively (turnover 3,663% vs. 49,035%/yr, net Sharpe −2.50
  vs. −8.49). With only 50 rounds, a 14B local model's outcome is not yet
  stable across sampling seeds — the *same* methodological point the paper
  makes about single train/test splits (Section 2.4) applies here too, to
  the LLM's own sampling noise. The walk-forward script's 6 independent
  fold-searches (each fold's search naturally uses a different point in the
  client's sampling stream, unaffected by the seed bug above since it's one
  process/client throughout) are the right tool to get a decision-relevant
  read on this with `--llm local`, but that full run (~45 min) has not yet
  been done — `results/local_llm_runs/run1/` and the `run3` log referenced
  above are the two single-split data points this note is based on, kept
  for reference rather than folded into the manuscript.

## Project layout

```
src/
  config.py         sample period, WFO_*/TRAIN_FRACTION, cost-model params, gamma1/gamma2, paths
  data_pipeline.py  S&P 100 snapshot list + yfinance download -> cached parquet panels; train_test_split_date()
  cost_engine.py    Corwin-Schultz spread estimator + sqrt market-impact model
  factor_eval.py    safe factor DSL (AST-whitelisted) + evaluate_factor(factor_code, eval_window=...)
  llm_client.py     LLMClient interface: ClaudeClient (Anthropic API) / LocalQwenClient (llama.cpp, local GPU) / MockLLMClient
  agent_loop.py     system prompt + generate->backtest->feedback loop restricted to eval_window, baseline vs execution-aware reward
  walk_forward.py   rolling WFO: generate_wfo_windows, run_walk_forward (per-fold search+OOS score), stitch_folds, run_wfo_experiment
  experiment.py     single-split (70/30) path: builds Table 1 (out-of-sample) + table_in_sample, sweeps 5/10/20bps flat-cost scenarios
  visualize.py      Figure 1 (architecture) and Figure 2 (cumulative net-of-cost returns; optional fold-boundary markers)
scripts/
  run_data_pipeline.py     download/cache the panels
  download_local_model.py  one-time ~9GB GGUF download for --llm local
  run_walk_forward.py      primary experiment: rolling WFO across all folds + stitched figures/tables
  run_experiment.py        single-split experiment (cheaper sanity check)
data/processed/          cached parquet panels (open/high/low/close/volume/returns/dollar_volume)
models/                  local GGUF model weights (downloaded, not committed)
results/{tables,figures,logs}/   experiment outputs
paper/manuscript.md      FRL manuscript draft
```

## Running it

```bash
# 1. Build/cache the S&P 100 panels (10y daily, auto_adjust=True)
conda run -n reccoai python scripts/run_data_pipeline.py

# 2a. Primary experiment: rolling walk-forward (6 folds at the defaults below:
#     4y train / 1y test / 1y step over the 10y sample). --llm mock needs no
#     API key and is fully deterministic (seeded); --llm claude calls the
#     Anthropic API (claude-sonnet-5 by default, config.CLAUDE_MODEL) and
#     requires ANTHROPIC_API_KEY; --llm local runs Qwen2.5-Coder-14B-Instruct
#     on your own GPU (see "Local LLM backend" above) — no key, no per-call
#     cost, but slower per round than a hosted API. Runs the full 50-round
#     search independently on every fold for both modes (~600+ backtests) —
#     a few minutes on mock, much longer on claude/local.
conda run -n reccoai python scripts/run_walk_forward.py --llm mock --n-iterations 50
conda run -n reccoai python scripts/run_walk_forward.py --llm claude --n-iterations 50
conda run -n reccoai python scripts/run_walk_forward.py --llm local --n-iterations 50
conda run -n reccoai python scripts/run_walk_forward.py --llm mock --train-years 4 --test-years 1 --step-years 1

# 2b. Cheaper single-split sanity check (one 70/30 split, not walk-forward):
conda run -n reccoai python scripts/run_experiment.py --llm mock --n-iterations 50
```

Walk-forward outputs: `results/tables/table1_wfo.{csv,md}` (stitched
out-of-sample performance — the headline numbers), `results/tables/wfo_folds.{csv,md}`
(one row per mode x fold: the winning factor and its own-fold OOS stats, so
you can see how the chosen expression evolves as the window rolls),
`results/figures/fig1_architecture.{png,pdf}`,
`results/figures/fig2_cumulative_returns.{png,pdf}` (stitched OOS curve with
refit-date markers), and per-fold search logs in
`results/logs/{baseline,execution_aware}_fold{N}_log.csv`.

Single-split outputs: `results/tables/table1.{csv,md}` (out-of-sample) and
`results/tables/table1_in_sample.{csv,md}` (same factors scored on the
search window, so the in-sample/out-of-sample gap — the degree of search
overfitting — is visible), plus `results/logs/{baseline,execution_aware}_log.csv`.

## Design notes / limitations (for the manuscript's Data & Methods / Limitations)

- **Universe**: a fixed *current* S&P 100 snapshot (`data_pipeline.SP100_TICKERS`),
  not point-in-time historical membership — introduces mild survivorship bias.
  98/101 names had a full 10y history via yfinance; 3 were dropped.
- **Prices**: yfinance `auto_adjust=True` (splits + dividends), so OHLC used in
  the Corwin-Schultz estimator is total-return adjusted, not raw traded prices —
  standard simplification for a short empirical note, noted as a limitation.
- **Factor DSL**: LLM output is restricted to a whitelisted "fast expression"
  grammar (rank/delay/delta/ts_mean/ts_std/.../decay_linear) validated by AST
  inspection before eval — this is what makes the strict-formatting system
  prompt enforceable and keeps LLM-generated code safe to execute without a
  sandboxed subprocess.
- **Cost model**: Corwin & Schultz (2012) two-day high-low spread estimator
  (base version, no overnight-return adjustment) for the bid-ask component,
  plus a Grinold-Kahn-style square-root impact model
  `impact = Y * sigma_daily * sqrt(participation)` calibrated with
  `Y = config.CostParams.impact_coefficient = 0.5`. Flat 5/10/20bps scenarios
  are provided alongside the full model for Table 1's sensitivity sweep.
- **Portfolio construction**: daily-rebalanced, rank-demeaned, dollar-neutral
  long/short at fixed gross exposure (200% = 100% long + 100% short); this
  is deliberately naive (no turnover throttling beyond what the factor
  expression itself encodes, e.g. `decay_linear`) so that the LLM agent's
  choice of turnover-reducing DSL primitives is the only lever — this is the
  mechanism the paper studies.
- **Mock LLM**: `MockLLMClient` is a template-mutation search that reads the
  same diagnostic feedback text a real LLM would read and hill-climbs on
  reward. It is used to validate the full pipeline end-to-end without API
  cost; the manuscript reports mock-run numbers and should be re-run with
  `--llm claude` before submission if a genuine LLM-generated result set is
  required.
- **Walk-forward windowing**: folds are anchored at the start of the sample
  and rolled forward by `WFO_STEP_YEARS`; a trailing fold is dropped if its
  test window would have fewer than `WFO_MIN_TEST_DAYS` trading days. Because
  every DSL primitive is causal (rolling/lagged, no forward-looking
  operations), factor scores are always computed over the *entire* panel and
  only the reporting window is restricted via `eval_window`, so there's no
  rolling-window warm-up artifact at a fold boundary — a fold's test-window
  turnover on day 1 is a same-factor diff against the last in-sample day, not
  an artifact of truncating history.
- **Fold independence and the matched client pair**: `run_wfo_experiment`
  constructs ONE matched pair of clients (`llm_client.make_matched_pair` —
  baseline and execution-aware each get their own independently-constructed,
  identically-seeded instance) *once*, before the fold loop, and reuses it
  across all folds — not a fresh pair per fold. Besides being simpler, this
  is required for `--llm local`: constructing fresh clients inside the fold
  loop reloads the ~9GB model 12 times for no benefit and can crash with a
  CUDA OOM (Python evaluates the new pair's model loads before releasing the
  previous fold's clients, so two folds' models are briefly resident on the
  same GPU at once — hit this in practice, see `walk_forward.py`'s
  docstring for the full mechanics). Reusing one pair also means baseline
  and execution-aware stay synchronized round-for-round across the *entire*
  run, not just within one fold, since both are called the same number of
  times per fold in lockstep. Folds are still not i.i.d. replicates of each
  other in the strict statistical sense (each fold's search is seeded by
  wherever its client's sampling stream happened to be after the previous
  fold, not freshly randomized) — different folds do pick different winning
  factors regardless, driven mainly by each fold's different training data,
  and the stitched curve is a realistic simulation of periodically
  re-optimizing and trading whatever the current model says, not a single
  fixed factor held for 10 years.
- **Single-split path**: `src/experiment.py` / `config.TRAIN_FRACTION` still
  exists as a cheaper, single-draw 70/30 alternative to WFO, useful for quick
  iteration on the harness itself; it should not be treated as a substitute
  for the walk-forward result in the manuscript.

## Reproducing the numbers in the current manuscript draft

`paper/manuscript.md` reports the `scripts/run_walk_forward.py --llm mock
--n-iterations 50` run committed in `results/` (defaults: 4y train / 1y test
/ 1y step, giving 6 folds with test windows 2020-08-11 through 2026-08-10).
The single-split numbers cited as a secondary check in Section 2.4/3 come
from the earlier `scripts/run_experiment.py --llm mock --n-iterations 50`
run, in-sample window through 2023-08-09, out-of-sample from 2023-08-10.
Re-running `scripts/run_data_pipeline.py` pulls fresh data (yfinance
snapshot is not pinned to a date), so exact figures will drift slightly on
re-download; re-running with `--llm claude` will differ more substantially
since it exercises a real model rather than the fixed template bank.

**Performance note**: the committed WFO run took roughly 20-25 minutes
because it used the original `ts_rank` implementation,
`rolling(n).apply(scipy.stats.rankdata, raw=True)` (~19.5s per call on the
full panel, hit repeatedly since two of the mock's templates use `ts_rank`).
It has since been replaced with pandas' native `rolling(n).rank(pct=True)`
— verified numerically identical (max abs diff 0.0 across the full panel)
and ~260x faster (18.5s -> 0.07s for the same call) — so the *results*
above are unaffected (same numbers), but re-running from the current code
should take roughly 5 minutes instead.
