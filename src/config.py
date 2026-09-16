"""Central configuration for the execution-aware alpha mining pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
LOGS = RESULTS / "logs"

for _d in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, TABLES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe / sample period
# ---------------------------------------------------------------------------
LOOKBACK_YEARS = 10
UNIVERSE_NAME = "sp100"

# Fraction of the sample (by date, earliest-first) the LLM agent loop is
# allowed to see during search. The remaining tail is a held-out
# out-of-sample window used only for final reporting (Table 1 / Figure 2),
# to avoid reporting search-selection-biased in-sample performance.
# Used by the single-split path in experiment.py; the walk-forward path
# (walk_forward.py) is the primary, institutional-grade methodology and
# does not use this parameter.
TRAIN_FRACTION = 0.7

# ---------------------------------------------------------------------------
# Walk-forward optimization (rolling re-search + stitched OOS evaluation)
# ---------------------------------------------------------------------------
WFO_TRAIN_YEARS = 5     # in-sample search window length
WFO_TEST_YEARS = 1      # held-out test window length, immediately following train
WFO_STEP_YEARS = 1      # how far the whole (train, test) pair rolls forward each fold
WFO_MIN_TEST_DAYS = 20  # drop a trailing fold if its test window would be shorter than this

# ---------------------------------------------------------------------------
# Point-in-time universe (pit_universe.get_pit_panels)
# ---------------------------------------------------------------------------
# Raw price history is fetched from PIT_FETCH_START so rolling windows (up to
# 252 days) have real history the moment PIT_TRAIN_START's first WFO fold
# begins; the panel returned by get_pit_panels still starts at PIT_FETCH_START
# (kept for warmup), but WFO folds are anchored at PIT_TRAIN_START so the
# first test window lands in 2006 as intended (5y train immediately before).
PIT_FETCH_START = "1999-06-01"
PIT_TRAIN_START = "2001-01-01"

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
LONG_SHORT_GROSS_EXPOSURE = 2.0  # sum(|w|) target, i.e. 100% long / 100% short
PORTFOLIO_NAV = 100_000_000.0    # notional NAV used to translate weights -> $ trade sizes

# ---------------------------------------------------------------------------
# Cost engine
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostParams:
    """Parameters for the Corwin-Schultz spread + square-root impact cost model."""

    impact_coefficient: float = 0.5   # "Y" in Impact = Y * sigma * sqrt(participation)
    min_spread_bps: float = 1.0       # floor applied to estimated CS spread (bps, one-way)
    max_spread_bps: float = 100.0     # cap to guard against noisy CS estimates
    adv_lookback: int = 21            # trading days used for Average Daily (dollar) Volume


DEFAULT_COST_PARAMS = CostParams()

# Fixed flat-cost scenarios used for the Table 1 sensitivity sweep (one-way, bps of notional)
FLAT_COST_SCENARIOS_BPS = (5.0, 10.0, 20.0)

# ---------------------------------------------------------------------------
# LLM agent loop
# ---------------------------------------------------------------------------
N_ITERATIONS = 50
CLAUDE_MODEL = "claude-sonnet-5"

# Execution-aware penalty weights: reward = Gross_IR - gamma1*Turnover - gamma2*Cost_Impact
GAMMA_1 = 0.5   # turnover penalty (turnover expressed as annualized multiple, e.g. 4.0 = 400%/yr)
GAMMA_2 = 8.0   # cost-impact penalty (cost impact expressed as annualized return drag, e.g. 0.02 = 2%)

RANDOM_SEED = 42

# Common warm-start expression handed to BOTH baseline and execution-aware
# agents at round 1 (apples-to-apples: same seed, same LLM params, only the
# reward function differs from there). A minimal, canonical 1-day reversal —
# 4 operator nodes under count_ast_nodes.
SEED_ALPHA = "rank(-1 * delta(close, 1))"

# Three seed alphas spanning distinct economic categories, used for the
# seed-alpha robustness grid (scripts/run_full_grid.py): does the baseline-
# vs-execution-aware finding hold regardless of which mechanism the search
# is warm-started from, not just the short-term-reversal default above.
# All validated via factor_eval.check_anti_patterns (0 violations) and
# count_ast_nodes (4-5, comfortably under MAX_DSL_NODES).
SEED_ALPHAS = {
    "reversal": "rank(-1 * delta(close, 1))",                          # 1-day price reversal
    "volume": "rank(-1 * zscore(volume, 20))",                         # abnormal-volume fade
    "volatility": "rank(-1 * stddev(delta(close, 1), 20))",            # low-realized-volatility anomaly
    "momentum": "rank(delta(close, 60) / (stddev(delta(close, 1), 60) + 1e-5))",  # vol-scaled 60-day momentum
}

# Expression-complexity budget (see factor_eval.count_ast_nodes for the exact
# definition and calibration rationale). Applied as a bloat penalty to BOTH
# modes' reward equally, and reported/critiqued each round regardless of mode
# — parsimony is a general research-quality concern, not an execution-aware-
# specific one.
MAX_DSL_NODES = 8
NODE_BLOAT_PENALTY_WEIGHT = 0.05   # reward -= this * max(0, node_count - MAX_DSL_NODES)

# Narrative turnover target quoted in the execution-aware agent's task text
# (annualized multiple; 3.0 = 300%/yr) — guidance for the LLM to reason
# about concretely, not a hard constraint on the underlying scalar reward,
# which continues to use GAMMA_1/GAMMA_2 above.
EXECUTION_AWARE_TURNOVER_TARGET = 3.0

# ---------------------------------------------------------------------------
# Local LLM (llama.cpp / GGUF), for Generator & Refiner without API cost
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_REPO = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
LOCAL_MODEL_FILE = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"  # ~9.0GB, Q4_K_M quant
LOCAL_MODEL_PATH = MODELS_DIR / LOCAL_MODEL_FILE

LOCAL_LLM_N_GPU_LAYERS = -1     # -1 = offload every layer to GPU (fits one 12GB card w/ headroom)
LOCAL_LLM_N_CTX = 4096          # generous margin over our ~1-2k token system+user prompts
LOCAL_LLM_TEMPERATURE = 0.7     # nonzero so 50 rounds don't collapse to repeating one proposal
LOCAL_LLM_TOP_P = 0.9
LOCAL_LLM_MAX_TOKENS = 400      # response is a short JSON object, not prose
