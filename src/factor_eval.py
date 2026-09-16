"""Week 1 output: a small, safe factor expression DSL plus the standalone
`evaluate_factor(factor_code)` function that returns Gross_IR, Turnover,
Cost_Impact, and Net_IR.

Design choice: rather than exec()-ing arbitrary LLM-generated Python, factor
code is a single-line mathematical expression over a whitelisted set of
cross-sectional / time-series primitives (WorldQuant "fast expression"
style). This keeps LLM output auditable, keeps the search space aligned with
the strict-formatting system prompt in `agent_loop.py`, and avoids arbitrary
code execution risk: the expression is AST-validated (only whitelisted names,
calls, and arithmetic nodes) before it is compiled and eval'd against a
builtins-free namespace.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from . import config
from .cost_engine import CostEngine, flat_cost_drag
from .data_pipeline import get_panels

# ---------------------------------------------------------------------------
# DSL primitives. Panels are (date x ticker) DataFrames; cross-sectional ops
# act row-wise (axis=1), time-series ops act per-ticker (rolling, axis=0).
# ---------------------------------------------------------------------------

def rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)

def delay(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(int(n))

def delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x - x.shift(int(n))

def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).mean()

def ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).std()

def ts_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).sum()

def ts_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).min()

def ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).max()

def ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Percentile rank of the most recent value within its trailing n-day
    window. Uses pandas' native Cython rolling-rank rather than a per-window
    Python/scipy callback: numerically identical, ~250x faster (verified by
    comparing against scipy.stats.rankdata on this project's actual panel:
    max abs diff 0.0 over 2513x98 cells; 18.5s -> 0.07s)."""
    return x.rolling(int(n)).rank(pct=True)

def correlation(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).corr(y)

def covariance(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(int(n)).cov(y)

def scale(x: pd.DataFrame) -> pd.DataFrame:
    s = x.abs().sum(axis=1)
    return x.div(s.replace(0, np.nan), axis=0)

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def abs_(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()

def log_(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.clip(lower=1e-8))

def power(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.pow(n)

def _linear_decay(arr: np.ndarray) -> float:
    w = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.dot(arr, w) / w.sum())

def decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Linearly weighted moving average (more weight on recent days). The
    standard turnover-control lever in formulaic-alpha DSLs: smoothing a
    fast signal with decay_linear damps day-to-day rank churn while
    preserving most of the underlying signal."""
    return x.rolling(int(n)).apply(_linear_decay, raw=True)

def min_(x, y):
    return np.minimum(x, y)

def max_(x, y):
    return np.maximum(x, y)

def zscore(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling z-score: (x - trailing mean) / trailing std, epsilon-protected
    internally so the LLM never needs to hand-write the division guard."""
    mean = ts_mean(x, n)
    std = ts_std(x, n)
    return (x - mean) / (std + 1e-8)


ALLOWED_FUNCS = {
    "rank": rank, "delay": delay, "delta": delta, "ts_mean": ts_mean,
    "ts_std": ts_std, "ts_sum": ts_sum, "ts_min": ts_min, "ts_max": ts_max,
    "ts_rank": ts_rank, "correlation": correlation, "covariance": covariance,
    "scale": scale, "sign": sign, "abs": abs_, "log": log_, "power": power,
    "min": min_, "max": max_, "decay_linear": decay_linear,
    # Aliases matching the curated operator table shown to the LLM
    # (agent_loop.SYSTEM_PROMPT) — same implementations, added rather than
    # renamed so historical factor_code strings (manuscript results, logs,
    # MockLLMClient's template bank) using the original names still parse.
    "cs_rank": rank, "sma": ts_mean, "stddev": ts_std, "zscore": zscore,
}

# Functions where the lookback/window argument is expected to be a genuine
# multi-observation window (degenerate or undefined at n=1: ts_std(x,1) is
# NaN, decay_linear(x,1)==x, etc.) — used by check_anti_patterns' invalid-
# lookback rule. delta/delay are deliberately excluded: a 1-day change
# (delta(x, 1)) or 1-day lag is a meaningful, standard quantity, not a
# degenerate rolling statistic, so N=1 is legitimate there.
_WINDOWED_FUNCS = {
    "ts_mean", "sma", "ts_std", "stddev", "ts_sum", "ts_min", "ts_max",
    "ts_rank", "decay_linear", "correlation", "covariance", "zscore",
}
_RANK_FUNCS = {"rank", "cs_rank"}
_SMOOTHER_FUNCS = {"decay_linear", "sma", "ts_mean"}

ALLOWED_VARIABLES = {"close", "open", "high", "low", "volume", "returns", "vwap"}

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
    ast.UAdd, ast.Mod, ast.Tuple, ast.Load,
)


class InvalidFactorCode(ValueError):
    pass


def validate_factor_code(code: str) -> ast.AST:
    """AST-whitelist a factor expression before it is ever compiled/eval'd.

    Rejects attribute access, subscripts, comprehensions, lambdas, imports,
    and any identifier that isn't a known variable or whitelisted function.
    This is what lets us safely eval() LLM-generated code without a sandbox
    process.
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise InvalidFactorCode(f"factor_code is not valid syntax: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise InvalidFactorCode(f"Disallowed syntax element: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in (ALLOWED_VARIABLES | ALLOWED_FUNCS.keys()):
            raise InvalidFactorCode(f"Unknown identifier '{node.id}'")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise InvalidFactorCode("Only whitelisted functions may be called")
    return tree


def count_ast_nodes(factor_code: str) -> int:
    """Complexity metric used for the bloat penalty and the agent's
    "Node Count" diagnostic. Counts operator nodes only (Call, BinOp,
    UnaryOp) — not bare variable/constant leaves — so it tracks how many
    operations are chained/nested, which is what actually drives both
    expression bloat and (usually) turnover, rather than penalizing a
    factor for referencing more variables or numeric literals per se.

    Calibration note: `rank(decay_linear(delta(close, 20) / (ts_std(returns,
    20) + 0.0001), 60))` — one of the better factors found in this project's
    own experiments — has 6 operator nodes under this definition (rank,
    decay_linear, delta, ts_std, Div, Add). config.MAX_DSL_NODES is set
    slightly above that so legitimate multi-primitive factors aren't
    penalized, while runaway N-term chains (bloat observed in some LLM runs)
    still are.
    """
    tree = ast.parse(factor_code, mode="eval")
    return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp)))


def ts_rank_wraps_decay_linear(factor_code: str) -> bool:
    """True if any `ts_rank(...)` call has a `decay_linear(...)` call
    nested somewhere inside its arguments — i.e. ts_rank is the outer
    operator around an already-smoothed signal. Taking a rolling
    percentile rank of a smoothed input reintroduces day-to-day churn and
    tends to cancel most of decay_linear's turnover reduction (observed
    empirically in this project: this exact pattern was the losing
    execution-aware factor in one local-LLM run). Used by the diagnostic
    critique generator, not by validation — the expression is still legal
    DSL, just usually a bad idea."""
    tree = ast.parse(factor_code, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ts_rank":
            for inner in ast.walk(node):
                if inner is not node and isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) \
                        and inner.func.id == "decay_linear":
                    return True
    return False


def _call_name(node: ast.AST) -> str | None:
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def _is_positive_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value > 0


def _denominator_is_protected(node: ast.AST) -> bool:
    """A division's denominator is considered safe if it's a nonzero literal
    (can't be data-dependent-zero), an epsilon-added expression (`x + eps`
    or `eps + x`, eps > 0), or itself a ranked value (rank/cs_rank/ts_rank
    output, which is bounded away from a hard zero by construction in this
    DSL's percentile-rank implementations)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value != 0:
        return True
    if _call_name(node) in (_RANK_FUNCS | {"ts_rank"}):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_positive_constant(node.left) or _is_positive_constant(node.right)
    return False


def check_anti_patterns(factor_code: str) -> list[str]:
    """Hard guardrails enforced on every LLM proposal (agent_loop rejects a
    candidate outright if this returns any violations — see
    "[DISALLOWED ANTI-PATTERNS & AST GUARDRAILS]" in SYSTEM_PROMPT):

      1. Redundant ranking: rank(rank(X)) / cs_rank(cs_rank(X)) / ts_rank(ts_rank(X,N),N)
         — checked as *direct* nesting only (a rank call's immediate argument
         is itself a same-family rank call), so a rank of a larger expression
         that merely *contains* a rank sub-term elsewhere (e.g.
         rank(f(X) * rank(Y)), combining two distinct ranked signals) is not
         flagged — that combination is meaningful, not redundant.
      2. Trivial scalars: a bare positive-constant Add/Mult wrapping the
         WHOLE expression (e.g. `rank(X) + 5`, `2 * rank(X)`), checked only
         at the outermost node so this doesn't collide with rule 4's
         required epsilon-protected divisions elsewhere in the tree. Sign
         flips (`-1 * X`) are exempt: negation reverses long/short direction,
         which is meaningful, unlike a positive affine rescale of a rank.
      3. Cascading smoothers: more than one of {decay_linear, sma/ts_mean}
         directly nested inside one another.
      4. Unprotected division: every `/` must have a protected denominator
         (see `_denominator_is_protected`).
      5. Unbounded powers: `power(x, n)` or `x ** n` with n > 2.
      6. Invalid lookbacks: the lookback argument of any windowed function
         (`_WINDOWED_FUNCS` — deliberately excludes delta/delay, see that
         set's docstring) must be a literal integer with 2 <= N <= 252.

    Returns a list of human-readable violation descriptions; empty = passes.
    Call after `validate_factor_code` (assumes the expression is already
    known to be syntactically legal DSL).
    """
    tree = ast.parse(factor_code, mode="eval")
    violations: list[str] = []

    # Rule 2: trivial scalar wrapping the outermost node only.
    top = tree.body
    if isinstance(top, ast.BinOp) and isinstance(top.op, (ast.Add, ast.Mult)):
        if _is_positive_constant(top.left) or _is_positive_constant(top.right):
            violations.append(
                "TRIVIAL_SCALARS: a positive constant is added/multiplied onto the entire expression "
                "(e.g. rank(X) + 5 or 2 * rank(X)) — cross-sectional sorting is invariant to this, so it's a no-op."
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)

        # Rule 1: redundant ranking (direct nesting only).
        if name in _RANK_FUNCS and node.args and _call_name(node.args[0]) in _RANK_FUNCS:
            violations.append(f"REDUNDANT_RANKING: {name}({_call_name(node.args[0])}(...)) directly "
                               "re-ranks an already-ranked signal.")
        if name == "ts_rank" and node.args and _call_name(node.args[0]) == "ts_rank":
            violations.append("REDUNDANT_RANKING: ts_rank(ts_rank(...), N) directly re-ranks an "
                               "already-ranked signal.")

        # Rule 3: cascading smoothers (direct nesting only).
        if name in _SMOOTHER_FUNCS and node.args and _call_name(node.args[0]) in _SMOOTHER_FUNCS:
            violations.append(f"CASCADING_SMOOTHERS: {name}({_call_name(node.args[0])}(...)) combines two "
                               "smoothing operators — use only one.")

        # Rule 5: unbounded powers.
        if name == "power" and len(node.args) == 2 and isinstance(node.args[1], ast.Constant):
            if isinstance(node.args[1].value, (int, float)) and node.args[1].value > 2:
                violations.append(f"UNBOUNDED_POWERS: power(x, {node.args[1].value}) exceeds the max exponent of 2.")

        # Rule 6: invalid lookbacks.
        if name in _WINDOWED_FUNCS and node.args:
            n_arg = node.args[-1]
            if isinstance(n_arg, ast.Constant) and isinstance(n_arg.value, (int, float)):
                if not (isinstance(n_arg.value, int) or n_arg.value.is_integer()) or not (2 <= n_arg.value <= 252):
                    violations.append(f"INVALID_LOOKBACKS: {name}(..., {n_arg.value}) — lookback must be an "
                                       "integer with 2 <= N <= 252.")
            else:
                violations.append(f"INVALID_LOOKBACKS: {name}'s lookback argument must be a literal integer.")

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not _denominator_is_protected(node.right):
                violations.append("UNPROTECTED_DIVISION: a division's denominator is not epsilon-protected "
                                   "(x / (y + 1e-5)) or a ranked value — it could be zero.")

    return violations


def compute_factor_scores(factor_code: str, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tree = validate_factor_code(factor_code)
    namespace = dict(ALLOWED_FUNCS)
    namespace.update({
        "close": panels["close"], "open": panels["open"], "high": panels["high"],
        "low": panels["low"], "volume": panels["volume"], "returns": panels["returns"],
        "vwap": (panels["high"] + panels["low"] + panels["close"]) / 3.0,
    })
    code_obj = compile(tree, "<factor_code>", "eval")
    result = eval(code_obj, {"__builtins__": {}}, namespace)  # noqa: S307 - AST-whitelisted above
    if not isinstance(result, pd.DataFrame):
        raise InvalidFactorCode("factor_code must evaluate to a (date x ticker) DataFrame")
    return result


def scores_to_weights(scores: pd.DataFrame, gross_exposure: float = config.LONG_SHORT_GROSS_EXPOSURE) -> pd.DataFrame:
    """Dollar-neutral, rank-demeaned long/short weights normalized to a fixed
    gross exposure (default 2.0 = 100% long + 100% short)."""
    demeaned = scores.sub(scores.mean(axis=1), axis=0)
    abs_sum = demeaned.abs().sum(axis=1).replace(0, np.nan)
    weights = demeaned.div(abs_sum, axis=0) * gross_exposure
    return weights.fillna(0.0)


def _pre_rebalance_weights(
    prior_weights: pd.DataFrame, prior_returns: pd.DataFrame, gross_exposure: float,
) -> pd.DataFrame:
    """How yesterday's held weights would have organically drifted due to
    that day's price returns, immediately before today's rebalance trade --
    Method 1 (Normalized Weight-Based Turnover), Step 1:
    w_{i,t-} = w_{i,t-1}(1+r_{i,t}) / sum_j w_{j,t-1}(1+r_{j,t}).

    Adapted for a dollar-neutral long/short book: the reference formula
    normalizes by net exposure (sum w_j), which is ~0 here by construction
    (longs and shorts cancel) and would blow up on division; we normalize by
    gross exposure (sum |w_j|) instead, preserving the same gross-exposure
    invariant `scores_to_weights` targets. This is what makes comparing
    today's target weight against this drifted (not yesterday's raw target)
    weight the correct measure of how much actually needs to be traded."""
    drifted = prior_weights * (1 + prior_returns)
    norm = drifted.abs().sum(axis=1).replace(0, np.nan)
    return drifted.div(norm, axis=0) * gross_exposure


def annualized_ir(returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return float(dd.min())


@dataclass
class FactorEvalResult:
    factor_code: str
    Gross_IR: float
    Turnover: float
    Cost_Impact: float
    Net_IR: float
    max_drawdown: float
    cumulative_return: float
    gross_returns: pd.Series = field(repr=False)
    net_returns: pd.Series = field(repr=False)
    daily_turnover: pd.Series = field(repr=False)


def evaluate_factor(
    factor_code: str,
    panels: dict[str, pd.DataFrame] | None = None,
    cost_params: config.CostParams | None = None,
    cost_model: str = "full",
    flat_bps: float | None = None,
    gross_exposure: float = config.LONG_SHORT_GROSS_EXPOSURE,
    nav: float = config.PORTFOLIO_NAV,
    eval_window: tuple[pd.Timestamp | str | None, pd.Timestamp | str | None] | None = None,
) -> FactorEvalResult:
    """Standalone factor evaluation harness (Week 1 deliverable).

    Parameters
    ----------
    factor_code : a single-line expression in the whitelisted DSL, e.g.
        "rank(-1 * delta(close, 5) / (ts_std(returns, 20) + 0.0001))"
    cost_model : "full" uses the Corwin-Schultz spread + square-root impact
        model (CostEngine); "flat" applies a fixed `flat_bps` one-way cost.
    eval_window : optional (start, end) date bounds (either side may be
        None for open-ended) restricting which dates' returns are summarized
        into Gross_IR/Turnover/Cost_Impact/Net_IR etc. Scores/weights are
        still computed causally over the *entire* panel (no lookahead either
        way), so this only controls the reporting window — e.g. pass
        (None, train_end) to keep an LLM search loop from ever seeing
        held-out performance, and (train_end, None) to score the held-out
        period only once search is over.

    Returns
    -------
    FactorEvalResult with Gross_IR, Turnover, Cost_Impact, Net_IR plus
    diagnostics (max drawdown, cumulative return, and the underlying return
    series for plotting).
    """
    if cost_model not in ("full", "flat"):
        raise ValueError("cost_model must be 'full' or 'flat'")
    if cost_model == "flat" and flat_bps is None:
        raise ValueError("flat_bps is required when cost_model='flat'")

    panels = panels if panels is not None else get_panels()
    scores = compute_factor_scores(factor_code, panels)
    eligible = panels.get("eligible")
    if eligible is not None:
        # Point-in-time universe (pit_universe.get_pit_panels): mask scores
        # to only the tickers that were actual index constituents on that
        # date, even though their raw price history (used above for rolling
        # computations) extends outside their membership window. scores_to_
        # weights below demeans/normalizes with skipna=True and fillna(0.0)
        # at the end, so a NaN'd-out score here correctly becomes zero
        # weight rather than participating in the cross-sectional rank.
        scores = scores.where(eligible.reindex_like(scores).fillna(False))
    weights = scores_to_weights(scores, gross_exposure)

    weights_lagged = weights.shift(1).fillna(0.0)
    gross_ret = (weights_lagged * panels["returns"]).sum(axis=1)

    # Method 1 (Normalized Weight-Based Turnover): compare today's target
    # weight against yesterday's weight *drifted* by that day's returns
    # (the pre-rebalance weight), not against yesterday's raw target --
    # otherwise passive price drift gets double-counted as if it were a
    # trade. weight_changes (full, per-stock, un-halved) is what actually
    # gets traded and is what the cost engine prices; daily_turnover halves
    # the portfolio-level sum to report *one-way* turnover (buys and sells
    # are equal in dollar terms for a dollar-neutral book, so summing both
    # legs without halving double-counts every rebalance).
    prior_weights = weights_lagged.shift(1).fillna(0.0)
    prior_returns = panels["returns"].shift(1).fillna(0.0)
    pre_rebalance = _pre_rebalance_weights(prior_weights, prior_returns, gross_exposure)
    weight_changes = (weights_lagged - pre_rebalance).fillna(weights_lagged)
    daily_turnover = 0.5 * weight_changes.abs().sum(axis=1)

    if cost_model == "full":
        engine = CostEngine(panels, cost_params or config.DEFAULT_COST_PARAMS)
        daily_cost_drag, _spread_drag, _impact_drag = engine.cost_drag(weight_changes, nav=nav)
    else:
        daily_cost_drag = flat_cost_drag(weight_changes, flat_bps)

    net_ret = gross_ret - daily_cost_drag

    valid = gross_ret.index[20:]  # burn in the rolling-window warmup period
    if eval_window is not None:
        start, end = eval_window
        if start is not None:
            valid = valid[valid >= pd.Timestamp(start)]
        if end is not None:
            valid = valid[valid <= pd.Timestamp(end)]
        if len(valid) < 20:
            raise InvalidFactorCode(f"eval_window leaves only {len(valid)} usable observations")
    gross_ret_v, net_ret_v, turnover_v, cost_v = (
        gross_ret.loc[valid], net_ret.loc[valid], daily_turnover.loc[valid], daily_cost_drag.loc[valid]
    )

    return FactorEvalResult(
        factor_code=factor_code,
        Gross_IR=annualized_ir(gross_ret_v),
        Turnover=float(turnover_v.mean() * config.TRADING_DAYS_PER_YEAR),
        Cost_Impact=float(cost_v.mean() * config.TRADING_DAYS_PER_YEAR),
        Net_IR=annualized_ir(net_ret_v),
        max_drawdown=max_drawdown(net_ret_v),
        cumulative_return=float((1 + net_ret_v).prod() - 1),
        gross_returns=gross_ret_v,
        net_returns=net_ret_v,
        daily_turnover=turnover_v,
    )
