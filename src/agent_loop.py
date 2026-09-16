"""Week 2 deliverable: the LLM agent loop.

Prompt architecture (static system prompt + dynamic diagnostic payload):

  STATIC SYSTEM PROMPT (fixed across all rounds): role framing, DSL grammar
  (allowed variables/functions/operators), the decay_linear-vs-ts_rank
  composition rule, and the expression-complexity (node) budget.

  DYNAMIC DIAGNOSTIC PAYLOAD (rebuilt every round): the best factor found so
  far and its metrics, the most recently proposed factor and its metrics, a
  rule-based structural critique of that proposal (why it fell short, with
  the corresponding fix folded into the same sentence -- see `_diagnose` /
  `generate_critique`), and the reward definition for this run's mode.

Apples-to-apples baseline vs. execution-aware: both modes are bootstrapped
from the IDENTICAL seed expression `config.SEED_ALPHA`, evaluated once before
round 1, so round 1's "best factor so far" is the same formula with the same
metrics for both modes -- the two searches diverge only because of which
reward function shapes what gets kept and refined from there. Pass each mode
an independently-constructed LLM client seeded identically (see
`llm_client.make_matched_pair`) so round-to-round sampling isn't a shared,
continuing stream between the two runs either.

Two modes control only the *reward* the search is steered by (Week 3
baseline-vs-proposed comparison):
  - "baseline":         reward = Gross_IR                      (raw returns only)
  - "execution_aware":  reward = Gross_IR - gamma1*Turnover - gamma2*Cost_Impact
Both rewards subtract the same bloat penalty above the node budget.
Diagnostics (including Net_IR/Turnover/Cost_Impact) are always computed and
logged for both modes so Table 1 can compare them on equal footing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .factor_eval import (
    FactorEvalResult,
    InvalidFactorCode,
    check_anti_patterns,
    count_ast_nodes,
    evaluate_factor,
    ts_rank_wraps_decay_linear,
)
from .llm_client import LLMClient

# Curated operator table and variable list shown to the LLM. This is
# intentionally narrower than the full internal DSL registry
# (factor_eval.ALLOWED_FUNCS/ALLOWED_VARIABLES, which stays a permissive
# superset for backward compatibility with historical factor_code strings
# in this project's logs/manuscript, e.g. `rank`, `ts_mean`, `ts_std`,
# `returns`) — new proposals are guided toward this smaller, more
# disciplined vocabulary, but old names remain valid if they slip through.
RECOMMENDED_VARIABLES = ["close", "open", "high", "low", "volume", "vwap"]

OPERATOR_TABLE = """| Operator Category | Allowed Operators | Purpose in Search Loop |
|---|---|---|
| Arithmetic | +, -, *, / | Basic feature combination and spread construction. |
| Cross-Sectional | cs_rank(X) | Converts raw features into dollar-neutral rank weights ([0,1] range). |
| Time-Series Normalization | ts_rank(X, N), zscore(X, N) | Measures rolling temporal anomalies relative to history. |
| Low-Pass / Turnover Controls | decay_linear(X, N), sma(X, N) | Smooths weight changes across rebalance periods to directly minimize turnover. |
| Volatility / Scale Adjustments | stddev(X, N), abs(X) | Scales position sizing by rolling historical volatility. |"""

ANTI_PATTERN_RULES = """[DISALLOWED ANTI-PATTERNS & AST GUARDRAILS]
Your candidate factor_code will be automatically rejected by the AST sandbox if it contains any of the following:

- REDUNDANT RANKING: Never wrap a rank directly inside a rank of the same kind (e.g. cs_rank(cs_rank(X)) or ts_rank(ts_rank(X, N), N)).
- TRIVIAL SCALARS: Do not multiply or add a positive constant onto the entire expression (e.g. 2 * X or X + 5) -- cross-sectional sorting is invariant to this, so it is a no-op. (Multiplying by -1 to flip long/short direction is fine -- that changes the sort order.)
- CASCADING SMOOTHERS: Do not directly nest more than one smoothing operator (e.g. do NOT do sma(decay_linear(X, 10), 10)).
- UNPROTECTED DIVISION: Every division X / Y must protect the denominator with an epsilon term (Y + 1e-5) or use a ranked denominator (e.g. cs_rank(Y)) -- Y could otherwise be zero.
- UNBOUNDED POWERS: Do not use exponents greater than 2.
- INVALID LOOKBACKS: All rolling time-series window parameters N (in ts_rank, zscore, sma, stddev, decay_linear) must be integers with 2 <= N <= 252. Windows of N <= 1 are degenerate and strictly forbidden for these functions."""

SYSTEM_PROMPT = f"""You are an expert quantitative research assistant mining daily
cross-sectional equity factors for a dollar-neutral long/short U.S. large-cap
strategy.

Your ONLY output must be a single JSON object, with no other text, of the form:
{{"factor_code": "<expression>", "rationale": "<one sentence>"}}

`factor_code` MUST be a single-line mathematical expression built from the
operators below, applied to the recommended data variables. No other syntax
is permitted: no imports, no attribute access, no assignment, no comments,
no comprehensions, no function definitions. The expression must evaluate to
a (date x ticker) score panel; higher score = more attractive to hold long
that day.

{OPERATOR_TABLE}

Recommended Data Variables (4 to 6 max) -- stick exclusively to standard,
daily price-volume market data:
  1. close  (Daily closing price)
  2. open   (Daily opening price)
  3. high   (Daily high price)
  4. low    (Daily low price)
  5. volume (Daily share volume)
  6. vwap   (Volume-weighted average price proxy: (high + low + close) / 3)

{ANTI_PATTERN_RULES}

Additional grammar and complexity guidance:
- If you combine `decay_linear` and `ts_rank`, prefer `decay_linear` as the
  OUTER operator, e.g. decay_linear(ts_rank(x, n), m) -- NOT the inner one,
  ts_rank(decay_linear(x, n), m). Taking a rolling rank of an already-
  smoothed signal reintroduces day-to-day churn and tends to cancel out the
  smoothing decay_linear was providing.
- Keep total expression complexity to at most {config.MAX_DSL_NODES} operator
  nodes (function calls plus binary/unary operators, e.g. cs_rank(a+b) is 2:
  one call, one add). Expressions over this budget incur a bloat penalty on
  top of the reward described each round, separate from the hard rejections
  above.

Each round you are given the best factor found so far and the most recently
proposed factor, each with its backtested diagnostics and, for the most
recent proposal, a short structural critique of what to fix. All diagnostics
are computed on an in-sample search window only; a later, held-out period
you never see will be used to judge whether your final factor generalizes.
Propose ONE new candidate factor_code, derived from the best factor so far,
that you expect to improve Reward.
"""


def reward_fn(
    mode: str, result: FactorEvalResult, gamma1: float, gamma2: float, node_count: int,
    node_limit: int = config.MAX_DSL_NODES, bloat_weight: float = config.NODE_BLOAT_PENALTY_WEIGHT,
) -> float:
    if mode == "baseline":
        base = result.Gross_IR
    elif mode == "execution_aware":
        base = result.Gross_IR - gamma1 * result.Turnover - gamma2 * result.Cost_Impact
    else:
        raise ValueError(f"Unknown mode: {mode}")
    bloat_penalty = bloat_weight * max(0, node_count - node_limit)
    return base - bloat_penalty


def _diagnose(factor_code: str, result: FactorEvalResult, mode: str, node_count: int) -> list[tuple[str, str]]:
    """Rule-based structural diagnosis of a candidate's failure modes,
    mirroring what a human quant researcher would flag on seeing the same
    backtest output. Mode-aware: baseline's diagnosis never references
    turnover or cost, because baseline's reward is Gross_IR only -- raising
    a turnover complaint there would leak execution-awareness into what is
    supposed to be the turnover-blind control condition.

    Returns a list of (diagnosis, task) pairs -- `diagnosis` explains WHY a
    metric failed, `task` is the corresponding imperative instruction for
    what to change. `generate_critique` renders both halves of each pair
    together (kept as a single prompt line for concision, rather than a
    separately-labeled task line -- the diagnosis already implies the fix
    closely enough that spelling it out twice added length without changing
    what the agent did in practice)."""
    pairs: list[tuple[str, str]] = []

    if mode == "execution_aware":
        if ts_rank_wraps_decay_linear(factor_code):
            pairs.append((
                "ts_rank is applied around a decay_linear-smoothed term (ts_rank(decay_linear(...), ...)). "
                "Taking a rolling rank of an already-smoothed signal reintroduces day-to-day churn and tends "
                "to cancel decay_linear's turnover reduction.",
                "Rewrite so decay_linear is the OUTER operator, e.g. decay_linear(ts_rank(x, n), m), "
                "not ts_rank(decay_linear(x, n), m)."
            ))
        if result.Turnover > config.EXECUTION_AWARE_TURNOVER_TARGET:
            has_decay = "decay_linear" in factor_code
            pairs.append((
                f"Turnover ({result.Turnover * 100:.0f}%/yr) exceeds the "
                f"{config.EXECUTION_AWARE_TURNOVER_TARGET * 100:.0f}%/yr target.",
                "Wrap the alpha signal in decay_linear(..., n) with a longer n."
                if not has_decay else
                "Try a longer decay window, or a slower base signal."
            ))
        cost_gap = result.Gross_IR - result.Net_IR
        if cost_gap > 1.0:
            pairs.append((
                f"Net_IR ({result.Net_IR:.2f}) trails Gross_IR ({result.Gross_IR:.2f}) by {cost_gap:.2f}, "
                "mostly from transaction cost.",
                "Prioritize turnover reduction over adding more terms."
            ))
    else:  # baseline: raw-return diagnosis only, never mentions turnover/cost
        if result.Gross_IR < 0:
            pairs.append((
                f"Gross_IR is negative ({result.Gross_IR:.2f}) -- the raw signal has no edge in-sample.",
                "Try a different economic mechanism (momentum, reversal, volume confirmation) rather than "
                "parameter tweaks to the current one."
            ))
        elif result.Gross_IR < 0.3:
            pairs.append((
                f"Gross_IR ({result.Gross_IR:.2f}) is weak.",
                "Combine with a complementary signal (e.g. volume or volatility confirmation) to strengthen "
                "raw return quality."
            ))

    if node_count > config.MAX_DSL_NODES:
        pairs.append((
            f"Expression complexity ({node_count} operator nodes) exceeds the {config.MAX_DSL_NODES}-node "
            "budget and incurs a bloat penalty -- the extra terms may be adding noise rather than signal.",
            "Simplify: remove or merge redundant terms to get back under the node budget."
        ))

    return pairs


def generate_critique(factor_code: str, result: FactorEvalResult, mode: str, node_count: int) -> str:
    """Diagnosis + corresponding fix, joined per issue (`"{why} {what to do}"`),
    across every rule `_diagnose` fired on this candidate."""
    pairs = _diagnose(factor_code, result, mode, node_count)
    if not pairs:
        return "No major structural issues detected; consider incremental refinement."
    return " ".join(f"{diagnosis} {task}" for diagnosis, task in pairs)


@dataclass
class RoundState:
    """One round's proposal plus everything derived from it. `error` is set
    (and `result`/`reward`/etc. left None) when the proposal failed to parse
    or evaluate; `critique` is only populated for successfully-evaluated
    rounds, since diagnosing a syntax error isn't meaningful."""
    code: str
    result: FactorEvalResult | None = None
    reward: float = field(default=float("-inf"))
    node_count: int | None = None
    error: str | None = None
    critique: str | None = None


def _fmt_metrics(state: RoundState) -> str:
    r = state.result
    return (
        f"code=`{state.code}` Gross_IR={r.Gross_IR:.3f} Net_IR={r.Net_IR:.3f} "
        f"Turnover={r.Turnover * 100:.0f}%/yr Cost_Impact={r.Cost_Impact:.4f} "
        f"MaxDD={r.max_drawdown:.1%} Nodes={state.node_count} reward={state.reward:.4f}"
    )


def build_user_prompt(
    iteration: int, n_iterations: int, mode: str, gamma1: float, gamma2: float,
    last: RoundState, best: RoundState,
) -> str:
    """Both `last` and `best` are always populated from round 1 onward: the
    loop bootstraps by evaluating `config.SEED_ALPHA` before round 1, so
    baseline and execution-aware both start round 1 from an identical,
    already-scored seed rather than a blank slate."""
    reward_def = (
        "Reward = Gross_IR (raw pre-cost return quality; turnover and cost are NOT scored), minus a "
        "penalty if expression complexity exceeds the node budget stated in the system prompt."
        if mode == "baseline"
        else f"Reward = Gross_IR - {gamma1}*Turnover - {gamma2}*Cost_Impact (turnover and transaction "
             "cost ARE penalized), minus a penalty if expression complexity exceeds the node budget. "
             f"Aim for the best Net_IR (net of realistic costs) while keeping annualized turnover under "
             f"{config.EXECUTION_AWARE_TURNOVER_TARGET * 100:.0f}%."
    )

    lines = [
        f"Round {iteration}/{n_iterations}. Mode={mode}.",
        reward_def,
        "",
        f"Best factor so far: {_fmt_metrics(best)}",
    ]
    if last.error is not None:
        lines.append(f"Most recent proposal FAILED to evaluate: code=`{last.code}` -- {last.error}")
    else:
        lines.append(f"Most recent proposal: {_fmt_metrics(last)}")
        if last.critique:
            lines.append(f"Diagnostic critique: {last.critique}")
    lines.append("Propose ONE new candidate factor_code, derived from the best factor so far, "
                  "as the required JSON object.")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise InvalidFactorCode(f"No JSON object found in LLM response: {text[:200]!r}")
    return json.loads(match.group(0))


@dataclass
class AgentLoopResult:
    log: pd.DataFrame
    best_result: FactorEvalResult
    best_reward: float


def run_agent_loop(
    llm_client: LLMClient,
    mode: str = "execution_aware",
    n_iterations: int = config.N_ITERATIONS,
    gamma1: float = config.GAMMA_1,
    gamma2: float = config.GAMMA_2,
    panels: dict[str, pd.DataFrame] | None = None,
    cost_params: config.CostParams | None = None,
    log_path: str | Path | None = None,
    verbose: bool = True,
    eval_window: tuple | None = None,
    seed_alpha: str = config.SEED_ALPHA,
    node_limit: int = config.MAX_DSL_NODES,
    bloat_weight: float = config.NODE_BLOAT_PENALTY_WEIGHT,
) -> AgentLoopResult:
    """`eval_window` (start, end), if given, is forwarded to every
    `evaluate_factor` call so the search loop's feedback is computed only
    over that date range (e.g. the in-sample split) — the agent never sees
    diagnostics computed on dates outside it.

    `seed_alpha` is evaluated once before round 1 and used to bootstrap both
    `last` and `best` — pass the SAME value (the default, config.SEED_ALPHA)
    for both baseline and execution-aware runs being compared, so both start
    from an identical, already-scored expression."""
    if panels is None:
        from .data_pipeline import get_panels
        panels = get_panels()

    seed_nodes = count_ast_nodes(seed_alpha)
    seed_result = evaluate_factor(seed_alpha, panels=panels, cost_params=cost_params, eval_window=eval_window)
    seed_reward = reward_fn(mode, seed_result, gamma1, gamma2, seed_nodes, node_limit, bloat_weight)
    seed_critique = generate_critique(seed_alpha, seed_result, mode, seed_nodes)
    seed_state = RoundState(seed_alpha, seed_result, seed_reward, seed_nodes, critique=seed_critique)

    last = seed_state
    best = seed_state
    rows: list[dict] = [{
        "iteration": 0, "mode": mode, "factor_code": seed_alpha, "error": None,
        "Gross_IR": seed_result.Gross_IR, "Turnover": seed_result.Turnover,
        "Cost_Impact": seed_result.Cost_Impact, "Net_IR": seed_result.Net_IR,
        "node_count": seed_nodes, "reward": seed_reward,
    }]
    if verbose:
        print(f"[{mode}] iter  0/{n_iterations} [SEED] reward={seed_reward:8.4f}  {seed_alpha}")

    for i in range(1, n_iterations + 1):
        user_prompt = build_user_prompt(i, n_iterations, mode, gamma1, gamma2, last, best)
        raw = llm_client.complete(SYSTEM_PROMPT, user_prompt)

        factor_code = None
        try:
            payload = _extract_json(raw)
            factor_code = payload["factor_code"]
            violations = check_anti_patterns(factor_code)
            if violations:
                raise InvalidFactorCode("Rejected by AST anti-pattern guardrails: " + "; ".join(violations))
            result = evaluate_factor(factor_code, panels=panels, cost_params=cost_params, eval_window=eval_window)
            node_count = count_ast_nodes(factor_code)
            reward = reward_fn(mode, result, gamma1, gamma2, node_count, node_limit, bloat_weight)
            critique = generate_critique(factor_code, result, mode, node_count)
            state = RoundState(factor_code, result, reward, node_count, critique=critique)
        except Exception as e:  # invalid JSON, invalid DSL, eval error, etc.
            state = RoundState(factor_code if factor_code is not None else raw[:200], error=str(e))

        rows.append({
            "iteration": i, "mode": mode, "factor_code": state.code, "error": state.error,
            "Gross_IR": state.result.Gross_IR if state.result else float("nan"),
            "Turnover": state.result.Turnover if state.result else float("nan"),
            "Cost_Impact": state.result.Cost_Impact if state.result else float("nan"),
            "Net_IR": state.result.Net_IR if state.result else float("nan"),
            "node_count": state.node_count if state.node_count is not None else float("nan"),
            "reward": state.reward,
        })

        last = state
        if state.result is not None and state.reward > best.reward:
            best = state

        if verbose:
            status = "OK " if state.error is None else "ERR"
            print(f"[{mode}] iter {i:2d}/{n_iterations} [{status}] reward={state.reward:8.4f}  {state.code}")

        if log_path is not None:
            pd.DataFrame(rows).to_csv(log_path, index=False)

    log_df = pd.DataFrame(rows)
    return AgentLoopResult(log=log_df, best_result=best.result, best_reward=best.reward)
