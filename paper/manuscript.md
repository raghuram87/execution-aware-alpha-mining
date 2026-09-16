# Execution-Aware Alpha Mining: Teaching LLM Factor Agents to Price Their Own Trading Costs

**Raghuram Nagireddy¹**

¹ [Affiliation/address — placeholder: fill in before submission], raghuram87@gmail.com

*Prepared for submission to Finance Research Letters*

---

## Highlights

- An LLM mines equity factors under a raw-return reward vs. a cost-penalized reward.
- A 21-fold walk-forward test (2006-2026) uses true point-in-time S&P membership.
- Cost-blind search hits 49-101%/day turnover, wiped out in all four factor categories.
- Penalizing turnover cuts it 22-38x and makes one category net profitable.
- Findings replicate across three independent LLM sampling seeds per category.

## Abstract

Large language model (LLM) agents increasingly generate candidate trading signals, but almost all "alpha mining" pipelines reward the agent on in-sample return quality alone, leaving transaction costs to be discovered at implementation. We build a closed-loop system in which a locally-hosted LLM (Qwen2.5-Coder-14B-Instruct) proposes daily cross-sectional equity factors in an AST-validated expression language; each candidate is backtested on a true point-in-time S&P universe (1999-2026, not a survivorship-biased snapshot) and priced for execution cost (a Corwin-Schultz spread estimator plus a square-root impact model); diagnostics feed back to the agent every round. Evaluation uses rolling walk-forward optimization — 21 five-year-train/one-year-test folds spanning 2006-2026, each re-searched from scratch — crossed with four seed alphas from distinct predictor categories (reversal, volume, volatility, momentum) and three LLM seeds, for 12 independently re-optimized comparisons. An agent rewarded purely on gross return converges to 49-101%/day turnover in every category and is wiped out (net Sharpe -4.7 to -11.2, cumulative return -100%) despite a genuine average gross Sharpe of 0.67 — the signal is real, but the reward is blind to its cost. Rewarding the identical search net of a turnover and cost penalty cuts turnover 22-38x in every category and, for the volatility-anomaly seed, turns a would-be loss into a net-profitable factor (net Sharpe +0.35, cumulative return +117%, ~71-day holding period), consistently across all three seeds. The other three categories still lose net of cost but far less catastrophically, showing the reward specification, not just the underlying signal, governs whether an automated search survives real trading frictions.

**Keywords:** LLM agents; alpha mining; transaction costs; factor turnover; execution-aware backtesting; point-in-time universe

**JEL classification:** G11, G12, C45, C63

---

## 1. Introduction

Generative language models are now routinely used as a search process over candidate trading signals: an LLM proposes a factor expression, a backtest scores it, and the loop repeats (in the spirit of automated formulaic-alpha search, Kakushadze, 2016). This turns factor discovery into a search problem in which the reward function *is* the research objective. If that reward is gross in-sample return quality — the default in almost every published or informally circulated pipeline — the search will systematically favor short-horizon, high-turnover signals, because fast cross-sectional effects are well known to carry attractive raw Sharpe ratios that are largely compensation for the spread and impact a real trader would pay to capture them (Novy-Marx and Velikov, 2016; Korajczyk and Sadka, 2004).

This paper asks whether telling a factor-mining LLM agent about turnover and cost *during* the search, not only at final evaluation, changes what it finds and whether that finding survives cost better. We build a reproducible pipeline: an LLM proposes factor expressions in a whitelisted, AST-validated expression grammar over price-volume panels; each candidate becomes a daily-rebalanced, dollar-neutral long/short portfolio; its cost is priced with a Corwin and Schultz (2012) spread estimator plus a square-root impact model; and the resulting diagnostics are returned to the agent. We compare two reward specifications on the identical loop — a *baseline* agent rewarded on gross return quality alone, and an *execution-aware* agent rewarded net of an explicit turnover and cost penalty — each bootstrapped from the same seed expression and LLM sampling seed, so the two searches differ only in reward.

Two design choices distinguish this study from related work. First, we use true point-in-time S&P constituent membership (1999-2026) rather than a present-day snapshot, avoiding the survivorship bias that a static universe of currently-listed names introduces into any long-history backtest. Second, we cross four seed alphas from economically distinct predictor categories — short-term reversal (Jegadeesh, 1990), abnormal-volume fade (Gervais, Kaniel and Mingelgrin, 2001), the low-volatility anomaly (Ang, Hodrick, Xing and Zhang, 2006), and cross-sectional momentum (Jegadeesh and Titman, 1993) — with three independent LLM sampling seeds each, so the comparison is not a single lucky search but 12 independently re-optimized outcomes. To our knowledge this is the first study to place a realistic cost model inside an LLM's feedback loop, rather than applying it only as an ex-post filter, and to test the resulting effect for robustness across both sampling randomness and the economic mechanism the search is warm-started from.

## 2. Data and Methodology

**Point-in-time universe.** We reconstruct true historical S&P constituent membership (1999-2026) from a public historical-components record, sourcing daily OHLCV from a local multi-vendor price archive supplemented by a Yahoo Finance gap-fill; combined coverage of point-in-time constituents ranges from roughly 75% (2001) to 99% (2026). For compute tractability, each walk-forward fold trades a liquidity-capped subset (~150-220 tickers) selected by trailing two-year dollar volume, reconstituted annually *within* each fold using only data available as of each reconstitution date — never the fold's own future — so universe selection is causal at every point (Appendix A.1).

**Cost engine.** Bid-ask spread is estimated per stock-day from the Corwin and Schultz (2012) high-low estimator (bounded to 1-100bps); impact follows a square-root law, `Impact = Y * sigma_daily * sqrt(participation)`, with `Y = 0.5` (Grinold and Kahn, 2000) and participation measured against 21-day average dollar volume. Daily cost drag is this one-way cost applied to that day's dollar weight change, summed across the book and subtracted from gross return.

**Factor language and guardrails.** Factor candidates are single-line expressions over a whitelisted operator set (`cs_rank`, `ts_rank`, `zscore`, `decay_linear`, `sma`, `stddev`, arithmetic) applied to `open/high/low/close/volume/vwap`, parsed with Python's `ast` module and rejected unless every node is whitelisted before evaluation. Beyond syntax whitelisting, six hard anti-pattern rules reject redundant ranking, no-op scalar rescaling, cascading smoothers, unprotected division, unbounded powers, and out-of-range lookback windows (2-252 days) before a candidate is ever backtested (Appendix A.2).

**Agent and reward.** Figure 1 summarizes the closed loop linking these components. The Generator/Refiner is Qwen2.5-Coder-14B-Instruct (Q4\_K\_M, run locally via llama.cpp, temperature 0.7 — validated against temperature 0 and 0.3, both of which collapse to a fixed point after 1-2 rounds rather than sustaining search, Appendix A.3). Each of the two reward conditions uses an independently-constructed, identically-seeded LLM client bootstrapped from the same seed alpha, so baseline and execution-aware are apples-to-apples: same starting expression, same sampling seed, same 50-round budget, differing only in reward. Baseline reward is Gross\_IR; execution-aware reward is `Gross_IR - gamma1*Turnover - gamma2*Cost_Impact` (gamma1=0.5, gamma2=8), both penalized identically for expression complexity above an 8-node budget. Turnover is computed against the drift-adjusted pre-rebalance weight (not the prior day's stale target), avoiding double-counting passive price drift as a trade (Appendix A.4).

**Walk-forward design.** Twenty-one non-overlapping-test rolling folds span the point-in-time sample: a 5-year train window, immediately followed by a 1-year test window, rolling forward one year at a time, with test windows from 2006 through mid-2026. On each fold the agent runs its full 50-round search restricted to that fold's train window; the winning factor is frozen and scored once on the held-out test window; the 21 folds' out-of-sample return streams are stitched into one continuous curve per (seed alpha, sampling seed, mode). We report the average daily turnover and its inverse, the average holding period in trading days, alongside net Sharpe under the full cost model and flat 5/10/20bps scenarios.

## 3. Results

Table 1 reports each category's stitched walk-forward result, averaged across the three LLM sampling seeds (standard deviation in parentheses).

**Table 1. Baseline vs. execution-aware, by seed-alpha category, averaged across 3 LLM sampling seeds (21 folds each, test periods 2006-2026)**

| Seed alpha | Mode | Net Sharpe (full cost) | Avg. daily turnover | Avg. holding period (days) | Cumulative return |
|---|---|---:|---:|---:|---:|
| Reversal | Baseline | -9.74 (0.80) | 92.3% | 1.1 | -100% |
| Reversal | Execution-aware | -0.38 (0.20) | 4.2% | 24.2 | -77% |
| Volume | Baseline | -11.24 (1.27) | 100.8% | 1.0 | -100% |
| Volume | Execution-aware | -0.25 (0.18) | 4.1% | 24.3 | -55% |
| Volatility | Baseline | -4.67 (0.91) | 49.0% | 2.0 | -100% |
| Volatility | Execution-aware | **+0.35 (0.02)** | 1.4% | 71.2 | **+117%** |
| Momentum | Baseline | -6.61 (1.21) | 70.8% | 1.4 | -100% |
| Momentum | Execution-aware | -0.10 (0.03) | 1.8% | 54.2 | -47% |

Figure 2 plots the stitched cumulative net return for one representative sampling seed per category; Figure 3 plots Net Sharpe against the assumed one-way cost, from a lenient 5bps flat assumption up to the full Corwin-Schultz-plus-impact model, for the mean across all three sampling seeds.

Three results stand out. First, the baseline is wiped out in *every* category — 49-101%/day turnover means the entire book turns over roughly once daily to nearly twice daily, an economically extreme trading pace no reward function that ignores cost has any reason to avoid. This is not a search that failed to find signal: the baseline's average gross Sharpe across categories is a genuine 0.67 (reversal 0.77, volume 0.62, volatility 0.73, momentum 0.55) — the reward is doing exactly what it was told to do, and what it was told to do is blind to the cost of capturing that signal.

Second, penalizing turnover and modeled cost inside the reward — nothing else about the search, universe, or cost model changes — cuts turnover by 22x (reversal) to 38x (momentum) in every category, and the improvement is not a knife-edge result: standard deviation across the three independent sampling seeds is small relative to the baseline-versus-execution-aware gap in every category (e.g., volatility's net Sharpe gap of 5.0 dwarfs a cross-seed standard deviation of 0.02-0.91).

Third, and most notably, the volatility-anomaly seed does not merely fail less badly — its execution-aware factor is net profitable, consistently across all three sampling seeds (net Sharpe +0.33 to +0.37, cumulative return +104% to +131%), with the lowest turnover of any category (1.4%/day, a roughly 71-trading-day average holding period). The mechanism is visible in the gross numbers: execution-aware's own gross Sharpe in this category (0.48 on average) is close to the baseline's (0.73), but its turnover is roughly 35x lower, so far less of that gross edge is spent on trading cost. The other three categories retain a smaller share of their gross edge under the execution-aware reward and remain net-negative, but dramatically less so than their cost-blind counterparts.

## 4. Discussion and limitations

The central finding — that penalizing turnover and modeled cost inside the search, rather than only at final evaluation, changes what an LLM factor-mining agent converges to, and that the change is large and robust across both sampling randomness and the economic category the search starts from — argues that the reward specification is doing real work in any LLM-driven alpha-mining pipeline, not a detail to be tuned after the fact. A pipeline that reports "the LLM found a factor with Sharpe X" without disclosing the reward function, or without re-searching across multiple folds and seeds, is silently answering a different and less trustworthy question than one that reports a finding stable across 12 independent re-optimizations.

Several limitations qualify these results. The point-in-time universe's price coverage is incomplete pre-2010 (roughly 75-90% of true constituents), a data-availability constraint of free/quasi-free sources rather than a methodological choice; because it affects both reward conditions identically within each fold, it should not bias the baseline-versus-execution-aware *comparison*, though it may affect absolute return levels in early folds. The liquidity cap (~150-220 tickers per fold, causally reconstituted) trades away some breadth for compute tractability; a single full-universe confirmatory run is a natural robustness check before submission. The cost model's impact calibration constant (`Y=0.5`) is not fit to realized execution data, motivating the flat-cost sensitivity columns. Portfolio construction is deliberately unconstrained — daily rebalancing, no no-trade band — so the factor expression itself, via `decay_linear`, is the only turnover-reduction lever available to the search; pairing execution-aware search with portfolio-level turnover control would likely narrow the gap further. Finally, we would caution against reading the negative net Sharpe ratios in three of four categories as evidence that these predictor families are categorically unprofitable net of cost; they instead reflect a deliberately unforgiving stress test — uncontrolled 200%-gross daily rebalancing — chosen to make the cost channel, and the agent's response to it, easy to see, with the volatility category showing that the same stress test can flip to a profitable outcome once the reward stops being blind to cost.

## 5. Conclusion

Holding the search process, universe, and cost model fixed, and varying only whether the reward function is blind or aware of trading cost, we find that a locally-hosted LLM factor-mining agent's raw-return reward converges to 49-101%/day turnover and is wiped out net of realistic cost in every one of four economically distinct seed-alpha categories, despite finding genuine gross signal (average gross Sharpe 0.67). Rewarding the identical search net of turnover and cost cuts turnover 22-38x in every category and, for a low-volatility-anomaly seed, converts the outcome into a net-profitable factor — consistently across three independent LLM sampling seeds. The result replicates across 12 independently re-optimized (seed-alpha, sampling-seed) combinations on a true point-in-time universe spanning 21 rolling walk-forward folds from 2006 to 2026. Whether or not any single resulting factor is itself investable, the exercise argues that execution cost belongs inside the loop an LLM-based factor-mining agent optimizes, not applied only as a filter after the fact.

---

## Figure captions

**Figure 1.** Architecture of the closed-loop system: the LLM agent proposes a factor expression, which is turned into weights and backtested by the Factor Evaluator, priced by the Cost Engine (Corwin-Schultz spread plus square-root impact), summarized into diagnostics (Gross\_IR, Turnover, Cost\_Impact, Net\_IR, Reward), and fed back to the agent for the next round.

**Figure 2.** Out-of-sample cumulative net-of-cost return, baseline (blue) vs. execution-aware (orange), one panel per seed-alpha category, for a single representative LLM sampling seed (1001); dashed vertical lines mark each walk-forward fold's annual refit date. Table 1 reports the full three-seed mean/standard deviation this figure illustrates one instance of.

**Figure 3.** Net Sharpe ratio as a function of the assumed one-way trading cost, from a lenient flat 5bps through 10bps and 20bps to the realistic full Corwin-Schultz-plus-impact model, baseline (blue) vs. execution-aware (orange), one panel per seed-alpha category, averaged across all three LLM sampling seeds. The baseline's steep leftward slope even at 5bps shows its edge is cost-fragile at any assumption tested; the execution-aware line's flatness across all four cost scenarios shows its low turnover, not a favorable cost assumption, is what protects it.

---

## CRediT authorship contribution statement

**Raghuram Nagireddy**: Conceptualization, Methodology, Software, Formal analysis, Data curation, Writing – original draft, Writing – review & editing, Visualization.

## Data availability statement

Point-in-time S&P constituent membership is sourced from a public historical-components record; daily OHLCV is sourced from a local multi-vendor price archive supplemented by the `yfinance` Python package (Yahoo Finance), subject to the respective sources' terms of use. All code implementing the point-in-time universe construction, cost engine, factor expression language and guardrails, LLM agent loop, and walk-forward experiment scripts, together with the full result logs underlying Table 1 (`grid_summary.csv` and per-fold tables for all 12 seed-alpha/sampling-seed combinations) and all figures, are available at: https://github.com/raghuram87/execution-aware-alpha-mining. The point-in-time price panels themselves are not stored in the repository, as they are fully regenerable from the code and public/local sources cited above (`src/pit_universe.get_pit_panels`); the local multi-vendor price archive used to supplement point-in-time coverage is not itself publicly redistributable, but is not required to reproduce results, since a `yfinance`-only fallback path is available at reduced pre-2016 coverage.

## Declaration of competing interest

The author(s) declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## References

Ang, A., Hodrick, R.J., Xing, Y., Zhang, X., 2006. The cross-section of volatility and expected returns. *Journal of Finance* 61(1), 259–299.

Bailey, D.H., Borwein, J.M., López de Prado, M., Zhu, Q.J., 2014. Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society* 61(5), 458–471.

Corwin, S.A., Schultz, P., 2012. A simple way to estimate bid-ask spreads from daily high and low prices. *Journal of Finance* 67(2), 719–760.

Gervais, S., Kaniel, R., Mingelgrin, D.H., 2001. The high-volume return premium. *Journal of Finance* 56(3), 877–919.

Grinold, R.C., Kahn, R.N., 2000. *Active Portfolio Management*, 2nd ed. McGraw-Hill, New York.

Jegadeesh, N., 1990. Evidence of predictable behavior of security returns. *Journal of Finance* 45(3), 881–898.

Jegadeesh, N., Titman, S., 1993. Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance* 48(1), 65–91.

Kakushadze, Z., 2016. 101 formulaic alphas. *Wilmott* 2016(84), 72–81.

Korajczyk, R.A., Sadka, R., 2004. Are momentum profits robust to trading costs? *Journal of Finance* 59(3), 1039–1082.

López de Prado, M., 2018. *Advances in Financial Machine Learning*. Wiley, Hoboken, NJ.

Novy-Marx, R., Velikov, M., 2016. A taxonomy of anomalies and their trading costs. *Review of Financial Studies* 29(1), 104–147.

Yang, H., Liu, X.-Y., Wang, C.D., 2023. FinGPT: Open-source financial large language models. *arXiv preprint* arXiv:2306.06031.

---

## Appendix A. Methodological detail (supplementary, not counted against the main text word limit)

### A.1 Point-in-time universe construction

Historical S&P constituent membership snapshots (~weekly granularity, 1996-2026) are forward-filled onto the daily trading calendar to build a (date x ticker) eligibility mask, materializing each snapshot as an explicit True/False vector over the full ticker universe before forward-filling — a "presence-only" long-format table would never record an explicit exit, causing membership to only ever accumulate. Daily OHLCV is resolved per ticker from a local multi-vendor archive (a Tiingo-sourced table checked first for its broader delisted-ticker coverage, falling back to a second vendor table), with ticker-symbol normalization (dot/dash dual-class variants) tried against both spellings; any ticker or date range not resolved locally is gap-filled via `yfinance`. Raw open/high/low are rescaled by the ratio of adjusted to unadjusted close before use, since the local archive stores a separately-adjusted close alongside raw OHLC rather than fully adjusted OHLC.

For compute tractability, each walk-forward fold selects its own liquidity-capped ticker subset rather than trading the full ~500-name point-in-time index: at each of five annual points within the fold's training window plus one at test start, the top-150 tickers by trailing two-year average dollar volume — computed only over days each ticker was an actual point-in-time constituent, and only using data strictly before that reconstitution date — are selected; a ticker's membership in the fold's day-varying "liquid" mask holds from its reconstitution date until the next one supersedes it. This differs from (i) a single global liquidity ranking computed over the full sample, which would let an early fold trade names that only became liquid decades later, and from (ii) one fixed snapshot at the fold's train start reused through the whole six-year fold, which would leave the test year trading an increasingly stale universe. The usable ("eligible") universe on a given day is the intersection of true point-in-time membership and the currently active liquidity reconstitution block; the union of a fold's six annual reconstitutions is typically only modestly larger than 150 (205-218 observed), since adjacent-year liquidity rankings overlap heavily.

### A.2 Anti-pattern guardrails

Beyond AST whitelisting of variables, function calls, and arithmetic operators, six rules reject a candidate outright before backtesting: (1) redundant direct re-ranking of an already-ranked signal; (2) a positive constant added to or multiplied onto the entire expression (a no-op under cross-sectional ranking; sign flips are exempt as they are meaningful); (3) more than one smoothing operator (`decay_linear`, `sma`) directly nested; (4) division whose denominator is not epsilon-protected or itself a ranked (and hence bounded-away-from-zero) value; (5) exponents above 2; (6) lookback windows outside [2, 252] trading days for any windowed operator (`delta`/`delay` are exempt, since a one-day change or lag is a meaningful quantity, not a degenerate rolling statistic at N=1).

### A.3 Local LLM temperature validation

Full-trajectory inspection (not just tail sampling) of temperature-0 and temperature-0.3 runs showed both collapse to a fixed point after round 1-2 and make no further progress for the remainder of a 25-50 round run, despite temperature-0.3 initially appearing, from partial inspection, to preserve exploration — a conclusion retracted on full-trajectory review. Temperature 0.7 was the lowest setting found to sustain genuine multi-round search; consequently all reported results characterize sampling variance via three independent seeds at temperature 0.7 rather than relying on a lower, more deterministic temperature.

### A.4 Turnover computation

Reported turnover follows a drift-adjusted, weight-based definition: today's target weight is compared not against yesterday's raw target but against how yesterday's weight would have organically drifted given that day's realized returns (`w_pre = w_prior * (1+r) / sum|w_prior * (1+r)| * gross_exposure`), so passive price drift is not double-counted as an active trade. The reported one-way daily turnover halves the resulting portfolio-level sum of absolute weight changes (buys and sells are equal in dollar terms for a dollar-neutral book); the cost engine itself is charged on the full, un-halved weight change, since realized transaction cost is paid on both legs of every rebalance.

---

*[Main text word count (Sections 1-5 only, excluding title/highlights/abstract/keywords/Table 1/statements/references/Appendix A/figure captions): 1,758 words (measured), well within FRL's 2,500-word letter format limit. Figures generated and ready for insertion: `results/figures/fig1_architecture.pdf` (architecture), `fig2_cumulative_returns.pdf` (small-multiples cumulative return by category, representative seed 1001), `fig3_bps_sensitivity.pdf` (small-multiples Net Sharpe vs. cost scenario, 3-seed mean) — to be inserted as Figures 1-3 when formatting for Elsevier Editorial Manager.]*
