Dear Editor,

I am pleased to submit my manuscript, "Execution-Aware Alpha Mining: Teaching LLM Factor Agents to Account for Trading Costs," for consideration as a research article in *Finance Research Open*. The manuscript comes to you through Elsevier's transfer service from *Finance Research Letters* (FRL-D-26-05416). It was declined there at the desk for presentation and formatting reasons, without an assessment of its content.

The paper asks how cost-awareness inside an automated search changes what a large language model (LLM) agent finds when it mines equity factors. A locally hosted LLM proposes factors in a constrained expression language, and each candidate is backtested on a survivorship-free, point-in-time S&P 500 universe and priced with a spread-and-impact cost model. Across 21 walk-forward folds (2006–2026), four seed signals and three sampling seeds, I compare three feedback conditions that share one search procedure: a cost-blind reward, a cost-penalized reward presented without explanation, and the same reward with explicit cost guidance. The main findings are:

- Cost-blind search reaches 36–103% daily turnover and loses its entire capital in every category, despite positive gross Sharpe ratios.
- The cost-penalized reward alone raises fold-level net Sharpe by 7.9 (95% CI 6.4–9.6). Explicit guidance adds a further 2.4 (1.3–3.7) and produces the long-horizon smoothing found in the winning factors.
- Only the low-volatility seed becomes profitable (net Sharpe 0.51; five-factor-plus-momentum alpha of 7.8% per year, t = 3.5), and there simple smoothing or monthly rebalancing rules do as well.

I believe the paper suits the journal's emphasis on methodologically sound research across finance, including asset pricing and digital finance. Several of its results are deliberately reported as limits on what LLM-driven search delivers, notably that simple rules match it in the one profitable category.

Since the version submitted to *Finance Research Letters*, the work has been substantially revised:

1. **Data.** I found that the earlier price data spliced two sources on 29 November 2021. One of them was already adjusted for later stock splits, so the splice produced spurious one-day price drops in several dozen constituents. The earlier source also missed many pre-2013 delistings. All data now come from the survivorship-free Sharadar database, whose point-in-time membership I verified against all 114 of its quarterly constituent snapshots, and every experiment has been rerun.
2. **Design.** The earlier version described the two arms as differing only in their reward. In fact the cost-aware arm also received a prompt that explained the cost penalty, and critiques that recommended smoothing. I added a third, reward-only arm to separate these two effects, and the manuscript now describes all three conditions precisely.
3. **New analyses.** These are comparisons with rules that use no search, factor regressions on the Fama–French five factors plus momentum, and a rerun of the penalty-weight sensitivity analysis.
4. **Format.** The paper is now a full-length article prepared with Elsevier's elsarticle template.

The manuscript is original, has not been published elsewhere, and is not under consideration at any other journal. A preprint of the earlier version appeared on SSRN when the paper entered review at *Finance Research Letters* (https://www.ssrn.com/abstract=7498983); I will replace it with the revised version. The manuscript file has been prepared for double-anonymized review: the code repository's URL appears only on the title page. I have no competing interests to declare and received no funding for this research.

Thank you for your consideration.

Sincerely,

Raghuram Nagireddy
[AFFILIATION]
[FULL POSTAL ADDRESS]
rrn2111@caa.columbia.edu
ORCID: 0009-0002-2203-3367
