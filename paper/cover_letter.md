Dear Editor,

I am pleased to submit my manuscript, "Execution-Aware Alpha Mining: Teaching LLM Factor Agents to Account for Trading Costs," for consideration as a Letter in *Finance Research Letters*.

The paper studies a question that sits squarely in FRL's scope: does a large language model (LLM) used to mine cross-sectional equity factors search differently, and find something that survives real trading costs better, if it is told about transaction costs *during* the search rather than only at final evaluation? I build a closed-loop pipeline — an LLM proposes factor expressions in a constrained, AST-validated language; each candidate is backtested on a reconstructed point-in-time S&P universe (1999-2026, built from historical constituent membership rather than a survivorship-biased current-day snapshot) and priced with a Corwin-Schultz spread plus square-root impact cost model; the diagnostics are fed back to the agent every round. Comparing an agent rewarded on gross returns alone against one rewarded net of an explicit turnover and cost penalty, across four economically distinct seed-alpha categories (reversal, volume, volatility, momentum) and three LLM sampling seeds — 12 independently seeded re-optimizations evaluated via 21-fold rolling walk-forward optimization spanning 2006-2026 — I find the cost-blind agent converges to 49-101%/day turnover and loses its entire notional in every category despite positive average gross Sharpe (0.67), while the cost-aware agent cuts turnover 22-38x and, for the low-volatility-anomaly seed, converts the outcome into a net-profitable factor. A cluster bootstrap over the 252 underlying fold-level comparisons (resampling by seed-alpha/sampling-seed combination, not by individual fold, since folds within a combination share overlapping training windows) puts the execution-aware Net Sharpe improvement at +9.96 (95% CI [8.3, 11.6]), and the mechanism registers in the discovered factors themselves: execution-aware winners use a long-lookback smoothing operator in 100% of fold-seed instances versus 26% for the cost-blind baseline.

The manuscript is 2,374 words in the main text (Sections 1-5), within the journal's 2,500-word letter format limit, with supplementary methodological detail in an appendix that does not count against that limit. A penalty-sensitivity sweep across ten reward configurations (Table 3) confirms the central finding is not an artifact of one specific weighting. It is original work, has not been published previously, and is not under consideration at any other journal. I have no conflicts of interest to declare and received no funding for this research. All code and result logs are publicly available at the repository cited in the manuscript's Data Availability statement.

Thank you for your consideration.

Sincerely,
Raghuram Nagireddy
Independent researcher
rrn2111@caa.columbia.edu
ORCID: 0009-0002-2203-3367
