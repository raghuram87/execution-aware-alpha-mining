"""Week 1: transaction cost harness.

Implements:
  1. The Corwin & Schultz (2012) high-low bid-ask spread estimator, computed
     from consecutive daily high/low prices.
  2. A standard square-root market-impact model (Grinold-Kahn style):
     Impact = Y * sigma_daily * sqrt(participation_rate).

Both are combined into a per-stock, per-day one-way transaction cost (in bps
of trade notional), which `factor_eval.evaluate_factor` applies to turnover
to produce net-of-cost returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostParams, DEFAULT_COST_PARAMS


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Corwin-Schultz (2012) two-day high-low spread estimator.

    For each date t, uses the (t-1, t) pair of daily ranges. Returns the
    estimated *round-trip* percentage spread (fraction of price), with
    negative estimates truncated to zero as in the original paper's
    simplified treatment.
    """
    log_hl = np.log(high / low)
    beta = log_hl.pow(2) + log_hl.shift(1).pow(2)

    high_2d = high.rolling(2).max()
    low_2d = low.rolling(2).min()
    gamma = np.log(high_2d / low_2d).pow(2)

    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(-50, 50)  # guard against overflow in exp()

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0.0)
    return spread


@dataclass
class CostEngine:
    """Bundles the Corwin-Schultz spread estimate with a square-root impact
    model to price the transaction cost of a given weight-change matrix."""

    panels: dict[str, pd.DataFrame]
    params: CostParams = DEFAULT_COST_PARAMS

    def __post_init__(self) -> None:
        self._spread_bps: pd.DataFrame | None = None
        self._daily_vol: pd.DataFrame | None = None
        self._adv_dollar: pd.DataFrame | None = None

    def spread_bps(self) -> pd.DataFrame:
        """One-way half-spread proxy input: full round-trip CS spread in bps,
        clipped to a sane [min, max] band to control for estimator noise."""
        if self._spread_bps is None:
            spread_frac = corwin_schultz_spread(self.panels["high"], self.panels["low"])
            bps = spread_frac * 10_000
            self._spread_bps = bps.clip(self.params.min_spread_bps, self.params.max_spread_bps)
        return self._spread_bps

    def daily_vol(self) -> pd.DataFrame:
        if self._daily_vol is None:
            self._daily_vol = self.panels["returns"].rolling(21).std()
        return self._daily_vol

    def adv_dollar(self) -> pd.DataFrame:
        if self._adv_dollar is None:
            self._adv_dollar = self.panels["dollar_volume"].rolling(self.params.adv_lookback).mean()
        return self._adv_dollar

    def transaction_cost_bps(
        self, weight_changes: pd.DataFrame, nav: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Per-stock, per-day one-way transaction cost in bps of trade notional.

        Returns (total_bps, spread_component_bps, impact_component_bps).
        """
        cols = weight_changes.columns
        dollar_traded = weight_changes.abs() * nav
        adv = self.adv_dollar().reindex(index=weight_changes.index, columns=cols)
        participation = (dollar_traded / adv).clip(upper=1.0).fillna(0.0)

        sigma = self.daily_vol().reindex(index=weight_changes.index, columns=cols)
        impact_bps = self.params.impact_coefficient * sigma * np.sqrt(participation) * 10_000
        impact_bps = impact_bps.fillna(0.0)

        spread_bps = self.spread_bps().reindex(index=weight_changes.index, columns=cols)
        half_spread_bps = (spread_bps / 2).fillna(self.params.min_spread_bps / 2)

        total_bps = half_spread_bps + impact_bps
        return total_bps, half_spread_bps, impact_bps

    def cost_drag(
        self, weight_changes: pd.DataFrame, nav: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Daily portfolio-level cost drag (fraction of NAV) from the full
        Corwin-Schultz spread + square-root impact model, decomposed into
        spread and impact components."""
        total_bps, spread_bps, impact_bps = self.transaction_cost_bps(weight_changes, nav)
        spread_drag = ((spread_bps / 10_000) * weight_changes.abs()).sum(axis=1)
        impact_drag = ((impact_bps / 10_000) * weight_changes.abs()).sum(axis=1)
        total_drag = spread_drag + impact_drag
        return total_drag, spread_drag, impact_drag


def flat_cost_drag(weight_changes: pd.DataFrame, bps: float) -> pd.Series:
    """Simplified fixed one-way cost assumption (e.g. 5/10/20 bps), used for
    the Table 1 sensitivity sweep alongside the full cost model."""
    return weight_changes.abs().sum(axis=1) * (bps / 10_000.0)
