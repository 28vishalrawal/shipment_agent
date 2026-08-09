"""Empirical rate + transit-percentile tables. Deterministic. No LLM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core import column_mapping as cm


@dataclass
class RateTable:
    global_rate: float
    mode_rate: dict[str, float]
    cell_rate: dict[tuple, tuple[int, float]]     # (mode,region,category) -> (n,rate)
    transit_p: dict[str, dict[int, float]]        # mode -> {percentile: days}
    scheduled: dict[str, float]                   # mode -> scheduled days (mode)

    def shrunk_rate(self, mode: str, region: str, category: str, k: int) -> tuple[float, str]:
        """Empirical-Bayes shrinkage toward the mode rate for thin cells."""
        base_mode = self.mode_rate.get(mode, self.global_rate)
        key = (mode, region, category)
        if key in self.cell_rate:
            n, rate = self.cell_rate[key]
            shrunk = (n * rate + k * base_mode) / (n + k)
            src = "cell" if n >= k else "mode"
            return float(shrunk), src
        if mode in self.mode_rate:
            return float(base_mode), "mode"
        return float(self.global_rate), "global"

    def eta_days(self, mode: str, percentile: int) -> float:
        table = self.transit_p.get(mode)
        if not table:
            return float(np.mean(list(self.scheduled.values()) or [0]))
        return float(table.get(percentile, max(table.values())))


def build_rate_table(closed: pd.DataFrame) -> RateTable:
    global_rate = float(closed["late"].mean())

    mode_rate = closed.groupby(cm.SHIPPING_MODE)["late"].mean().to_dict()

    have_cat = cm.CATEGORY in closed.columns
    have_reg = cm.ORDER_REGION in closed.columns
    cell_rate: dict[tuple, tuple[int, float]] = {}
    if have_cat and have_reg:
        g = closed.groupby([cm.SHIPPING_MODE, cm.ORDER_REGION, cm.CATEGORY])["late"].agg(
            ["size", "mean"]
        )
        for idx, row in g.iterrows():
            cell_rate[tuple(idx)] = (int(row["size"]), float(row["mean"]))

    transit_p: dict[str, dict[int, float]] = {}
    scheduled: dict[str, float] = {}
    if cm.DAYS_REAL in closed.columns:
        for mode, sub in closed.groupby(cm.SHIPPING_MODE):
            real = sub[cm.DAYS_REAL].dropna()
            transit_p[mode] = {
                50: float(real.quantile(0.50)),
                75: float(real.quantile(0.75)),
                90: float(real.quantile(0.90)),
            }
            if cm.DAYS_SCHEDULED in sub.columns:
                scheduled[mode] = float(sub[cm.DAYS_SCHEDULED].median())

    return RateTable(
        global_rate=global_rate,
        mode_rate={str(k): float(v) for k, v in mode_rate.items()},
        cell_rate=cell_rate,
        transit_p=transit_p,
        scheduled=scheduled,
    )
