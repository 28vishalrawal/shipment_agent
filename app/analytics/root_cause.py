"""Deterministic systemic root-cause engine (Lane B core). No LLM anywhere.

Pipeline:
  enumerate -> Gate0 support -> Stage1 baseline -> Gate2 effect -> Gate3 FDR
  -> Gate4 confound -> Gate5 stability -> confidence -> rank.

Every rejected candidate is retained with its failing gate. M (the number of
tests entering FDR) is fixed at the support floor and reported, because
correcting against the wrong M invalidates the false-discovery guarantee.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from app.core import column_mapping as cm
from app.domain.models import (
    EvidenceGrade,
    GateResult,
    RejectedCandidate,
    ValidatedFinding,
)

# Dimensions available for segmentation. Shipping Mode included by default; the
# caller may drop it to search for non-mode structure.
DEFAULT_DIMS = [
    cm.SHIPPING_MODE,
    cm.ORDER_REGION,
    cm.MARKET,
    cm.CATEGORY,
    cm.DEPARTMENT,
    cm.CUSTOMER_SEGMENT,
]


@dataclass
class GateParams:
    support_floor: int = 200
    effect_size_min: float = 0.15
    fdr_q: float = 0.05
    confound_margin: float = 0.15
    stability_var_max: float = 0.30


@dataclass
class RootCauseOutput:
    findings: list[ValidatedFinding]
    rejected: list[RejectedCandidate]
    candidates_enumerated: int
    m_tests_conducted: int
    global_rate: float
    protective: list[ValidatedFinding] = field(default_factory=list)
    # Strongest candidate segments by |excess late orders|, captured BEFORE the
    # strict gates, so callers can always surface the top signals (with metrics
    # and their eventual validation status) even when none clear full validation.
    top_candidates: list[dict] = field(default_factory=list)


def _key_str(dims: tuple[str, ...], values: tuple) -> str:
    return " | ".join(f"{d}={v}" for d, v in zip(dims, values))


def run_root_cause(
    closed: pd.DataFrame,
    params: GateParams,
    dims: list[str] | None = None,
    avg_margin: float = 0.0,
) -> RootCauseOutput:
    dims = [d for d in (dims or DEFAULT_DIMS) if d in closed.columns]
    G = float(closed["late"].mean())

    # --- main-effect strength drives baseline (parent) selection ---
    main_eff = {d: float(closed.groupby(d)["late"].mean().std(ddof=0)) for d in dims}

    # cache group rate tables for parent lookups
    cache: dict[tuple[str, ...], pd.DataFrame] = {}

    def group(dset: tuple[str, ...]) -> pd.DataFrame:
        if dset not in cache:
            cache[dset] = closed.groupby(list(dset))["late"].agg(["size", "mean"])
        return cache[dset]

    def rate_of(dset: tuple[str, ...], key: tuple) -> float:
        if not dset:
            return G
        g = group(dset)
        k = key if len(dset) > 1 else key[0]
        if k in g.index:
            return float(np.asarray(g.loc[k, "mean"]).item())
        return float("nan")

    # --- enumerate candidates: 1-, 2-way over all dims; 3-way containing mode ---
    combos: list[tuple[str, ...]] = []
    for r in (1, 2):
        combos += list(itertools.combinations(dims, r))
    combos += [c for c in itertools.combinations(dims, 3) if cm.SHIPPING_MODE in c]

    rows = []
    for dset in combos:
        g = closed.groupby(list(dset))["late"].agg(n="size", rate="mean")
        for key, r in g.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"dims": dset, "key": key, "n": int(r["n"]), "rate": float(r["rate"])})
    cand = pd.DataFrame(rows)
    candidates_enumerated = len(cand)

    rejected: list[RejectedCandidate] = []

    def reject(row, gate: str, reason: str) -> None:
        rejected.append(
            RejectedCandidate(
                pattern_id=_key_str(row["dims"], row["key"]),
                dims=dict(zip(row["dims"], map(str, row["key"]))),
                failed_gate=gate,
                reason=reason,
                p_value=float(row["p"]) if "p" in row and pd.notna(row.get("p")) else None,
            )
        )

    # --- Gate 0: support floor ---
    below = cand[cand["n"] < params.support_floor]
    for _, row in below.iterrows():
        reject(row, "gate0_support", f"n={row['n']} < {params.support_floor}")
    cand = cand[cand["n"] >= params.support_floor].reset_index(drop=True)
    m_tests = len(cand)  # M fixed here

    if m_tests == 0:
        return RootCauseOutput([], rejected, candidates_enumerated, 0, G)

    # --- Stage 1: baseline assignment (drop strongest dim) ---
    def parent_rate(dset: tuple[str, ...], key: tuple) -> tuple[float, str]:
        if len(dset) == 1:
            return G, "GLOBAL"
        drop = max(dset, key=lambda x: main_eff.get(x, 0.0))
        pdm = tuple(x for x in dset if x != drop)
        pk = tuple(key[i] for i, x in enumerate(dset) if x != drop)
        return rate_of(pdm, pk), _key_str(pdm, pk)

    bases, base_keys = [], []
    for _, row in cand.iterrows():
        br, bk = parent_rate(row["dims"], row["key"])
        bases.append(br)
        base_keys.append(bk)
    cand["base"] = bases
    cand["base_key"] = base_keys
    cand["lift"] = cand["rate"] / cand["base"]
    cand["excess"] = cand["n"] * (cand["rate"] - cand["base"])

    # --- Gate 3 stats computed for all (needed for FDR over M) ---
    se = np.sqrt(cand["base"] * (1 - cand["base"]) / cand["n"])
    cand["z"] = (cand["rate"] - cand["base"]) / se
    cand["p"] = 2 * (1 - stats.norm.cdf(cand["z"].abs()))

    # Snapshot the strongest DELAY candidate segments (most excess LATE orders)
    # now, before the strict effect/FDR/confound/stability gates prune them.
    # Only positive-excess segments are delay contributors; negative-excess
    # segments are protective (they ship better than baseline) and are excluded.
    # Metrics only; the caller attaches each candidate's final validation status.
    _ranked = cand[cand["excess"] > 0].sort_values("excess", ascending=False)
    top_candidates = [
        {
            "pattern_id": _key_str(row["dims"], row["key"]),
            "dims": dict(zip(row["dims"], map(str, row["key"]))),
            "n": int(row["n"]),
            "rate": float(row["rate"]),
            "base": float(row["base"]),
            "lift": float(row["lift"]),
            "excess": float(row["excess"]),
            "p_value": float(row["p"]) if pd.notna(row["p"]) else None,
        }
        for _, row in _ranked.head(20).iterrows()
    ]

    # --- Gate 2: effect size ---
    passed_g2 = (cand["lift"] - 1).abs() >= params.effect_size_min
    for _, row in cand[~passed_g2].iterrows():
        reject(row, "gate2_effect", f"|lift-1|={abs(row['lift'] - 1):.3f} < {params.effect_size_min}")
    g2 = cand[passed_g2].copy()

    # --- Gate 3: Benjamini-Hochberg FDR over M ---
    srt = g2.sort_values("p").reset_index(drop=True)
    srt["rank"] = np.arange(1, len(srt) + 1)
    srt["bh"] = params.fdr_q * srt["rank"] / m_tests
    if (srt["p"] <= srt["bh"]).any():
        cut = int(srt.loc[srt["p"] <= srt["bh"], "rank"].max())
    else:
        cut = 0
    g3 = srt[srt["rank"] <= cut].copy()
    for _, row in srt[srt["rank"] > cut].iterrows():
        reject(row, "gate3_fdr", f"p={row['p']:.2e} > BH threshold (M={m_tests})")

    # --- Gate 4: confound / all-parents ---
    def confound_ok(dset: tuple[str, ...], key: tuple, r: float) -> bool:
        if len(dset) == 1:
            return abs(r / G - 1) >= params.confound_margin
        for drop in dset:
            pdm = tuple(x for x in dset if x != drop)
            pk = tuple(key[i] for i, x in enumerate(dset) if x != drop)
            br = rate_of(pdm, pk)
            # Guard: a parent with a zero (or NaN) late rate can't be divided
            # into. Treat it as a lift measured against a tiny epsilon so the
            # comparison stays well-defined instead of crashing.
            if np.isnan(br):
                return False
            if br == 0:
                # segment rate r against a 0% parent: if r>0 it's clearly worse,
                # otherwise both zero -> no effect.
                if r <= 0:
                    return False
            elif abs(r / br - 1) < params.confound_margin:
                return False
            if np.sign(r - br) != np.sign(r - G):
                return False
        return True

    g4_rows = []
    for _, row in g3.iterrows():
        if confound_ok(row["dims"], row["key"], row["rate"]):
            g4_rows.append(row)
        else:
            reject(row, "gate4_confound", "effect vanishes when conditioned on a parent")
    g4 = pd.DataFrame(g4_rows)

    # --- Gate 5: temporal stability ---
    def stability(dset, key, base) -> tuple[float, bool, str]:
        if "_half" not in closed.columns or closed["_half"].isna().all():
            return 0.5, True, "stability_skipped_no_date"
        m = pd.Series(True, index=closed.index)
        for d, v in zip(dset, key):
            m &= closed[d] == v
        s = closed[m]
        halves = {}
        for h, gg in s.groupby("_half"):
            if len(gg) >= 50:
                halves[int(h)] = float(gg["late"].mean()) - base
        if len(halves) < 2:
            return 0.25, False, "no_history_in_one_half"
        # Take the two present deltas directly (values are floats, never None).
        deltas = list(halves.values())
        d0, d1 = float(deltas[0]), float(deltas[1])
        if np.sign(d0) != np.sign(d1):
            return 0.0, False, "sign_flip"
        var = abs(d0 - d1) / max(abs((d0 + d1) / 2.0), 1e-9)
        ok = var < params.stability_var_max
        return max(0.0, 1 - var), ok, ("stable" if ok else "too_variable")

    findings: list[ValidatedFinding] = []
    protective: list[ValidatedFinding] = []
    if len(g4):
        for _, row in g4.iterrows():
            score, ok, why = stability(row["dims"], row["key"], row["base"])
            if not ok:
                reject(row, "gate5_stability", why)
                continue
            gates = [
                GateResult(gate="gate0_support", passed=True, reason=f"n={row['n']}"),
                GateResult(gate="gate2_effect", passed=True, reason=f"lift={row['lift']:.3f}"),
                GateResult(gate="gate3_fdr", passed=True, reason=f"p={row['p']:.2e}"),
                GateResult(gate="gate4_confound", passed=True, reason="survives all parents"),
                GateResult(gate="gate5_stability", passed=True, reason=why),
            ]
            sig = min(1.0, max(0.0, (-np.log10(max(row["p"], 1e-300))) / 10))
            samp = min(1.0, np.log10(row["n"] / params.support_floor + 1) / np.log10(20))
            eff = min(1.0, abs(row["excess"]) / 2000)
            confidence = 0.30 * sig + 0.20 * samp + 0.25 * eff + 0.15 * 1.0 + 0.10 * score

            finding = ValidatedFinding(
                pattern_id=_key_str(row["dims"], row["key"]),
                label=_key_str(row["dims"], row["key"]),
                dims=dict(zip(row["dims"], map(str, row["key"]))),
                n=int(row["n"]),
                seg_rate=float(row["rate"]),
                baseline_rate=float(row["base"]),
                lift=float(row["lift"]),
                excess_orders=float(row["excess"]),
                excess_margin=float(row["excess"] * avg_margin),
                p_value=float(row["p"]),
                confidence=float(confidence),
                gates=gates,
                evidence_grade=EvidenceGrade.DATA_SUPPORTED,
            )
            if finding.excess_orders >= 0:
                findings.append(finding)
            else:
                protective.append(finding)

    findings.sort(key=lambda f: abs(f.excess_orders) * f.confidence, reverse=True)
    protective.sort(key=lambda f: abs(f.excess_orders) * f.confidence, reverse=True)

    return RootCauseOutput(
        findings=findings,
        rejected=rejected,
        candidates_enumerated=candidates_enumerated,
        m_tests_conducted=m_tests,
        global_rate=G,
        protective=protective,
        top_candidates=top_candidates,
    )