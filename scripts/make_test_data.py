"""Rich synthetic DataCo-shaped generator for multi-run testing.

Produces ~100k orders with the same STRUCTURE as the real dataset, plus a set of
deliberately planted signals so each analytic gate has something to find and
something to reject. No real PII. Columns match the canonical mapping.

Planted signals (what a correct pipeline should conclude):
  * First Class  -> structurally 100% late (scheduled 1, actual always 2).
    Should be finding #1, clears all five gates.
  * Second Class -> ~80% late (scheduled 2, actual uniform 2-6).
    Should be finding #2.
  * Standard/Same Day -> lower late rates (protective factors).
  * A late-onboarded region ("West Asia") with NO orders before a cutoff date,
    so it must fail Gate 5 (no_history_in_one_half) -> stays in rejected register.
  * A transient anomaly region ("US Center") whose late rate spikes only in the
    first half, then normalises -> must fail Gate 5 (too_variable).
  * Category / segment effects kept near baseline -> must fail Gate 2 (effect
    size) or Gate 4 (confound), proving mode is the true driver.

Usage:
  python scripts/make_test_data.py --rows 100000 --out data/test_100k.csv
  python scripts/make_test_data.py --rows 100000 --runs 5 --outdir data/tests
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
MODE_P = [0.58, 0.19, 0.15, 0.08]

# Region -> Market (nested, like the real export).
REGION_MARKET = {
    "Western Europe": "Europe", "Southern Europe": "Europe", "Northern Europe": "Europe",
    "Central America": "LATAM", "South America": "LATAM", "Caribbean": "LATAM",
    "Pacific Asia": "Pacific Asia", "Oceania": "Pacific Asia", "Southeast Asia": "Pacific Asia",
    "US Center": "USCA", "West of USA": "USCA", "East of USA": "USCA",
    "West Africa": "Africa", "North Africa": "Africa",
    "West Asia": "Pacific Asia",  # late-onboarded region
}
REGIONS = list(REGION_MARKET)

# Category -> Department (nested).
CATEGORY_DEPT = {
    "Cleats": "Footwear", "Men's Footwear": "Footwear",
    "Electronics": "Technology", "Computers": "Technology",
    "Cardio Equipment": "Fitness", "Strength Training": "Fitness",
    "Water Sports": "Outdoors", "Camping & Hiking": "Outdoors",
    "Accessories": "Accessories", "Garden": "Outdoors",
}
CATEGORIES = list(CATEGORY_DEPT)

SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SEGMENT_P = [0.55, 0.30, 0.15]
OPEN_STATUS = ["PENDING", "PENDING_PAYMENT", "PROCESSING", "ON_HOLD", "PAYMENT_REVIEW"]

SCHED = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
ACTUAL_RANGE = {
    "Standard Class": (2, 6), "Second Class": (2, 6),
    "First Class": (2, 2), "Same Day": (0, 1),
}

# Timeline: 2015-01-01 .. ~2018 (about 1150 days), split-half near mid-2016.
START = pd.Timestamp("2015-01-01")
DAYS_SPAN = 1150
CUTOFF_DAY = 700           # West Asia only appears after this day (> median split)
FIRST_HALF_MAX = 575       # US Center anomaly lives in the first half


def generate(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    mode = rng.choice(MODES, n, p=MODE_P)
    region = rng.choice(REGIONS, n)
    category = rng.choice(CATEGORIES, n)
    segment = rng.choice(SEGMENTS, n, p=SEGMENT_P)
    day = rng.integers(0, DAYS_SPAN, n)

    # West Asia is late-onboarded: ALL its orders occur strictly after the
    # cutoff, so it has zero first-half history -> triggers the ingestion
    # "regions_absent_first_half" data-quality flag and Gate 5 no_history.
    wa = region == "West Asia"
    day = np.where(wa, rng.integers(CUTOFF_DAY, DAYS_SPAN, n), day)

    sched = np.array([SCHED[m] for m in mode])
    actual = np.array([rng.integers(*ACTUAL_RANGE[m], endpoint=True) for m in mode])
    late = (actual > sched).astype(int)

    # Concentrated, time-STABLE regional anomaly: North Africa on Standard Class
    # runs persistently later across both halves, so it clears support + effect +
    # confound + stability -> a genuine region finding beyond mode. Exercises the
    # full validated-finding path for a non-mode dimension.
    na_std = (region == "North Africa") & (mode == "Standard Class")
    late = np.where(na_std & (rng.random(n) < 0.30), 1, late)

    # Transient anomaly: US Center in the FIRST half only gets extra lateness,
    # then normalises -> must fail temporal stability (Gate 5, too_variable).
    usc_first = (region == "US Center") & (day < FIRST_HALF_MAX)
    late = np.where(usc_first & (rng.random(n) < 0.35), 1, late)

    delivery_status = np.where(
        late == 1, "Late delivery",
        rng.choice(["Advance shipping", "Shipping on time"], n, p=[0.56, 0.44]),
    )

    # ~4% cancelled overrides delivery status (no outcome).
    cancel = rng.random(n) < 0.04
    delivery_status = np.where(cancel, "Shipping canceled", delivery_status)

    order_status = rng.choice(["COMPLETE"] + OPEN_STATUS, n,
                              p=[0.58, 0.13, 0.11, 0.09, 0.05, 0.04])
    dates = START + pd.to_timedelta(day, unit="D")

    df = pd.DataFrame({
        "Order Id": np.arange(1, n + 1),
        "Delivery Status": delivery_status,
        "Late_delivery_risk": late,
        "Shipping Mode": mode,
        "Order Region": region,
        "Market": [REGION_MARKET[r] for r in region],
        "Category Name": category,
        "Department Name": [CATEGORY_DEPT[c] for c in category],
        "Customer Segment": segment,
        "Days for shipment (scheduled)": sched,
        "Days for shipping (real)": actual,
        "Order Item Total": np.round(rng.gamma(2.0, 60, n), 2),
        "Benefit per order": np.round(rng.normal(22, 40, n), 2),
        "Sales": np.round(rng.gamma(2.0, 65, n), 2),
        "Order Status": order_status,
        "order date (DateOrders)": dates,
        "Customer Country": rng.choice(["USA", "Germany", "Brazil", "India", "Mexico"], n),
        "Customer State": rng.choice(["CA", "NY", "TX", "BY", "SP"], n),
        "product_name": [f"{c} item" for c in category],
        "quantity": rng.integers(1, 5, n),
    })
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/test_100k.csv")
    ap.add_argument("--runs", type=int, default=1,
                    help="generate this many files with different seeds")
    ap.add_argument("--outdir", default="data/tests")
    args = ap.parse_args()

    if args.runs > 1:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.runs):
            df = generate(args.rows, seed=args.seed + i)
            path = out / f"test_{args.rows}_seed{args.seed + i}.csv"
            df.to_csv(path, index=False)
            print(f"wrote {len(df):,} rows -> {path}")
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df = generate(args.rows, seed=args.seed)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
