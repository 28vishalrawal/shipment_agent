"""Generate synthetic but realistic DataCo-shaped data. No real PII."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
REGIONS = ["Western Europe", "Central America", "South America", "Pacific Asia",
           "Southern Europe", "Oceania", "West Asia", "Caribbean"]
MARKETS = {"Western Europe": "Europe", "Southern Europe": "Europe",
           "Central America": "LATAM", "South America": "LATAM", "Caribbean": "LATAM",
           "Pacific Asia": "Pacific Asia", "Oceania": "Pacific Asia", "West Asia": "Pacific Asia"}
CATEGORIES = ["Cleats", "Electronics", "Cardio Equipment", "Water Sports", "Accessories"]
DEPARTMENTS = {"Cleats": "Footwear", "Electronics": "Technology",
               "Cardio Equipment": "Fitness", "Water Sports": "Outdoors",
               "Accessories": "Accessories"}
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
OPEN_STATUS = ["PENDING", "PROCESSING", "ON_HOLD", "PAYMENT_REVIEW"]

# Scheduled days per mode and the (uniform) actual transit ranges, reproducing
# the structural SLA mismatch: First Class promises 1 day but always ships in 2.
SCHED = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
ACTUAL_RANGE = {"Standard Class": (2, 6), "Second Class": (2, 6),
                "First Class": (2, 2), "Same Day": (0, 1)}


def generate(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mode = rng.choice(MODES, n, p=[0.58, 0.19, 0.15, 0.08])
    region = rng.choice(REGIONS, n)
    category = rng.choice(CATEGORIES, n)
    segment = rng.choice(SEGMENTS, n, p=[0.55, 0.30, 0.15])
    sched = np.array([SCHED[m] for m in mode])
    actual = np.array([rng.integers(*ACTUAL_RANGE[m], endpoint=True) for m in mode])
    late = (actual > sched).astype(int)
    status_delivery = np.where(late == 1, "Late delivery",
                               rng.choice(["Advance shipping", "Shipping on time"], n))
    # ~4% cancelled, ~40% still open (no delivery outcome used for those)
    order_status = rng.choice(["COMPLETE"] + OPEN_STATUS, n, p=[0.6, 0.15, 0.13, 0.07, 0.05])
    dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")

    df = pd.DataFrame({
        "Order Id": np.arange(1, n + 1),
        "Delivery Status": status_delivery,
        "Late_delivery_risk": late,
        "Shipping Mode": mode,
        "Order Region": region,
        "Market": [MARKETS[r] for r in region],
        "Category Name": category,
        "Department Name": [DEPARTMENTS[c] for c in category],
        "Customer Segment": segment,
        "Days for shipment (scheduled)": sched,
        "Days for shipping (real)": actual,
        "Order Item Total": np.round(rng.gamma(2.0, 60, n), 2),
        "Benefit per order": np.round(rng.normal(22, 40, n), 2),
        "Order Status": order_status,
        "order date (DateOrders)": dates,
        "product_name": [f"{c} item" for c in category],
        "quantity": rng.integers(1, 5, n),
    })
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", default="data/synthetic_orders.csv")
    args = ap.parse_args()
    generate(args.rows).to_csv(args.out, index=False)
    print(f"wrote {args.rows} rows to {args.out}")
