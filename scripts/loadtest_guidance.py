"""Load-test guidance for ~180k orders.

Lane B is pure computation and scales with candidate count, not row count, so it
stays fast even at full size. Lane A latency is dominated by LLM calls; the cost
is (#notifications after queue cap) x per-call latency, bounded by concurrency.

Run:  python scripts/loadtest_guidance.py --rows 180000
"""
from __future__ import annotations

import argparse
import time

from app.analytics.ingest import ingest
from app.analytics.rate_table import build_rate_table
from app.analytics.root_cause import GateParams, run_root_cause
from app.analytics.triage import score_open_orders
from scripts.make_synthetic_data import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=180000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = generate(args.rows, seed=11)
    t_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = ingest(df)
    t_ingest = time.perf_counter() - t0

    t0 = time.perf_counter()
    rt = build_rate_table(res.closed)
    triage = score_open_orders(res.open_orders, rt, 50, 75, 200)
    t_triage = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = run_root_cause(res.closed, GateParams(), avg_margin=22.0)
    t_lb = time.perf_counter() - t0

    print(f"rows                : {args.rows:,}")
    print(f"generate            : {t_gen:6.2f}s")
    print(f"ingest              : {t_ingest:6.2f}s")
    print(f"lane A (score+prio) : {t_triage:6.2f}s   triaged={len(triage)}")
    print(f"lane B (5 gates)    : {t_lb:6.2f}s   candidates={out.candidates_enumerated} "
          f"M={out.m_tests_conducted} findings={len(out.findings)}")
    print()
    print("Lane A LLM projection (notifications after cap = min(triaged, 200)):")
    n_notif = min(len(triage), 200)
    for lat in (0.6, 1.2):
        for conc in (10, 20):
            secs = n_notif / conc * lat
            print(f"  {n_notif} notifs @ {lat}s/call, concurrency {conc:2d} -> {secs:5.1f}s")
    print("\nGuidance:")
    print("- Lane B is CPU-bound and completes in seconds at 180k; no queue needed.")
    print("- Lane A: cap the queue (done) and raise concurrency to trade latency for")
    print("  provider rate-limit headroom. Move to a worker + DLQ for large batches.")


if __name__ == "__main__":
    main()
