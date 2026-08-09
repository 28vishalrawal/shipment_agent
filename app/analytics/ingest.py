"""Deterministic ingestion. No LLM. Produces the clean frames every lane uses."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

from app.core import column_mapping as cm
from app.domain.models import DeliveryOutcome

OPEN_STATUSES = {"PENDING", "PENDING_PAYMENT", "PROCESSING", "ON_HOLD", "PAYMENT_REVIEW"}


@dataclass
class IngestResult:
    closed: pd.DataFrame          # orders with a delivery outcome (analysis base)
    open_orders: pd.DataFrame     # in-flight orders (triage universe)
    input_rows: int
    analysis_rows: int
    global_late_rate: float
    input_hash: str
    resolved_columns: dict[str, str]
    data_quality_flags: list[str]


def _hash_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:16]


def ingest(df_raw: pd.DataFrame, mapping: cm.ColumnMapping | None = None) -> IngestResult:
    mapping = mapping or cm.ColumnMapping()
    df, resolved = mapping.apply(df_raw)
    input_rows = len(df)
    input_hash = _hash_frame(df)

    flags: list[str] = []

    # Whitespace hygiene on categorical joins (source carries trailing spaces).
    for col in [cm.ORDER_REGION, cm.MARKET, cm.CATEGORY, cm.DEPARTMENT,
                cm.CUSTOMER_SEGMENT, cm.SHIPPING_MODE]:
        if col in df.columns and df[col].dtype == object:
            before = df[col].astype(str)
            after = before.str.strip()
            if (before != after).any():
                flags.append(f"whitespace_stripped:{col}")
            df[col] = after

    # Binary late label from delivery status (deterministic, vendor-independent).
    df["late"] = (df[cm.DELIVERY_STATUS] == DeliveryOutcome.LATE.value).astype(int)

    # Analysis base excludes cancelled shipments (no delivery outcome).
    canceled = df[cm.DELIVERY_STATUS] == DeliveryOutcome.CANCELED.value
    closed = df[~canceled].copy()

    # Independence caveat: many rows can share one Order Id (line-items). If an
    # order id column exists, we keep line-items but flag the design effect.
    if cm.ORDER_ID in closed.columns:
        n_rows, n_orders = len(closed), closed[cm.ORDER_ID].nunique()
        if n_orders and n_rows / n_orders > 1.2:
            flags.append(f"line_item_grain:ratio={n_rows / n_orders:.2f}")

    # Open cohort for Lane A triage, if status present.
    if cm.ORDER_STATUS in df.columns:
        open_orders = df[df[cm.ORDER_STATUS].isin(OPEN_STATUSES)].copy()
    else:
        open_orders = df.iloc[0:0].copy()
        flags.append("no_order_status:triage_universe_empty")

    # Optional temporal column for stability gate.
    if cm.ORDER_DATE in closed.columns:
        closed["_dt"] = pd.to_datetime(closed[cm.ORDER_DATE], errors="coerce")
        mid = closed["_dt"].quantile(0.5)
        closed["_half"] = (closed["_dt"] >= mid).astype("Int64")
        # Detect dimension values absent from the first half (market expansion).
        if cm.ORDER_REGION in closed.columns:
            first_half = set(closed.loc[closed["_half"] == 0, cm.ORDER_REGION].unique())
            all_vals = set(closed[cm.ORDER_REGION].unique())
            late_arrivals = all_vals - first_half
            if late_arrivals:
                flags.append(f"regions_absent_first_half:{len(late_arrivals)}")
    else:
        closed["_half"] = pd.NA
        flags.append("no_order_date:stability_gate_skipped")

    global_late_rate = float(closed["late"].mean()) if len(closed) else 0.0

    return IngestResult(
        closed=closed,
        open_orders=open_orders,
        input_rows=input_rows,
        analysis_rows=len(closed),
        global_late_rate=global_late_rate,
        input_hash=input_hash,
        resolved_columns=resolved,
        data_quality_flags=flags,
    )
