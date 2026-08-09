"""Configurable column-mapping layer.

Real source exports rename columns constantly. Every downstream module refers to
CANONICAL names only; this layer resolves whatever the source file happens to use.
Assumption: the DataCo standard export has NO carrier column, only a shipping mode.
The system therefore never emits a "carrier" claim (see domain.CarrierPolicy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

# Canonical field names used everywhere in the codebase.
ORDER_ID = "order_id"
DELIVERY_STATUS = "delivery_status"
LATE_RISK_FLAG = "late_delivery_risk"
SHIPPING_MODE = "shipping_mode"
ORDER_REGION = "order_region"
MARKET = "market"
CATEGORY = "category"
DEPARTMENT = "department"
CUSTOMER_SEGMENT = "customer_segment"
CUSTOMER_COUNTRY = "customer_country"
CUSTOMER_STATE = "customer_state"
DAYS_SCHEDULED = "days_scheduled"
DAYS_REAL = "days_real"
ORDER_ITEM_TOTAL = "order_item_total"
BENEFIT_PER_ORDER = "benefit_per_order"
SALES = "sales"
ORDER_STATUS = "order_status"
ORDER_DATE = "order_date"

# Default mapping: canonical -> list of accepted source aliases (first match wins).
DEFAULT_ALIASES: dict[str, list[str]] = {
    ORDER_ID: ["Order Id", "OrderID", "order_id"],
    DELIVERY_STATUS: ["Delivery Status", "delivery_status"],
    LATE_RISK_FLAG: ["Late_delivery_risk", "late_delivery_risk"],
    SHIPPING_MODE: ["Shipping Mode", "shipping_mode"],
    ORDER_REGION: ["Order Region", "order_region"],
    MARKET: ["Market", "market"],
    CATEGORY: ["Category Name", "category_name", "Category"],
    DEPARTMENT: ["Department Name", "department_name", "Department"],
    CUSTOMER_SEGMENT: ["Customer Segment", "customer_segment", "Segment"],
    CUSTOMER_COUNTRY: ["Customer Country", "customer_country"],
    CUSTOMER_STATE: ["Customer State", "customer_state"],
    DAYS_SCHEDULED: ["Days for shipment (scheduled)", "days_scheduled"],
    DAYS_REAL: ["Days for shipping (real)", "days_real"],
    ORDER_ITEM_TOTAL: ["Order Item Total", "order_item_total"],
    BENEFIT_PER_ORDER: ["Benefit per order", "benefit_per_order", "Order Profit Per Order"],
    SALES: ["Sales", "sales"],
    ORDER_STATUS: ["Order Status", "order_status"],
    ORDER_DATE: ["order date (DateOrders)", "Order Date", "order_date"],
}

REQUIRED_FIELDS = [ORDER_ID, DELIVERY_STATUS, SHIPPING_MODE, ORDER_REGION]


@dataclass
class ColumnMapping:
    aliases: Mapping[str, list[str]] = field(default_factory=lambda: DEFAULT_ALIASES)

    def resolve(self, df_columns: list[str]) -> dict[str, str]:
        """Return {canonical: actual_source_column} for every field we can find."""
        # Whitespace-tolerant lookup: source files often carry trailing spaces.
        stripped = {c.strip().lower(): c for c in df_columns}
        resolved: dict[str, str] = {}
        for canonical, options in self.aliases.items():
            for opt in options:
                key = opt.strip().lower()
                if key in stripped:
                    resolved[canonical] = stripped[key]
                    break
        return resolved

    def missing_required(self, resolved: dict[str, str]) -> list[str]:
        return [f for f in REQUIRED_FIELDS if f not in resolved]

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """Rename source columns to canonical names; return (df, resolved_map)."""
        resolved = self.resolve(list(df.columns))
        missing = self.missing_required(resolved)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        inverse = {src: canon for canon, src in resolved.items()}
        out = df.rename(columns=inverse)
        # keep only canonical columns we know about
        keep = [c for c in out.columns if c in self.aliases]
        return out[keep].copy(), resolved
