"""
inventory_optimization.py
--------------------------
Turns demand forecasts into concrete inventory decisions:
  - Safety stock (buffer for demand variability during lead time)
  - Reorder point (inventory level that triggers a new purchase order)
  - Economic Order Quantity (EOQ) -- the order size that minimizes
    total holding + ordering cost

This is the "so what" layer on top of the forecasting model: a
forecast alone doesn't tell a supply-chain team when or how much to
reorder. These formulas are the standard textbook inventory-management
equations, parameterized here with the model's forecasted demand and
its residual error (used as a proxy for demand uncertainty).

Run:
    python src/inventory_optimization.py
Output:
    results/inventory_plan.csv
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm  # ships with scikit-learn's dependency chain

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
METRICS_PATH = ROOT / "results" / "forecast_metrics.csv"
OUTPUT_PATH = ROOT / "results" / "inventory_plan.csv"

# Business assumptions (would normally come from procurement/finance)
LEAD_TIME_DAYS = 5
SERVICE_LEVEL = 0.95          # probability of not stocking out during lead time
ORDERING_COST_PER_ORDER = 50  # currency units per purchase order placed
HOLDING_COST_PER_UNIT_PER_YEAR = 8


def z_score_for_service_level(service_level: float) -> float:
    return float(norm.ppf(service_level))


def compute_plan() -> pd.DataFrame:
    sales = pd.read_csv(DATA_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    rf_metrics = metrics[metrics["model"] == "RandomForest"].set_index("sku_id")

    z = z_score_for_service_level(SERVICE_LEVEL)
    rows = []

    for sku_id, g in sales.groupby("sku_id"):
        avg_daily_demand = g["units_sold"].mean()
        annual_demand = avg_daily_demand * 365

        # Use the model's RMSE as a proxy for daily forecast-error std dev
        demand_std = rf_metrics.loc[sku_id, "RMSE"] if sku_id in rf_metrics.index else g["units_sold"].std()

        safety_stock = z * demand_std * math.sqrt(LEAD_TIME_DAYS)
        reorder_point = avg_daily_demand * LEAD_TIME_DAYS + safety_stock

        eoq = math.sqrt(
            (2 * annual_demand * ORDERING_COST_PER_ORDER) / HOLDING_COST_PER_UNIT_PER_YEAR
        )

        rows.append({
            "sku_id": sku_id,
            "avg_daily_demand": round(avg_daily_demand, 1),
            "forecast_rmse": round(float(demand_std), 1),
            "safety_stock_units": round(safety_stock, 0),
            "reorder_point_units": round(reorder_point, 0),
            "economic_order_qty_units": round(eoq, 0),
            "service_level_target": SERVICE_LEVEL,
            "lead_time_days": LEAD_TIME_DAYS,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    plan = compute_plan()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUTPUT_PATH, index=False)
    print(plan.to_string(index=False))
    print(f"\nSaved inventory plan -> {OUTPUT_PATH}")
