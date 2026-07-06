"""
generate_data.py
----------------
Generates a synthetic, multi-SKU daily sales dataset that mimics common
e-commerce demand patterns: overall growth trend, weekly seasonality
(weekend spikes), yearly seasonality (festive/holiday season demand
spikes similar to Big Billion Days / Diwali sales), and product-level
noise. This stands in for a real Flipkart-scale transactional dataset,
which isn't publicly available, while preserving the same statistical
properties a forecasting model needs to learn from.

Run:
    python src/generate_data.py
Output:
    data/sales_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"

# Simulate 3 product categories with different demand profiles
SKUS = [
    {"sku_id": "SKU_ELEC_001", "category": "Electronics", "base_demand": 120, "trend_per_day": 0.05, "festive_boost": 2.8, "noise_std": 12},
    {"sku_id": "SKU_FASH_001", "category": "Fashion",     "base_demand": 200, "trend_per_day": 0.03, "festive_boost": 3.5, "noise_std": 20},
    {"sku_id": "SKU_HOME_001", "category": "Home & Kitchen", "base_demand": 80, "trend_per_day": 0.02, "festive_boost": 1.8, "noise_std": 8},
]

START_DATE = "2023-01-01"
NUM_DAYS = 730  # 2 years of daily data


def is_festive_window(date: pd.Timestamp) -> float:
    """Returns a multiplier > 1 during major Indian e-commerce sale windows."""
    month, day = date.month, date.day
    # Big Billion Days-style window (early-mid October)
    if month == 10 and 1 <= day <= 15:
        return 1.0
    # Diwali run-up (varies year to year, approximated to late Oct/early Nov)
    if (month == 10 and day >= 20) or (month == 11 and day <= 5):
        return 1.0
    # End-of-year sale (Dec 25 - Jan 1)
    if (month == 12 and day >= 25) or (month == 1 and day <= 1):
        return 0.6
    return 0.0


def generate() -> pd.DataFrame:
    dates = pd.date_range(start=START_DATE, periods=NUM_DAYS, freq="D")
    rows = []

    for sku in SKUS:
        for t, date in enumerate(dates):
            weekday = date.dayofweek  # 0=Mon ... 6=Sun
            weekend_multiplier = 1.25 if weekday >= 5 else 1.0

            festive_strength = is_festive_window(date)
            festive_multiplier = 1.0 + festive_strength * (sku["festive_boost"] - 1.0)

            trend = sku["base_demand"] + sku["trend_per_day"] * t
            noise = np.random.normal(0, sku["noise_std"])

            demand = trend * weekend_multiplier * festive_multiplier + noise
            demand = max(0, round(demand))

            rows.append({
                "date": date,
                "sku_id": sku["sku_id"],
                "category": sku["category"],
                "units_sold": demand,
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows across {df['sku_id'].nunique()} SKUs -> {OUTPUT_PATH}")
