"""
features.py
-----------
Feature engineering for the demand forecasting model. Converts a raw
daily sales series per SKU into a supervised-learning table using lag
features, rolling statistics, and calendar features -- the standard
approach for framing time-series forecasting as a regression problem
that tree-based Scikit-learn models can consume.
"""

import pandas as pd

LAGS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [7, 14, 30]


def _is_sale_season(date: pd.Timestamp) -> bool:
    """
    Flags dates that fall within known, pre-announced e-commerce
    promotional windows (e.g. Big Billion Days / Diwali / year-end
    sale). In production this would come from a marketing calendar
    published ahead of time -- a realistic, non-leaky feature since
    these dates are known in advance, not derived from the target.
    """
    month, day = date.month, date.day
    if month == 10:
        return True
    if month == 11 and day <= 5:
        return True
    if month == 12 and day >= 25:
        return True
    if month == 1 and day <= 1:
        return True
    return False


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: columns [date, sku_id, category, units_sold], one row per SKU per day.
    Returns the same frame enriched with lag, rolling, and calendar features.
    Rows where lag/rolling features can't be computed (start of series) are dropped.
    """
    df = df.sort_values(["sku_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    out_frames = []
    for sku_id, g in df.groupby("sku_id"):
        g = g.sort_values("date").copy()

        for lag in LAGS:
            g[f"lag_{lag}"] = g["units_sold"].shift(lag)

        for window in ROLLING_WINDOWS:
            g[f"roll_mean_{window}"] = g["units_sold"].shift(1).rolling(window).mean()
            g[f"roll_std_{window}"] = g["units_sold"].shift(1).rolling(window).std()

        g["day_of_week"] = g["date"].dt.dayofweek
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        g["day_of_month"] = g["date"].dt.day
        g["month"] = g["date"].dt.month
        g["day_of_year"] = g["date"].dt.dayofyear
        g["is_sale_season"] = g["date"].apply(_is_sale_season).astype(int)

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    result = result.dropna().reset_index(drop=True)
    return result


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + ["day_of_week", "is_weekend", "day_of_month", "month", "day_of_year", "is_sale_season"]
)
TARGET_COLUMN = "units_sold"
