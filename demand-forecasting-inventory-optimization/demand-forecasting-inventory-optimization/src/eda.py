"""
eda.py
------
Exploratory data analysis on the raw sales data: overall trend,
weekly seasonality, and category-level demand comparison. Produces
the plots referenced in the project README under "EDA".

Run:
    python src/eda.py
Output:
    results/eda_trend.png
    results/eda_weekday_seasonality.png
    results/eda_category_comparison.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
RESULTS_DIR = ROOT / "results"


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # 1. Overall demand trend by category (7-day rolling mean to smooth noise)
    plt.figure(figsize=(10, 4))
    for sku_id, g in df.groupby("sku_id"):
        g = g.sort_values("date")
        rolling = g.set_index("date")["units_sold"].rolling(7).mean()
        plt.plot(rolling.index, rolling.values, label=sku_id)
    plt.title("7-Day Rolling Average Demand by SKU")
    plt.xlabel("Date")
    plt.ylabel("Units Sold (smoothed)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_trend.png", dpi=120)
    plt.close()

    # 2. Weekday seasonality
    df["day_of_week"] = df["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = df.groupby(["day_of_week", "sku_id"])["units_sold"].mean().unstack()
    weekday_avg = weekday_avg.reindex(order)

    plt.figure(figsize=(9, 4.5))
    weekday_avg.plot(kind="bar", ax=plt.gca())
    plt.title("Average Demand by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Units Sold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_weekday_seasonality.png", dpi=120)
    plt.close()

    # 3. Category comparison (total volume + variability)
    summary = df.groupby("category")["units_sold"].agg(["mean", "std", "sum"]).round(1)
    summary = summary.rename(columns={"mean": "avg_daily_units", "std": "std_dev", "sum": "total_units_2yr"})

    plt.figure(figsize=(7, 4))
    summary["avg_daily_units"].plot(kind="bar", color="#2E86AB")
    plt.title("Average Daily Demand by Category")
    plt.ylabel("Avg Units Sold / Day")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_category_comparison.png", dpi=120)
    plt.close()

    print("Category-level summary statistics:\n")
    print(summary.to_string())
    print(f"\nSaved plots -> {RESULTS_DIR}/eda_*.png")


if __name__ == "__main__":
    run()
