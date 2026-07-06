"""
forecast.py
-----------
Trains and evaluates per-SKU demand forecasting models using
Scikit-learn (RandomForestRegressor, GradientBoostingRegressor),
compared against a naive baseline (yesterday's value). Uses a
time-ordered train/test split (no shuffling -- shuffling a time
series would leak future information into training).

Run:
    python src/forecast.py
Outputs:
    results/forecast_metrics.csv
    results/forecast_plot_<sku_id>.png  (actual vs predicted, per SKU)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
RESULTS_DIR = ROOT / "results"
TEST_FRACTION = 0.15


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def time_ordered_split(g: pd.DataFrame, test_fraction: float):
    n_test = max(1, int(len(g) * test_fraction))
    train = g.iloc[:-n_test]
    test = g.iloc[-n_test:]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 2),
        "MAPE_%": round(mape(y_true, y_pred), 2),
    }


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    feat_df = build_features(raw)

    all_metrics = []

    for sku_id, g in feat_df.groupby("sku_id"):
        train, test = time_ordered_split(g, TEST_FRACTION)

        X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
        X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

        # Naive baseline: predict yesterday's actual value (lag_1)
        baseline_pred = X_test["lag_1"].values
        baseline_metrics = evaluate(y_test, baseline_pred)

        rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_metrics = evaluate(y_test, rf_pred)

        gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        gbr.fit(X_train, y_train)
        gbr_pred = gbr.predict(X_test)
        gbr_metrics = evaluate(y_test, gbr_pred)

        for model_name, m in [("Naive Baseline", baseline_metrics),
                               ("RandomForest", rf_metrics),
                               ("GradientBoosting", gbr_metrics)]:
            all_metrics.append({"sku_id": sku_id, "model": model_name, **m})

        # Plot actual vs. best model prediction (RandomForest) for this SKU
        plt.figure(figsize=(10, 4))
        plt.plot(test["date"].values, y_test.values, label="Actual", linewidth=1.8)
        plt.plot(test["date"].values, rf_pred, label="RandomForest Forecast", linewidth=1.8, linestyle="--")
        plt.title(f"Demand Forecast vs. Actual — {sku_id}")
        plt.xlabel("Date")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"forecast_plot_{sku_id}.png", dpi=120)
        plt.close()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "forecast_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    print(f"\nSaved metrics -> {RESULTS_DIR / 'forecast_metrics.csv'}")
    print(f"Saved plots   -> {RESULTS_DIR}/forecast_plot_<sku_id>.png")


if __name__ == "__main__":
    run()
