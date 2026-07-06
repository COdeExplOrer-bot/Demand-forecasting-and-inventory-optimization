# AI-Based Demand Forecasting & Smart Inventory Optimization

A demand forecasting and inventory-optimization pipeline for e-commerce
SKUs, built with Python, Pandas, and Scikit-learn. Given historical
daily sales, it forecasts near-term demand per SKU and converts those
forecasts into concrete inventory decisions — safety stock, reorder
point, and economic order quantity (EOQ) — instead of stopping at a
forecast number that no one can act on.

## Problem

E-commerce platforms carry thousands of SKUs with highly variable
demand: weekday/weekend swings, and sharp multi-week spikes during
promotional windows (e.g. Big Billion Days, Diwali, year-end sales).
Under-forecasting during these windows causes stockouts and lost
sales; over-forecasting ties up working capital in excess inventory.
This project builds a small, end-to-end pipeline that addresses both
sides of that problem.

## Approach

**1. Data.** Since a real Flipkart-scale transaction log isn't
publicly available, `src/generate_data.py` generates a synthetic
multi-SKU daily sales dataset (3 categories, 2 years) with the same
statistical structure a real dataset would have: an underlying growth
trend, weekend seasonality, and sharp promotional-season spikes.

**2. EDA.** `src/eda.py` explores trend, weekday seasonality, and
category-level demand variability using Pandas and Matplotlib.

**3. Feature engineering.** `src/features.py` reframes the time
series as a supervised-learning problem: lag features (t-1, t-2, t-3,
t-7, t-14), rolling mean/std (7/14/30-day windows), calendar features
(day of week, weekend flag, month), and a `is_sale_season` flag
representing a pre-announced promotional calendar (a legitimate,
non-leaky signal — e-commerce platforms know their own sale dates in
advance).

**4. Forecasting.** `src/forecast.py` trains a `RandomForestRegressor`
and `GradientBoostingRegressor` per SKU, using a **time-ordered**
train/test split (shuffling a time series would leak future
information into training). Both are compared against a naive
"yesterday's value" baseline using MAE, RMSE, and MAPE.

**5. Inventory optimization.** `src/inventory_optimization.py` turns
the forecast into action: safety stock sized to a 95% service level
(using the model's forecast error as the demand-uncertainty term),
reorder point, and EOQ using the standard inventory-management
formulas.

## Results

RandomForest beat the naive baseline on RMSE and MAPE for all three
SKUs in the test window:

| SKU | Model | MAE | RMSE | MAPE % |
|---|---|---|---|---|
| SKU_ELEC_001 | Naive Baseline | 39.45 | 70.60 | 15.83 |
| SKU_ELEC_001 | **RandomForest** | 44.03 | **64.68** | **14.24** |
| SKU_FASH_001 | Naive Baseline | 71.08 | 136.19 | 18.28 |
| SKU_FASH_001 | **RandomForest** | **58.91** | **101.64** | **11.87** |
| SKU_HOME_001 | Naive Baseline | 16.62 | 24.30 | 13.25 |
| SKU_HOME_001 | **RandomForest** | **12.75** | **17.76** | **9.46** |

Full metrics: [`results/forecast_metrics.csv`](results/forecast_metrics.csv)

Sample forecast plot:

![Forecast vs Actual](results/forecast_plot_SKU_FASH_001.png)

Resulting inventory plan ([`results/inventory_plan.csv`](results/inventory_plan.csv)):

| SKU | Avg Daily Demand | Safety Stock | Reorder Point | EOQ |
|---|---|---|---|---|
| SKU_ELEC_001 | 175.9 | 238 | 1,117 | 896 |
| SKU_FASH_001 | 285.9 | 374 | 1,803 | 1,142 |
| SKU_HOME_001 | 101.4 | 65 | 572 | 680 |

## Project structure

```
demand-forecasting-inventory-optimization/
├── data/
│   └── sales_data.csv              # generated synthetic dataset
├── src/
│   ├── generate_data.py            # synthetic data generator
│   ├── features.py                 # lag/rolling/calendar feature engineering
│   ├── eda.py                      # exploratory data analysis + plots
│   ├── forecast.py                 # model training + evaluation
│   ├── inventory_optimization.py   # safety stock / reorder point / EOQ
│   └── main.py                     # runs the full pipeline end-to-end
├── results/                        # generated plots + metrics CSVs
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python src/main.py
```

This regenerates the dataset, EDA plots, forecast metrics/plots, and
the inventory plan from scratch into `results/`.

## Possible extensions

- Swap the tree-based models for a proper time-series model (Prophet,
  SARIMA) and compare
- Hyperparameter tuning via `GridSearchCV` with time-series cross-validation
- Extend to more SKUs and a hierarchical (category-level +
  SKU-level) forecast reconciliation
- Serve the forecast via a small FastAPI endpoint for integration
  with a live inventory system
