"""
main.py
-------
Runs the full pipeline end-to-end:
  1. Generate synthetic sales data
  2. Exploratory data analysis
  3. Train + evaluate forecasting models
  4. Compute inventory optimization plan

Run:
    python src/main.py
"""

import generate_data
import eda
import forecast
import inventory_optimization


def main():
    print("=" * 60)
    print("STEP 1/4: Generating sales data")
    print("=" * 60)
    df = generate_data.generate()
    generate_data.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(generate_data.OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {generate_data.OUTPUT_PATH}\n")

    print("=" * 60)
    print("STEP 2/4: Exploratory Data Analysis")
    print("=" * 60)
    eda.run()
    print()

    print("=" * 60)
    print("STEP 3/4: Forecasting (RandomForest / GradientBoosting vs. baseline)")
    print("=" * 60)
    forecast.run()
    print()

    print("=" * 60)
    print("STEP 4/4: Inventory Optimization (safety stock, reorder point, EOQ)")
    print("=" * 60)
    plan = inventory_optimization.compute_plan()
    inventory_optimization.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(inventory_optimization.OUTPUT_PATH, index=False)
    print(plan.to_string(index=False))

    print("\nPipeline complete. See the results/ folder for plots and CSVs.")


if __name__ == "__main__":
    main()
