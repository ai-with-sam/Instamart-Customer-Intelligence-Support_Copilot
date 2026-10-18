
from pathlib import Path

import pandas as pd


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ORDERS_FILE = CLEAN_DIR / "orders.csv"
OUTPUT_FILE = OUTPUT_DIR / "rfm.csv"

# Required snapshot moment
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


def main():
    # 1. Load cleaned order data
    orders = pd.read_csv(ORDERS_FILE)

    # 2. Convert order timestamps to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # 3. Keep completed orders only
    completed = orders.loc[
        orders["status"].str.strip().str.lower() == "completed"
    ].copy()

    if completed.empty:
        raise ValueError("No completed orders found.")

    # 4. Calculate RFM for each customer
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    # 5. Recency is the number of whole days since the last order
    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    # 6. Keep the requested columns and order
    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary"] = rfm["monetary"].round(2)

    # 7. Save the result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUT_FILE, index=False)

    # 8. Validate against the project reference values
    assert len(rfm) == 1175, (
        f"Expected 1175 customers, got {len(rfm)}"
    )
    assert rfm["recency"].min() == 0
    assert rfm["recency"].max() == 180
    assert rfm["recency"].median() == 19
    assert round(rfm["frequency"].mean(), 2) == 9.46
    assert rfm["frequency"].max() == 33
    assert round(rfm["monetary"].mean(), 2) == 2552.06
    assert round(rfm["monetary"].max(), 2) == 14766.55
    assert not rfm.isna().any().any()

    # 9. Print a concise report
    print("RFM TABLE CREATED SUCCESSFULLY")
    print(f"Customers: {len(rfm):,}")
    print(f"Median recency: {rfm['recency'].median():.0f} days")
    print(
        f"Recency range: {rfm['recency'].min()}–"
        f"{rfm['recency'].max()} days"
    )
    print(f"Mean frequency: {rfm['frequency'].mean():.2f}")
    print(f"Maximum frequency: {rfm['frequency'].max()}")
    print(f"Mean monetary: INR {rfm['monetary'].mean():,.2f}")
    print(f"Maximum monetary: INR {rfm['monetary'].max():,.2f}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
