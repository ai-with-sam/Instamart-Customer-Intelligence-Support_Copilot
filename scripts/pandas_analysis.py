
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/instamart.db")

with sqlite3.connect(DB_PATH) as connection:
    orders = pd.read_sql_query("SELECT * FROM orders", connection)
    customers = pd.read_sql_query("SELECT * FROM customers", connection)
    products = pd.read_sql_query("SELECT * FROM products", connection)

# Ensure the fields used in calculations are numeric.
for column in [
    "order_amount", "discount_pct", "quantity", "delivery_minutes"
]:
    orders[column] = pd.to_numeric(orders[column], errors="coerce")

completed = orders.loc[orders["status"] == "completed"].copy()

# --------------------------------------------------
# A. Rebuild SQL Q2 with a pandas pivot table
# --------------------------------------------------
completed_details = (
    completed
    .merge(customers[["customer_id", "city"]], on="customer_id")
    .merge(products[["product_id", "category"]], on="product_id")
)

pivot = pd.pivot_table(
    completed_details,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All",
)

print("=" * 60)
print("PANDAS CROSS-CHECK: CITY AND CATEGORY REVENUE")
print("=" * 60)
print(pivot.round(2).to_string())

grand_total = round(float(pivot.loc["All", "All"]), 2)
print(f"\nPandas grand total: INR {grand_total:,.2f}")

assert grand_total == 2998668.80, "Pandas total differs from SQL reference"

# --------------------------------------------------
# B. Completed-order revenue by city
# --------------------------------------------------
city_summary = (
    completed_details
    .groupby("city")
    .agg(
        revenue_sum=("order_amount", "sum"),
        revenue_mean=("order_amount", "mean"),
        order_count=("order_amount", "count"),
    )
    .sort_index()
)

city_summary["revenue_sum"] = city_summary["revenue_sum"].round(2)
city_summary["revenue_mean"] = city_summary["revenue_mean"].round(2)

print("\n" + "=" * 60)
print("COMPLETED-ORDER SUMMARY BY CITY")
print("=" * 60)
print(city_summary.to_string())

assert city_summary.loc["Bengaluru", "revenue_sum"] == 523643.40
assert city_summary.loc["Bengaluru", "revenue_mean"] == 276.47
assert city_summary.loc["Bengaluru", "order_count"] == 1894

assert city_summary.loc["Mumbai", "revenue_sum"] == 479058.50
assert city_summary.loc["Mumbai", "revenue_mean"] == 297.00
assert city_summary.loc["Mumbai", "order_count"] == 1613

# --------------------------------------------------
# C. Descriptive statistics and outlier fences
# --------------------------------------------------
amount = completed["order_amount"].dropna()

q1 = amount.quantile(0.25)
median = amount.quantile(0.50)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

lower_outliers = int((amount < lower_fence).sum())
upper_outliers = int((amount > upper_fence).sum())

print("\n" + "=" * 60)
print("COMPLETED ORDER AMOUNT STATISTICS")
print("=" * 60)
print(f"Count: {amount.count():,}")
print(f"Mean: INR {amount.mean():.2f}")
print(f"Median: INR {median:.2f}")
print(f"Skewness: {amount.skew():.4f}")
print(f"Minimum: INR {amount.min():.2f}")
print(f"25th percentile (Q1): INR {q1:.2f}")
print(f"75th percentile (Q3): INR {q3:.2f}")
print(f"Maximum: INR {amount.max():.2f}")
print(f"IQR: INR {iqr:.2f}")
print(f"Lower outlier fence: INR {lower_fence:.2f}")
print(f"Upper outlier fence: INR {upper_fence:.2f}")
print(f"Outliers below lower fence: {lower_outliers}")
print(f"Outliers above upper fence: {upper_outliers}")

# --------------------------------------------------
# D. Correlation matrix for completed orders
# --------------------------------------------------
corr_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes",
]

correlation = completed[corr_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
print(correlation.round(4).to_string())

# Find the strongest absolute correlation between distinct variables.
pairs = []
for i, first in enumerate(corr_columns):
    for second in corr_columns[i + 1:]:
        value = correlation.loc[first, second]
        pairs.append((first, second, value, abs(value)))

strongest = max(pairs, key=lambda item: item[3])

print(
    "\nLargest absolute correlation: "
    f"{strongest[0]} vs {strongest[1]} = {strongest[2]:.4f}"
)

print("\nPANDAS ANALYSIS COMPLETE")
