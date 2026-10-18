
from pathlib import Path
import sqlite3
import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
DB_PATH = Path("data/instamart.db")

CLEAN.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load raw CSV files without changing the originals
# --------------------------------------------------
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

# --------------------------------------------------
# 2. Clean customers
# --------------------------------------------------
customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Normalize gender spelling and capitalization
gender = customers["gender"].astype("string").str.strip().str.lower()

gender_map = {
    "male": "male",
    "m": "male",
    "man": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
}

customers["gender"] = gender.map(gender_map).fillna("unknown")

# Convert ages to numeric; invalid ages become missing
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

# Ages outside the allowed range are treated as missing
customers.loc[
    (customers["age"] < 10) | (customers["age"] > 100),
    "age"
] = pd.NA

# Fill missing ages using the median age for each city
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# Store age as an integer
customers["age"] = customers["age"].round().astype("int64")

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# --------------------------------------------------
# 3. Clean products
# --------------------------------------------------
products["product_id"] = products["product_id"].astype(str)
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="raise"
)

# --------------------------------------------------
# 4. Clean orders
# --------------------------------------------------
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

# Normalize status text
orders["status"] = (
    orders["status"].astype("string").str.strip().str.lower()
)

# For duplicate order IDs, keep the row with the latest timestamp
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset=["order_id"], keep="last").copy()

# Convert numeric fields
orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise")
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="raise"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Join product prices and categories
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

# Fill invalid order amounts using product price, quantity and discount
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

# Negative delivery times are invalid
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# For completed orders, fill missing delivery times using the
# median delivery time for that product category among completed orders
completed = orders["status"].eq("completed")
category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(category_medians)
)

# Remove helper columns from the final orders table
orders = orders.drop(columns=["price_inr", "category"])

# Store order timestamps in the required SQLite-friendly format
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# --------------------------------------------------
# 5. Save clean CSV files
# --------------------------------------------------
customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(CLEAN / "orders.csv", index=False)

# --------------------------------------------------
# 6. Create SQLite database
# --------------------------------------------------
if DB_PATH.exists():
    DB_PATH.unlink()

with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders.to_sql("orders", connection, if_exists="replace", index=False)

print("CLEANING COMPLETE")
print(f"Customers: {len(customers):,}")
print(f"Products: {len(products):,}")
print(f"Orders: {len(orders):,}")
print(f"Completed orders: {(orders['status'] == 'completed').sum():,}")
print(f"Cancelled orders: {(orders['status'] == 'cancelled').sum():,}")
print(f"Revenue (completed): INR {orders.loc[orders['status'] == 'completed', 'order_amount'].sum():,.2f}")
print(f"Clean CSV folder: {CLEAN}")
print(f"SQLite database: {DB_PATH}")
