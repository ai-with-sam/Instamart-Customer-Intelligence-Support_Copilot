
from pathlib import Path
import sqlite3
import pandas as pd

CLEAN = Path("data/clean")
DB_PATH = Path("data/instamart.db")

# Load cleaned CSV files
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Convert columns to numeric for reliable checks
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

# 1–4: File and row-count checks
assert len(customers) == 1200, "Customer count mismatch"
assert len(products) == 40, "Product count mismatch"
assert len(orders) == 12309, "Order count mismatch"
assert orders["order_id"].nunique() == len(orders), "Duplicate order IDs remain"

# 5–7: Order status checks
assert completed.sum() == 11110, "Completed order count mismatch"
assert cancelled.sum() == 1199, "Cancelled order count mismatch"
assert not orders["status"].eq("pending").any(), "Pending orders remain"

# 8–10: Order amount and delivery checks
assert orders["order_amount"].notna().all(), "Missing order amounts remain"
assert orders.loc[completed, "delivery_minutes"].notna().all(), (
    "Completed orders have missing delivery times"
)
assert orders.loc[cancelled, "delivery_minutes"].isna().all(), (
    "Cancelled orders should have missing delivery times"
)

# 11–13: Customer data checks
assert customers["age"].notna().all(), "Customer ages are missing"
assert customers["age"].between(18, 60).all(), "Customer ages outside 18–60"
assert set(customers["gender"].unique()).issubset(
    {"male", "female", "unknown"}
), "Unexpected gender values"

# 14: Revenue check
revenue = round(orders.loc[completed, "order_amount"].sum(), 2)
assert revenue == 2998668.80, f"Revenue mismatch: {revenue}"

# 15: SQLite database checks
assert DB_PATH.exists(), "SQLite database file is missing"

with sqlite3.connect(DB_PATH) as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    assert {"customers", "products", "orders"}.issubset(tables), (
        "One or more database tables are missing"
    )

    db_order_count = connection.execute(
        "SELECT COUNT(*) FROM orders"
    ).fetchone()[0]

    assert db_order_count == 12309, "Database order count mismatch"

print("ALL CHECKPOINTS PASSED")
