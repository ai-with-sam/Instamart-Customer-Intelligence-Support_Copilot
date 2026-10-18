from pathlib import Path
import pandas as pd

RAW = Path("data/raw")

customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False, dtype=str)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False, dtype=str)
orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False, dtype=str)

def show_audit(name, df):
    print(f"\n{'=' * 60}")
    print(f"{name.upper()} — SHAPE: {df.shape}")
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values by column:")
    for col in df.columns:
        values = df[col].astype(str).str.strip()
        missing = values.eq("").sum()
        percentage = missing / len(df) * 100 if len(df) else 0
        print(f"{col}: {missing} ({percentage:.2f}%)")

show_audit("customers", customers)
show_audit("products", products)
show_audit("orders", orders)

# Convert columns temporarily for auditing; the raw files remain unchanged.
amount = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(customers["age"], errors="coerce")

print(f"\n{'=' * 60}")
print("REFERENCE AUDIT CHECKS")
checks = {
    "Raw order rows": len(orders),
    "Exact duplicate order rows": int(orders.duplicated().sum()),
    "Pending order rows": int(orders["status"].str.strip().str.lower().eq("pending").sum()),
    "Distinct order IDs": int(orders["order_id"].nunique()),
    "Non-numeric order amounts": int(amount.isna().sum()),
    "Negative delivery times": int(delivery.lt(0).sum()),
    "Blank delivery times": int(orders["delivery_minutes"].str.strip().eq("").sum()),
    "Blank customer ages": int(customers["age"].str.strip().eq("").sum()),
    "Ages outside 10–100": int(((age < 10) | (age > 100)).sum()),
    "Distinct raw city spellings": int(customers["city"].nunique()),
    "Distinct nonblank raw gender spellings": int(
        customers.loc[customers["gender"].str.strip().ne(""), "gender"].nunique()
    ),
    "Blank genders": int(customers["gender"].str.strip().eq("").sum()),
}

for label, value in checks.items():
    print(f"{label}: {value}")