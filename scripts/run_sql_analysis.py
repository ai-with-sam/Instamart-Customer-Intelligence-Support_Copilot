
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")

# Read the seven SQL statements from the file.
sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [
    statement.strip()
    for statement in sql_text.split(";")
    if any(
        line.strip() and not line.strip().startswith("--")
        for line in statement.splitlines()
    )
]

if len(queries) != 7:
    raise ValueError(
        f"Expected 7 SQL queries, but found {len(queries)}."
    )

# Run every query using pandas and SQLite.
with sqlite3.connect(DB_PATH) as connection:
    results = [
        pd.read_sql_query(query, connection)
        for query in queries
    ]

q1, q2, q3, q4, q5, q6, q7 = results

print("=" * 60)
print("DAY 3 — SQL ANALYSIS RESULTS")
print("=" * 60)

# Q1
print(f"\nQ1 — Qualifying high-spend customers: {len(q1)}")
assert len(q1) == 30, "Q1 reference count mismatch"

# Q2
q2_total = round(q2["revenue"].sum(), 2)
largest_cell = q2.loc[q2["revenue"].idxmax()]

print(f"\nQ2 — City/category combinations: {len(q2)}")
print(f"Total completed revenue: INR {q2_total:,.2f}")
print(
    "Largest revenue cell: "
    f"{largest_cell['city']} / {largest_cell['category']} "
    f"= INR {largest_cell['revenue']:,.2f}"
)

assert len(q2) == 30, "Q2 row count mismatch"
assert q2_total == 2998668.80, "Q2 revenue total mismatch"
assert largest_cell["city"] == "Bengaluru", "Q2 largest city mismatch"
assert largest_cell["category"] == "Electronics", (
    "Q2 largest category mismatch"
)
assert round(largest_cell["revenue"], 2) == 233909.45, (
    "Q2 largest revenue mismatch"
)

# Q3
print(f"\nQ3 — Products above average catalogue price: {len(q3)}")
assert len(q3) == 11, "Q3 product count mismatch"

# Q4
print(f"\nQ4 — Customers with no completed orders: {len(q4)}")
assert len(q4) == 25, "Q4 customer count mismatch"

# Q5
average_spend = float(q5["average_spend"].iloc[0]) if len(q5) else None

print(f"\nQ5 — Customers above average spend: {len(q5)}")
print(f"Average completed spend: INR {average_spend:,.2f}")

assert len(q5) == 550, "Q5 customer count mismatch"
assert round(average_spend, 2) == 2552.06, (
    "Q5 average spend mismatch"
)

# Q6
print(f"\nQ6 — Customers ranked: {len(q6)}")

top_customers = (
    q6.loc[q6["spend_rank"] == 1, [
        "city", "customer_id", "total_spend"
    ]]
    .sort_values("city")
)

print("\nTop-spending customer in each city:")
print(top_customers.to_string(index=False))

different_ranks = int((q6["spend_rank"] != q6["spend_row_number"]).sum())
print(f"Rows where RANK differs from ROW_NUMBER: {different_ranks}")

expected_top = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10),
}

actual_top = {
    row["city"]: (row["customer_id"], round(row["total_spend"], 2))
    for _, row in top_customers.iterrows()
}

assert actual_top == expected_top, "Q6 city leader mismatch"
assert different_ranks == 8, "Q6 ranking tie checkpoint mismatch"

# Q7
print(f"\nQ7 — Days with completed revenue: {len(q7)}")
last_cumulative = float(q7["cumulative_revenue"].iloc[-1])

print(f"Final cumulative revenue: INR {last_cumulative:,.2f}")

assert len(q7) == 183, "Q7 daily row count mismatch"
assert round(last_cumulative, 2) == 2998668.80, (
    "Q7 cumulative revenue mismatch"
)

print("\nALL 7 SQL QUERIES PASSED THEIR REFERENCE CHECKS")
