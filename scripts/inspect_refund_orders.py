
import sqlite3

DB_PATH = "data/instamart.db"

queries = {
    "AUTOMATIC REFUND CANDIDATES": """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE status = 'completed'
          AND delivery_minutes > 45
          AND order_amount <= 1000
        LIMIT 3
    """,
    "HUMAN ESCALATION CANDIDATES": """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE status = 'completed'
          AND delivery_minutes > 45
          AND order_amount > 1000
        LIMIT 3
    """,
    "NO REFUND CANDIDATES": """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE status = 'completed'
          AND delivery_minutes <= 45
        LIMIT 3
    """,
    "NON-COMPLETED ORDERS": """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE status != 'completed'
        LIMIT 3
    """,
}

with sqlite3.connect(DB_PATH) as conn:
    for label, sql in queries.items():
        print(f"\n{label}")
        rows = conn.execute(sql).fetchall()

        if rows:
            for row in rows:
                print(row)
        else:
            print("No matching orders found.")
