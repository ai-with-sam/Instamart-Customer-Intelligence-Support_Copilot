
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORDERS_FILE = PROJECT_ROOT / "data" / "clean" / "orders.csv"
CUSTOMERS_FILE = PROJECT_ROOT / "data" / "clean" / "customers.csv"
PRODUCTS_FILE = PROJECT_ROOT / "data" / "clean" / "products.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "lapse_dataset.csv"

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")


def main():
    # Load cleaned data.
    orders = pd.read_csv(ORDERS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Define completed orders and the historical feature window.
    completed = orders[orders["status"].eq("completed")].copy()

    history = orders[
        orders["order_ts"].lt(CUTOFF)
        & orders["status"].isin(["completed", "cancelled"])
    ].copy()

    completed_history = history[
        history["status"].eq("completed")
    ].copy()

    # Eligible cohort: at least one completed order before cutoff.
    cohort_ids = completed_history["customer_id"].unique()

    # Target labels use September activity only.
    september_completed = completed[
        completed["order_ts"].ge(CUTOFF)
        & completed["order_ts"].lt(LABEL_END)
    ]

    active_in_september = set(
        september_completed["customer_id"].unique()
    )

    cohort = customers[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(active_in_september)
    ).astype(int)

    # Build features using only pre-cutoff completed orders.
    grouped = completed_history.groupby("customer_id")

    last_order = grouped["order_ts"].max()
    features = pd.DataFrame(index=last_order.index)

    features["recency"] = (
        (CUTOFF - last_order).dt.days
    )
    features["frequency"] = grouped["order_id"].nunique()
    features["monetary"] = grouped["order_amount"].sum()
    features["avg_discount_pct"] = grouped["discount_pct"].mean()
    features["avg_delivery_minutes"] = (
        grouped["delivery_minutes"].mean()
    )

    # Cancellation rate: cancelled / all final-status orders
    # strictly before the cutoff.
    history_counts = history.groupby("customer_id").size()
    cancelled_counts = (
        history[history["status"].eq("cancelled")]
        .groupby("customer_id")
        .size()
    )

    features["cancellation_rate"] = (
        cancelled_counts.reindex(features.index, fill_value=0)
        / history_counts.reindex(features.index)
    )

    # Most frequently ordered product category, using history only.
    if "category" not in products.columns:
        raise ValueError(
            "products.csv must contain a 'category' column. "
            f"Found columns: {products.columns.tolist()}"
        )

    category_orders = completed_history.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    category_orders = category_orders.dropna(subset=["category"])

    favorite_category = (
        category_orders.groupby("customer_id")["category"]
        .agg(lambda values: values.mode().iloc[0])
    )
    features["favorite_category"] = favorite_category

    # Add clean customer attributes and tenure.
    cohort = cohort.merge(
        features.reset_index(),
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    cohort["tenure_days"] = (
        CUTOFF - cohort["signup_date"]
    ).dt.days

    # Fill missing feature values without using the target.
    for column in [
        "avg_discount_pct",
        "avg_delivery_minutes",
        "cancellation_rate",
        "recency",
        "frequency",
        "monetary",
        "tenure_days",
        "age",
    ]:
        cohort[column] = pd.to_numeric(
            cohort[column], errors="coerce"
        )
        if cohort[column].isna().any():
            median_value = cohort[column].median()
            cohort[column] = cohort[column].fillna(
                0 if pd.isna(median_value) else median_value
            )

    for column in ["city", "gender", "favorite_category"]:
        cohort[column] = (
            cohort[column].fillna("Unknown").astype(str)
        )

    feature_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "favorite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
        "lapsed",
    ]

    dataset = cohort[feature_columns].copy()

    # Validate that no feature values are missing.
    model_features = [
        column for column in feature_columns
        if column not in ["customer_id", "lapsed"]
    ]

    missing_counts = dataset[model_features].isna().sum()
    if missing_counts.any():
        raise ValueError(
            "Missing feature values remain:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    # Verify that no post-cutoff order data entered feature creation.
    assert history["order_ts"].lt(CUTOFF).all()
    assert completed_history["order_ts"].lt(CUTOFF).all()

    # Reproducible stratified split for the next modeling step.
    X = dataset[model_features]
    y = dataset["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print("LAPSE DATASET CREATED")
    print(f"Cohort customers: {len(dataset):,}")
    print(f"Lapsed customers: {int(y.sum()):,}")
    print(f"Not lapsed: {int((y == 0).sum()):,}")
    print(f"Lapsed rate: {y.mean():.2%}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print("\nTraining label counts:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nTest label counts:")
    print(y_test.value_counts().sort_index().to_string())
    print("\nMissing values in model features:")
    print(int(dataset[model_features].isna().sum().sum()))
    print(f"\nSaved dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
