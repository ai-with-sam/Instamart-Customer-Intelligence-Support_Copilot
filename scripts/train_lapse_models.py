
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.dummy import DummyClassifier


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "outputs" / "lapse_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET = "lapsed"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "favorite_category",
    "city",
    "gender",
]


def evaluate_model(name, model, X_test, y_test):
    """Print metrics for the positive class: lapsed = 1."""
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    return f1_score(y_test, predictions, zero_division=0)


def main():
    # --------------------------------------------------
    # 2. Load the dataset and create the same split
    # --------------------------------------------------
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[ID_COLUMN, TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("LAPSE MODEL TRAINING")
    print("=" * 50)
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows:     {len(X_test)}")
    print(f"Training lapsed customers: {(y_train == 1).sum()}")
    print(f"Training active customers: {(y_train == 0).sum()}")

    # --------------------------------------------------
    # 3. Baseline: always predict not lapsed
    # --------------------------------------------------
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    evaluate_model(
        "BASELINE: MOST FREQUENT CLASS",
        baseline,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # 4. Preprocessing
    # Fit scaler and encoder on training data only.
    # --------------------------------------------------
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # --------------------------------------------------
    # 5. Tune Logistic Regression
    # --------------------------------------------------
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
    }

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    print("\nTuning Logistic Regression (5-fold cross-validation)...")
    logistic_search.fit(X_train, y_train)

    print(f"Best Logistic Regression settings: {logistic_search.best_params_}")
    print(f"Best cross-validation F1: {logistic_search.best_score_:.3f}")

    logistic_f1 = evaluate_model(
        "TUNED LOGISTIC REGRESSION — TEST SET",
        logistic_search.best_estimator_,
        X_test,
        y_test,
    )

    # Show the frequency mean learned from training data.
    fitted_preprocessor = (
        logistic_search.best_estimator_.named_steps["preprocessor"]
    )
    fitted_scaler = (
        fitted_preprocessor
        .named_transformers_["num"]
        .named_steps["scaler"]
    )
    frequency_index = NUMERIC_FEATURES.index("frequency")
    learned_frequency_mean = fitted_scaler.mean_[frequency_index]
    full_cohort_frequency_mean = df["frequency"].mean()

    print("\nFREQUENCY SCALING CHECK")
    print(f"Full-cohort mean frequency: {full_cohort_frequency_mean:.3f}")
    print(f"Training-only scaler mean:  {learned_frequency_mean:.3f}")
    print(
        "These may differ because the scaler is fitted only on the training split."
    )

    # --------------------------------------------------
    # 6. Tune Random Forest
    # --------------------------------------------------
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    rf_grid = {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    }

    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    print("\nTuning Random Forest (45 fits across 5 folds)...")
    rf_search.fit(X_train, y_train)

    print(f"Best Random Forest settings: {rf_search.best_params_}")
    print(f"Best cross-validation F1: {rf_search.best_score_:.3f}")

    rf_f1 = evaluate_model(
        "TUNED RANDOM FOREST — TEST SET",
        rf_search.best_estimator_,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # 7. Compare models
    # --------------------------------------------------
    print("\nMODEL COMPARISON")
    print("=" * 50)
    print(f"Logistic Regression test F1: {logistic_f1:.3f}")
    print(f"Random Forest test F1:      {rf_f1:.3f}")

    if logistic_f1 >= rf_f1:
        print("Higher test F1: Logistic Regression")
    else:
        print("Higher test F1: Random Forest")

    print(
        "\nNote: test-set results are for comparison. "
        "Use cross-validation scores for tuning decisions."
    )
    
    # --------------------------------------------------
    # 8. Oversample lapsed customers in training only
    # --------------------------------------------------
    print("\nOVERSAMPLING TRAINING DATA")
    print("=" * 50)

    positive_rows = X_train.loc[y_train == 1]
    needed = int((y_train == 0).sum() - (y_train == 1).sum())

    sampled_lapsed = positive_rows.sample(
        n=needed,
        replace=True,
        random_state=42,
    )

    X_balanced = pd.concat([X_train, sampled_lapsed], axis=0)
    y_balanced = pd.concat([
        y_train,
        pd.Series(1, index=sampled_lapsed.index),
    ])

    print("Balanced training counts:")
    print(y_balanced.value_counts().sort_index().to_string())

    balanced_logistic = clone(logistic_search.best_estimator_)
    balanced_logistic.fit(X_balanced, y_balanced)

    balanced_f1 = evaluate_model(
        "OVERSAMPLED LOGISTIC REGRESSION — TEST SET",
        balanced_logistic,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # 9. Random Forest feature importance
    # --------------------------------------------------
    print("\nTOP 5 RANDOM FOREST FEATURES")
    print("=" * 50)

    rf_pipeline_best = rf_search.best_estimator_
    rf_preprocessor = rf_pipeline_best.named_steps["preprocessor"]
    rf_model = rf_pipeline_best.named_steps["model"]

    rf_feature_names = rf_preprocessor.get_feature_names_out()
    importance = pd.DataFrame({
        "feature": rf_feature_names,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(importance.head(5).to_string(index=False))

    importance.to_csv(
        OUTPUT_DIR / "lapse_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------
    # 10. Logistic Regression coefficient interpretation
    # --------------------------------------------------
    print("\nLOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 50)

    log_preprocessor = (
        logistic_search.best_estimator_.named_steps["preprocessor"]
    )
    log_model = logistic_search.best_estimator_.named_steps["model"]

    log_feature_names = log_preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame({
        "feature": log_feature_names,
        "coefficient": log_model.coef_[0],
    })

    print("\nThree strongest positive coefficients:")
    print(
        coefficients.nlargest(3, "coefficient").to_string(index=False)
    )

    print("\nThree strongest negative coefficients:")
    print(
        coefficients.nsmallest(3, "coefficient").to_string(index=False)
    )

    coefficients.to_csv(
        OUTPUT_DIR / "lapse_logistic_coefficients.csv",
        index=False,
    )

    # --------------------------------------------------
    # 11. Select a final model using cross-validation
    # --------------------------------------------------
    # Prefer cross-validation results for model selection,
    # rather than choosing a model solely on test-set results.
    
    if rf_search.best_score_ >= logistic_search.best_score_:
        selected_model = rf_search.best_estimator_
        selected_name = "Random Forest"
    else:
        selected_model = logistic_search.best_estimator_
        selected_name = "Logistic Regression"

    print(f"\nSelected final model: {selected_name}")
    print(
        "Selection based on cross-validation F1, not the test-set F1."
    )

    # --------------------------------------------------
    # 12. Save, reload, and verify predictions
    # --------------------------------------------------
    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)

    reloaded_model = joblib.load(model_path)

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)

    assert (original_predictions == reloaded_predictions).all()

    print(f"\nSaved model: {model_path}")
    print("Reload check passed: predictions match exactly.")

    # --------------------------------------------------
    # 13. Premium-customer retention list
    # --------------------------------------------------
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    segments = pd.read_csv(segments_path)

    premium_ids = set(
        segments.loc[
            segments["is_premium"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            ),
            "customer_id",
        ]
    )

    # X_test retains the original dataset row indices.
    test_customer_ids = df.loc[X_test.index, ID_COLUMN]
    premium_mask = test_customer_ids.isin(premium_ids)
    premium_test_X = X_test.loc[premium_mask]

    if premium_test_X.empty:
        raise ValueError(
            "No Premium customers were found in the test set."
        )

    lapse_probabilities = reloaded_model.predict_proba(
        premium_test_X
    )[:, list(reloaded_model.classes_).index(1)]

    retention = pd.DataFrame({
        "customer_id": df.loc[premium_test_X.index, ID_COLUMN],
        "lapse_probability": lapse_probabilities,
        "actually_lapsed": y_test.loc[premium_test_X.index].values,
    })

    retention = retention.sort_values(
        "lapse_probability",
        ascending=False,
    ).head(10)

    retention_path = OUTPUT_DIR / "retention_list.csv"
    retention.to_csv(retention_path, index=False)

    print("\nTOP 10 PREMIUM CUSTOMERS FOR RETENTION")
    print(retention.to_string(index=False))
    print(
        "\nActually lapsed among these 10: "
        f"{int(retention['actually_lapsed'].sum())}"
    )
    print(f"Saved retention list: {retention_path}")

    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
