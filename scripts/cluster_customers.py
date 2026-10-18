
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "rfm.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "kmeans_evaluation.csv"


def main():
    # Load RFM features
    rfm = pd.read_csv(INPUT_FILE)
    features = ["recency", "frequency", "monetary"]
    X = rfm[features]

    # Scale the three features to comparable ranges
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    # Evaluate each required number of clusters
    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": score,
        })

    results_df = pd.DataFrame(results)
    results_df["inertia"] = results_df["inertia"].round(2)
    results_df["silhouette"] = results_df["silhouette"].round(3)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("K-MEANS EVALUATION")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
