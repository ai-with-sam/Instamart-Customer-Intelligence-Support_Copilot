
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "rfm.csv"
SEGMENTS_FILE = PROJECT_ROOT / "outputs" / "customer_segments.csv"
PROFILE_FILE = PROJECT_ROOT / "outputs" / "segment_profiles.csv"
PCA_FILE = PROJECT_ROOT / "outputs" / "pca_summary.csv"

FEATURES = ["recency", "frequency", "monetary"]
FINAL_K = 4



def assign_segment_name(profile, premium_cluster):
    """Give each cluster a descriptive name based on its RFM profile."""
    cluster_id = profile.name

    if cluster_id == premium_cluster:
        return "Premium"

    # Smaller recency means the customer ordered more recently.
    if (
        profile["mean_frequency"] >= 15
        and profile["mean_recency"] <= 20
    ):
        return "Loyal Active"

    if profile["mean_recency"] >= 60:
        return "At Risk"

    return "Regular"



def main():
    rfm = pd.read_csv(INPUT_FILE)

    # Standardize RFM features before clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[FEATURES])

    # Fit the selected final model.
    model = KMeans(
        n_clusters=FINAL_K,
        n_init=10,
        random_state=42
    )
    rfm["segment_id"] = model.fit_predict(X_scaled)

    # Profile clusters using the original, interpretable RFM values.
    profiles = (
        rfm.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "nunique"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
        )
        .sort_values("mean_monetary", ascending=False)
    )

    # The cluster with the highest mean monetary value is Premium.
    premium_cluster = profiles["mean_monetary"].idxmax()

    # Add business-friendly segment names.
    profiles["segment_name"] = profiles.apply(
        lambda row: assign_segment_name(row, premium_cluster),
        axis=1
    )

    # If descriptive rules produce duplicate names, keep names unique.
    used_names = set()
    for cluster_id in profiles.index:
        name = profiles.loc[cluster_id, "segment_name"]
        if name in used_names:
            profiles.loc[cluster_id, "segment_name"] = (
                f"{name} {cluster_id}"
            )
        used_names.add(profiles.loc[cluster_id, "segment_name"])

    name_mapping = profiles["segment_name"].to_dict()
    rfm["segment_name"] = rfm["segment_id"].map(name_mapping)
    rfm["is_premium"] = (
        rfm["segment_id"] == premium_cluster
    ).astype(int)

    # Save the required customer-level output.
    required_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
    rfm[required_columns].to_csv(SEGMENTS_FILE, index=False)

    # Save a readable cluster profile.
    profiles.reset_index().to_csv(PROFILE_FILE, index=False)

    # PCA on the same standardized RFM features.
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    pca_summary = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    pca_summary.loc[len(pca_summary)] = [
        "Total",
        pca.explained_variance_ratio_.sum()
    ]
    pca_summary.to_csv(PCA_FILE, index=False)

    print("CUSTOMER SEGMENTATION COMPLETE")
    print(f"Customers segmented: {len(rfm):,}")
    print(f"Selected K: {FINAL_K}")
    print(f"Premium cluster ID: {premium_cluster}")
    print("\nSEGMENT PROFILES")
    print(
        profiles.reset_index().round(2).to_string(index=False)
    )
    print("\nPCA EXPLAINED VARIANCE")
    print(pca_summary.round(3).to_string(index=False))
    print(f"\nSaved customer segments: {SEGMENTS_FILE}")
    print(f"Saved segment profiles: {PROFILE_FILE}")
    print(f"Saved PCA summary: {PCA_FILE}")


if __name__ == "__main__":
    main()
