import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle
import os

# loading the dataset 
df = pd.read_csv("cleaned_online_retail.csv")
print("Dataset loaded successfully!")
print(df.shape)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])


# Feature Engineering - Calculating RFM
print("\n--- Feature Engineering to calculate RFM ---")

# adding new column TotalPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)     # latest date of purchase + 1 day 
print(f"Snapshot date: {snapshot_date}")

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,   # Recency : snapshot_date - most recent purchase date 
    "InvoiceNo": "nunique",                                    # Frequence : counts the unique id's showing the number of time a customer comes 
    "TotalPrice": "sum"                                        # Monetary : sum of all products purchased by a customer 
})

rfm.columns = ["Recency", "Frequency", "Monetary"]      # renaming the columns 

print("\nRFM DataFrame (first 10 rows):")
print(rfm.head(10))
print(f"RFM Shape: {rfm.shape}")

# seeing new column individually 
for col in ["Recency", "Frequency", "Monetary"]:
    print(f"\n--- {col} ---")
    print(rfm[col].describe())

# Handling Outliers (Winsorization - IQR Capping)
print("\n--- Handling Outliers (IQR Capping) ---")

rfm_clean = rfm.copy()

for col in ["Recency", "Frequency", "Monetary"]:
    lower = rfm_clean[col].quantile(0.05)
    upper = rfm_clean[col].quantile(0.95)
    rfm_clean[col] = rfm_clean[col].clip(lower=lower, upper=upper)
    print(f"{col}: clipped to [{lower:.2f}, {upper:.2f}]")             # limiting values to a specified range by replacing them

print(f"RFM shape after capping: {rfm_clean.shape}")

# Standardize/Normalize RFM values - to calculate K-Means
print("\n--- Standardization ---")

# Method 1: Standard Scaler - finding mean and standard deviation of each column
scaler_standard = StandardScaler()
rfm_standardized = scaler_standard.fit_transform(rfm_clean) 

#Method 2: MinMaxScaler (0-1 normalization) - scaling each column independently 
scaler_minmax = MinMaxScaler()
rfm_normalized = scaler_minmax.fit_transform(rfm_clean)

print("Standardized RFM (first 5 rows):")
print(rfm_standardized[:5])
print("\nNormalized RFM (first 5 rows):")
print(rfm_normalized[:5])


# lbow Method + Silhouette Score → Choose Best K

print("\n--- Elbow Method and Silhouette Score ---")

K_range = range(2, 11)
inertia           = []    # squared distances of all points from their assigned centroid
silhouette_scores = []    # to check if clusters are separated and customers are grouped with similar customers.


for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_standardized)
    inertia.append(km.inertia_)
    score = silhouette_score(rfm_standardized, km.labels_)
    silhouette_scores.append(score)
    print(f"K={k} | Inertia={km.inertia_:.2f} | Silhouette={score:.4f}")

# Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker="o", color="blue")
plt.title("Elbow Method for Optimal K", fontsize=14)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("10_elbow_curve.png")
plt.show()

# Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette_scores, marker="o", color="green")
plt.title("Silhouette Score for Different K", fontsize=14)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("11_silhouette_scores.png")
plt.show()

# Force minimum 4 clusters for meaningful business segments
min_clusters = 4
valid_k      = [k for k in K_range if k >= min_clusters]
valid_scores = [silhouette_scores[i] for i, k in enumerate(K_range) if k >= min_clusters]

best_k          = valid_k[np.argmax(valid_scores)]
best_silhouette = max(valid_scores)

print(f"\nBest K (min={min_clusters}): {best_k}")
print(f"Best Silhouette Score     : {best_silhouette:.4f}")


# STEP 5: Run Clustering

# --- KMeans ---
print("\n--- KMeans Clustering ---")
kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm_clean["Cluster"] = kmeans_best.fit_predict(rfm_standardized)
print(f"Cluster distribution (K={best_k}):")
print(rfm_clean["Cluster"].value_counts().sort_index())

# --- DBSCAN ---
print("\n--- DBSCAN Clustering ---")
dbscan = DBSCAN(eps=0.5, min_samples=5)
rfm_clean["DBSCAN_Cluster"] = dbscan.fit_predict(rfm_standardized)
print("DBSCAN Cluster distribution:")
print(rfm_clean["DBSCAN_Cluster"].value_counts())

# --- Hierarchical ---
print("\n--- Hierarchical Clustering ---")
hierarchical = AgglomerativeClustering(n_clusters=best_k)
rfm_clean["Hierarchical_Cluster"] = hierarchical.fit_predict(rfm_standardized)
print("Hierarchical Cluster distribution:")
print(rfm_clean["Hierarchical_Cluster"].value_counts())


# Label Clusters — RFM SCORE BASED APPROACH

#   Recency  rank: LOWER days = BETTER (rank high = most recent)
#   Frequency rank: HIGHER count = BETTER
#   Monetary  rank: HIGHER spend = BETTER


print("\n--- Cluster Profiles (RFM Averages per Cluster) ---")

cluster_profiles = (rfm_clean.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2))
print(cluster_profiles)

def label_clusters_by_rank(profiles):
    """
    Label each cluster by ranking clusters on each RFM dimension.

    Ranking logic (1=worst, N=best):
      Recency  : rank ascending=False → highest days(oldest)=rank1(worst)
                                        lowest days(recent) =rankN(best)
      Frequency: rank ascending=True  → lowest freq =rank1(worst)
                                        highest freq =rankN(best)
      Monetary : rank ascending=True  → lowest money=rank1(worst)
                                        highest money=rankN(best)
    """
    n = len(profiles)  # number of clusters

    # Rank clusters against each other (1=worst, N=best)
    r_rank = profiles["Recency"].rank(ascending=False)  # low days=high rank=best
    f_rank = profiles["Frequency"].rank(ascending=True) # high freq=high rank=best
    m_rank = profiles["Monetary"].rank(ascending=True)  # high money=high rank=best

    print(f"\n  Total clusters (n): {n}")
    print(f"  Ranks go from 1 (worst) to {n} (best)\n")

    labels = {}

    for cluster_id in profiles.index:
        r = r_rank[cluster_id]
        f = f_rank[cluster_id]
        m = m_rank[cluster_id]
        # clearly printing the values 
        print(f"  Cluster {cluster_id}: "
              f"R_rank={r:.0f}  F_rank={f:.0f}  M_rank={m:.0f} "
              f"| Recency={profiles.loc[cluster_id,'Recency']:.1f}  "
              f"Frequency={profiles.loc[cluster_id,'Frequency']:.1f}  "
              f"Monetary={profiles.loc[cluster_id,'Monetary']:.1f}")

        # Conditions for the clusters  
        if r == n and f == n and m == n:
            labels[cluster_id] = "High-Value"

        elif r == 1 and f == 1 and m == 1:
            labels[cluster_id] = "At-Risk"

        elif f <= (n // 2) and m <= (n // 2):
            labels[cluster_id] = "Occasional"

        else:
            labels[cluster_id] = "Regular"

    return labels


print("\n--- Ranking Clusters to Assign Segment Labels ---")
segment_map = label_clusters_by_rank(cluster_profiles)

print(f"\nSegment Map (Cluster → Label): {segment_map}")

# Add Segment column to cluster_profiles for display
cluster_profiles["Segment"] = cluster_profiles.index.map(segment_map)
print("\n--- Cluster Profiles with Segment Labels ---")
print(cluster_profiles)

# Assign segment to every customer
rfm_clean["Segment"] = rfm_clean["Cluster"].map(segment_map)

print("\n--- Customer Segment Distribution ---")
print(rfm_clean["Segment"].value_counts())          # counts the number of customers in each segment 


# CHECKING THE OUTPUT — print average RFM per SEGMENT
# (confirms labels match the expected RFM patterns)

print("\n--- Sanity Check: Average RFM per Segment ---")
sanity = rfm_clean.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2)
sanity["Count"] = rfm_clean.groupby("Segment")["Cluster"].count()
print(sanity)
print("""
Expected pattern:
  High-Value : LOW Recency, HIGH Frequency, HIGH Monetary
  Regular    : MID Recency, MID Frequency,  MID Monetary
  Occasional : ANY Recency, LOW Frequency,  LOW Monetary
  At-Risk    : HIGH Recency,LOW Frequency,  LOW Monetary
""")

# FINAL RFM TABLE
print("\n--- Final RFM with Clusters and Segments (first 10) ---")
print(rfm_clean[["Recency", "Frequency", "Monetary",
                  "Cluster", "Segment"]].head(10))



# Visualize Clusters
segment_palette = {"High-Value": "red", "Regular":    "blue", "At-Risk":    "orange", "Occasional": "green"}

# Scatter 1: Recency vs Monetary
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Recency", y="Monetary",
    hue="Segment",
    data=rfm_clean,
    palette=segment_palette,
    alpha=0.6
)
plt.title("Customer Clusters: Recency vs Monetary", fontsize=14)
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Monetary (total spending £)")
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig("12_cluster_scatter_recency_monetary.png")
plt.show()

# Scatter 2: Frequency vs Monetary
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Frequency", y="Monetary",
    hue="Segment",
    data=rfm_clean,
    palette=segment_palette,
    alpha=0.6
)
plt.title("Customer Clusters: Frequency vs Monetary", fontsize=14)
plt.xlabel("Frequency (number of transactions)")
plt.ylabel("Monetary (total spending £)")
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig("13_cluster_scatter_frequency_monetary.png")
plt.show()

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection="3d")

for segment, color in segment_palette.items():
    if segment in rfm_clean["Segment"].unique():
        subset = rfm_clean[rfm_clean["Segment"] == segment]
        ax.scatter(
            subset["Recency"],
            subset["Frequency"],
            subset["Monetary"],
            label=segment,
            color=color,
            alpha=0.6,
            s=20
        )

ax.set_xlabel("Recency (days)")
ax.set_ylabel("Frequency (orders)")
ax.set_zlabel("Monetary (£)")
ax.set_title("3D Customer Clusters (RFM)", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("14_cluster_3d_plot.png")
plt.show()


# Cluster Summary Statistics
print("\n--- Cluster Summary Statistics ---")

summary = (
    rfm_clean
    .groupby("Segment")
    .agg(
        Recency_Avg  =("Recency",   "mean"),
        Frequency_Avg=("Frequency", "mean"),
        Monetary_Avg =("Monetary",  "mean"),
        Count        =("Cluster",   "count")
    )
    .round(2)
    .sort_values("Monetary_Avg", ascending=False)
)
print(summary)


# Save the Best Model
print("\n--- Saving Best Model ---")

model_data = {
    "kmeans_model":     kmeans_best,
    "scaler":           scaler_standard,
    "best_k":           best_k,
    "segment_labels":   segment_map,
    "cluster_profiles": cluster_profiles
}

with open("best_kmeans_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("Saved: best_kmeans_model.pkl")

rfm_clean.to_csv("rfm_with_clusters.csv")
print("Saved: rfm_with_clusters.csv")

cluster_profiles.to_csv("cluster_profiles.csv")
print("Saved: cluster_profiles.csv")


# Prediction Function for Streamlit
def predict_customer_segment(recency, frequency, monetary):
    """
    Predict segment for a new customer.

    Parameters
    ----------
    recency   : int   - days since last purchase (lower = better)
    frequency : int   - number of unique invoices
    monetary  : float - total amount spent (£)

    Returns
    -------
    cluster   : int - KMeans cluster number
    segment   : str - business segment label
    """
    new_customer       = np.array([[recency, frequency, monetary]])
    new_customer_scaled = scaler_standard.transform(new_customer)
    cluster            = kmeans_best.predict(new_customer_scaled)[0]
    segment            = segment_map.get(cluster, "Regular")
    return cluster, segment


print("\n--- Example Predictions ---")
print(f"{'R':>5} {'F':>5} {'M':>8} | {'Cluster':>7} | Segment")
print("-" * 45)

test_cases = [
    (5,   20, 8000),   # Very recent, very frequent, high spend → High-Value
    (250,  1,  100),   # Very old, once,  tiny spend → At-Risk
    (180,  1,  200),   # Old, once, small spend      → Occasional / At-Risk
    (60,   4, 1200),   # Medium everything            → Regular
    (15,  10, 3000),   # Recent, moderate freq        → Regular / High-Value
]

for r, f, m in test_cases:
    c, seg = predict_customer_segment(r, f, m)
    print(f"R={r:>4} F={f:>3} M={m:>7} | Cluster {c} | {seg}")


# FINAL SUMMARY
print("\n" + "="*60)
print("CLUSTERING ANALYSIS COMPLETE")
print("="*60)
print(f"Best K            : {best_k}")
print(f"Best Silhouette   : {best_silhouette:.4f}")
print(f"\nSegment Distribution:")
print(rfm_clean["Segment"].value_counts())
print("\nFiles saved:")
print("  best_kmeans_model.pkl")
print("  rfm_with_clusters.csv")
print("  cluster_profiles.csv")
print("  10_elbow_curve.png")
print("  11_silhouette_scores.png")
print("  12_cluster_scatter_recency_monetary.png")
print("  13_cluster_scatter_frequency_monetary.png")
print("  14_cluster_3d_plot.png")
print("\nDone!")