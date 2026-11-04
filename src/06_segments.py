import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

# ---------- Task 1: RFM table ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = (done.groupby("customer_id")
           .agg(last_order=("order_ts", "max"),
                frequency=("order_id", "nunique"),
                monetary=("order_amount", "sum"))
           .reset_index())
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("customers:", len(rfm), "(ref 1175)")
print("recency median/min/max:", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max(), "(ref 19 / 0 / 180)")
print("frequency mean/max:", round(rfm["frequency"].mean(), 2), rfm["frequency"].max(), "(ref 9.46 / 33)")
print("monetary mean/max:", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2), "(ref 2552.06 / 14766.55)")

# ---------- Task 2: K-means, K = 2..8 ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k,
                 "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
print("\n", pd.DataFrame(rows).to_string(index=False))

# ---------- Task 3a: profile the chosen K ----------
K = 4   # we will confirm this choice together once you've seen the table
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

profile = (rfm.groupby("segment_id")
              .agg(size=("customer_id", "count"),
                   mean_recency=("recency", "mean"),
                   mean_frequency=("frequency", "mean"),
                   mean_monetary=("monetary", "mean"))
              .round(2))
print("\nSegment profile (K=%d):\n" % K, profile)

# ---------- Task 3c: PCA ----------
pca = PCA(n_components=2).fit(X)
ratio = pca.explained_variance_ratio_
print("\nPCA explained variance:", ratio.round(3), "total:", round(ratio.sum(), 3), "(ref 0.634 / 0.218 / 0.852)")

# ---------- Task 3b: names, Premium flag, output file ----------
premium_id = profile["mean_monetary"].idxmax()            # highest mean monetary = Premium
rest = profile.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()                # most frequent of the others
dormant_id = rest["mean_recency"].idxmax()                # longest since last order

names = {premium_id: "Premium Big-Ticket Buyers",
         loyal_id: "Frequent Loyalists",
         dormant_id: "Dormant At-Risk"}
rfm["segment_name"] = rfm["segment_id"].map(names).fillna("Occasional Shoppers")
rfm["is_premium"] = rfm["segment_id"] == premium_id

rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)

print("\n", rfm.groupby(["segment_id", "segment_name", "is_premium"]).size())
print("premium customers:", rfm["is_premium"].sum(), "(should be 138)")