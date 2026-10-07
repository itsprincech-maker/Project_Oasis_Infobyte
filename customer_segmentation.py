# %% [markdown]
# Task 2 — Customer Segmentation Analysis (RFM + K-Means)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
DATA_FILE = Path("online_retail.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError("Place the e-commerce dataset as 'online_retail.csv' in this folder.")

df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# %%
def find_col(candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for actual in df.columns:
        a = actual.lower().replace("_", " ").strip()
        if any(c.lower() in a for c in candidates):
            return actual
    return None

customer_col = find_col(["customerid", "customer id", "customer"])
date_col = find_col(["invoicedate", "invoice date", "date", "order date"])
qty_col = find_col(["quantity", "qty"])
price_col = find_col(["unitprice", "unit price", "price"])

print("Detected:", customer_col, date_col, qty_col, price_col)

if not all([customer_col, date_col, qty_col, price_col]):
    raise ValueError("Could not detect Customer ID, Date, Quantity and Unit Price columns.")

df = df.dropna(subset=[customer_col, date_col, qty_col, price_col]).copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df = df.dropna(subset=[date_col, qty_col, price_col])
df["TotalAmount"] = df[qty_col] * df[price_col]
df = df[df["TotalAmount"] > 0]

# %%
# RFM features
snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
rfm = df.groupby(customer_col).agg(
    Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
    Frequency=(date_col, "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

rfm["AveragePurchaseValue"] = rfm["Monetary"] / rfm["Frequency"]
rfm["CustomerLifetimeValue"] = rfm["Monetary"]

print(rfm.describe())

# %%
# Standardisation
features = ["Recency", "Frequency", "Monetary"]
X = rfm[features].replace([np.inf, -np.inf], np.nan).dropna()
rfm = rfm.loc[X.index].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# %%
# Elbow Method
inertias = []
k_values = range(2, 11)
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(9,5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

# Default K after elbow inspection; change if the elbow in your dataset suggests another value.
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# %%
# Cluster visualisation 1
plt.figure(figsize=(9,6))
sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="tab10", s=70)
plt.title("Customer Clusters: Recency vs Monetary")
plt.tight_layout()
plt.show()

# %%
# Cluster visualisation 2
plt.figure(figsize=(9,6))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster", palette="tab10", s=70)
plt.title("Customer Clusters: Frequency vs Monetary")
plt.tight_layout()
plt.show()

# %%
# Cluster profiling
profile = rfm.groupby("Cluster")[features + ["AveragePurchaseValue", "CustomerLifetimeValue"]].mean().round(2)
counts = rfm["Cluster"].value_counts().sort_index()

print("Cluster profile:")
print(profile)
print("\nCustomers per cluster:")
print(counts)

plt.figure(figsize=(9,5))
counts.plot(kind="bar")
plt.title("Number of Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Customers")
plt.tight_layout()
plt.show()

# %%
# Customer-type description and marketing action
for cluster, row in profile.iterrows():
    print(f"\nCluster {cluster}")
    print(f"Recency: {row['Recency']:.1f}, Frequency: {row['Frequency']:.1f}, Monetary: {row['Monetary']:.2f}")
    if row["Monetary"] >= profile["Monetary"].median() and row["Frequency"] >= profile["Frequency"].median():
        print("Type: High-value / loyal customers")
        print("Marketing action: loyalty rewards, premium offers and cross-selling.")
    elif row["Recency"] <= profile["Recency"].median() and row["Monetary"] < profile["Monetary"].median():
        print("Type: Recent / developing customers")
        print("Marketing action: onboarding offers and personalized recommendations.")
    elif row["Recency"] > profile["Recency"].median():
        print("Type: At-risk or inactive customers")
        print("Marketing action: reactivation campaigns and limited-time incentives.")
    else:
        print("Type: Mid-value customers")
        print("Marketing action: targeted bundles and frequency-building promotions.")
