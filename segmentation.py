"""
Customer Segmentation using K-Means clustering (scikit-learn).

Steps
1. Load customer data (customers.csv). If missing, a realistic sample is generated.
2. Build features: demographics (age, income) + behavior (RFM: recency, frequency, monetary).
3. Scale features, choose k with the elbow method + silhouette score.
4. Fit K-Means, name each segment, and profile it.
5. Save charts (charts/) and results (customers_segmented.csv, segment_summary.csv).

Run:  python segmentation.py
Use your own data: put a customers.csv in this folder with the columns listed in FEATURES
below (plus CustomerID, Gender, PreferredCategory if you have them).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(HERE, "customers.csv")
CHART_DIR = os.path.join(HERE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)
FEATURES = ["Age", "AnnualIncome", "Recency", "Frequency", "Monetary"]
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------- 1. DATA
def make_sample(n=1200, seed=42):
    """Synthetic customers drawn from 4 hidden behaviour groups."""
    rng = np.random.default_rng(seed)
    groups = [  # share, age, income(k), recency(days), frequency, avg order value
        (0.20, 38, 95, 15, 28, 160),   # loyal high spenders
        (0.30, 29, 45, 40, 12, 55),    # young regulars
        (0.25, 52, 70, 120, 4, 90),    # older, occasional
        (0.25, 34, 35, 220, 2, 40),    # lapsed / low value
    ]
    rows = []
    for share, age, inc, rec, fre, aov in groups:
        m = int(n * share)
        a = rng.normal(age, 6, m).clip(18, 75)
        i = rng.normal(inc, 12, m).clip(12, 200)
        r = rng.gamma(4, rec / 4, m).clip(1, 365)
        f = rng.poisson(fre, m).clip(1, None)
        mon = f * rng.normal(aov, aov * 0.15, m).clip(10, None)
        rows.append(pd.DataFrame({"Age": a.round(), "AnnualIncome": (i * 1000).round(-2),
                                  "Recency": r.round(), "Frequency": f, "Monetary": mon.round(2)}))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "CustomerID", [f"C{1000 + i}" for i in range(len(df))])
    df.insert(2, "Gender", rng.choice(["Female", "Male"], len(df), p=[0.55, 0.45]))
    cats = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports"]
    df["PreferredCategory"] = [rng.choice(cats, p=_cat_probs(a, inc)) for a, inc in zip(df.Age, df.AnnualIncome)]
    return df


def _cat_probs(age, income):
    p = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    if age < 32: p += [1.5, 1.5, 0, 0.5, 1.5]
    if age > 45: p += [0, 0.3, 2.0, 1.0, 0]
    if income > 80000: p += [1.5, 0.5, 0.5, 1.0, 0]
    return p / p.sum()


if os.path.exists(CSV_IN):
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} customers from customers.csv")
else:
    df = make_sample()
    df.to_csv(CSV_IN, index=False)
    print(f"No customers.csv found - generated {len(df)} sample customers")

df = df.dropna(subset=FEATURES).drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------------- 2. FEATURES
X = df[FEATURES].copy()
X["Monetary"] = np.log1p(X["Monetary"])      # tame skew
X["Recency"] = np.log1p(X["Recency"])
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------- 3. CHOOSE K
ks = range(2, 10)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))
best_k = list(ks)[int(np.argmax(sil))]
print(f"Silhouette scores: { {k: round(s, 3) for k, s in zip(ks, sil)} }")
print(f"Chosen k = {best_k}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(list(ks), sil, "o-", color="tab:green"); ax[1].axvline(best_k, ls="--", color="gray")
ax[1].set(title="Silhouette score (higher = better)", xlabel="k", ylabel="Score")
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/1_choose_k.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 4. CLUSTER + NAME
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_scaled)
df["Cluster"] = km.labels_

prof = df.groupby("Cluster")[FEATURES].mean()
# Score each cluster: recent (low recency) + frequent + high spend = better
score = (-prof["Recency"].rank() + prof["Frequency"].rank() + prof["Monetary"].rank())
order = score.sort_values(ascending=False).index.tolist()
names_pool = ["Champions", "Loyal Regulars", "Potential Loyalists", "At Risk", "Hibernating", "Lost",
              "Newcomers", "Occasional", "Low Value"]
if best_k == 4:
    names_pool = ["Champions", "Loyal Regulars", "At Risk", "Lost / Low Value"]
name_map = {c: names_pool[i] for i, c in enumerate(order)}
df["Segment"] = df["Cluster"].map(name_map)

summary = df.groupby("Segment").agg(
    Customers=("CustomerID", "count") if "CustomerID" in df else ("Age", "count"),
    AvgAge=("Age", "mean"), AvgIncome=("AnnualIncome", "mean"),
    AvgRecencyDays=("Recency", "mean"), AvgOrders=("Frequency", "mean"),
    AvgTotalSpend=("Monetary", "mean"), TotalRevenue=("Monetary", "sum"),
).round(1)
summary["PctCustomers"] = (summary.Customers / summary.Customers.sum() * 100).round(1)
summary["PctRevenue"] = (summary.TotalRevenue / summary.TotalRevenue.sum() * 100).round(1)
if "PreferredCategory" in df:
    summary["TopCategory"] = df.groupby("Segment")["PreferredCategory"].agg(lambda s: s.mode()[0])
summary = summary.loc[[name_map[c] for c in order]]
print("\n", summary.to_string())

df.to_csv(f"{HERE}/customers_segmented.csv", index=False)
summary.to_csv(f"{HERE}/segment_summary.csv")

# ---------------------------------------------------------------- 5. VISUALS
seg_order = list(summary.index)
palette = dict(zip(seg_order, sns.color_palette("Set2", len(seg_order))))

# PCA scatter
pc = PCA(n_components=2).fit(X_scaled)
xy = pc.transform(X_scaled)
plt.figure(figsize=(7, 5.5))
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=df["Segment"], hue_order=seg_order, palette=palette, s=22, alpha=.8)
plt.title(f"Customer segments (PCA, {pc.explained_variance_ratio_.sum():.0%} variance)")
plt.xlabel("Component 1"); plt.ylabel("Component 2"); plt.tight_layout()
plt.savefig(f"{CHART_DIR}/2_segments_pca.png", dpi=150); plt.close()

# Segment size + revenue share
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].pie(summary.Customers, labels=seg_order, autopct="%1.0f%%", colors=[palette[s] for s in seg_order])
ax[0].set_title("Share of customers")
ax[1].bar(seg_order, summary.PctRevenue, color=[palette[s] for s in seg_order])
ax[1].set(title="Share of revenue (%)", ylabel="%"); ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/3_size_vs_revenue.png", dpi=150); plt.close()

# Profile heatmap (relative to overall average)
rel = (df.groupby("Segment")[FEATURES].mean() / df[FEATURES].mean()).loc[seg_order]
plt.figure(figsize=(7, 4))
sns.heatmap(rel, annot=True, fmt=".2f", cmap="RdYlGn", center=1)
plt.title("Segment profile vs. average customer (1.0 = average)"); plt.tight_layout()
plt.savefig(f"{CHART_DIR}/4_profile_heatmap.png", dpi=150); plt.close()

# Behaviour distributions
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col in zip(ax, ["Recency", "Frequency", "Monetary"]):
    sns.boxplot(data=df, x="Segment", y=col, order=seg_order, palette=palette, ax=a, hue="Segment", legend=False)
    a.set_title(col); a.set_xlabel(""); a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{CHART_DIR}/5_rfm_boxplots.png", dpi=150); plt.close()

# Purchase preferences
if "PreferredCategory" in df:
    ct = pd.crosstab(df["Segment"], df["PreferredCategory"], normalize="index").loc[seg_order] * 100
    ct.plot(kind="bar", stacked=True, figsize=(8, 4.5), colormap="tab20c")
    plt.title("Preferred category by segment (%)"); plt.ylabel("%"); plt.xticks(rotation=20)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/6_category_preferences.png", dpi=150); plt.close()

print("\nDone. See charts/, customers_segmented.csv and segment_summary.csv")
