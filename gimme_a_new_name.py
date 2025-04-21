import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# --- Configuration
DATA_PATH = "kaggle_phishing_dataset.csv"    # path to your CSV file
OUTPUT_DIR = "cluster_analysis_plots"        # folder for all output
KS = [3, 5, 7]                                # values of k to analyze
TOP_N_FEATS = 10                             # how many top features to inspect

# --- Prepare output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load dataset
df = pd.read_csv(DATA_PATH)
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)

# --- Features & target separation
FEATURES = [c for c in df.columns if c != 'class']
df_phish = df[df['class'] == 1].copy()
df_legit = df[df['class'] == -1].copy()

# --- Pre-scale phishing features for clustering
df_phish_vals = df_phish[FEATURES]
cluster_scaler = StandardScaler()
X_phish_scaled = cluster_scaler.fit_transform(df_phish_vals)

for k in KS:
    print(f"\n=== k-means: k={k} ===")
    # --- Fit k-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    phish_labels = kmeans.fit_predict(X_phish_scaled)
    df_phish['cluster'] = phish_labels

    # --- Stats on cluster sizes
    counts = df_phish['cluster'].value_counts().sort_index()
    print("Cluster sizes:")
    print(counts)

    # --- Analyze each cluster in turn
    for cid, size in counts.items():
        print(f"\n-- Cluster {cid} (size={size}) --")
        df_cluster = df_phish[df_phish['cluster'] == cid]

        # 1) Compare descriptive stats with overall phishing
        stats_cluster = df_cluster[FEATURES].describe().T[['mean','std']]
        stats_all     = df_phish[FEATURES].describe().T[['mean','std']]
        mean_diff     = (stats_cluster['mean'] - stats_all['mean']).abs().sort_values(ascending=False)
        top_feats     = mean_diff.head(TOP_N_FEATS).index.tolist()

        print("Top features by mean-shift:")
        print(mean_diff.head(TOP_N_FEATS))

        # 2) Plot distributions for top features
        for feat in top_feats:
            plt.figure(figsize=(5,3))
            sns.kdeplot(df_cluster[feat], label=f"Cluster {cid}", fill=True)
            sns.kdeplot(df_phish[feat], label="All Phish", fill=True)
            plt.title(f"k={k}, Cluster={cid}: {feat}")
            plt.legend()
            plt.tight_layout()
            # sanitize feature name for filename
            safe_feat = re.sub(r"[\\/*?:\"<>|]", "_", feat)
            fname = f"k{k}_c{cid}_{safe_feat}_kde.png"
            plt.savefig(os.path.join(OUTPUT_DIR, fname))
            plt.close()

        # 3) Train RF on (cluster + legit) and inspect importances
        df_special = pd.concat([df_cluster, df_legit], ignore_index=True)
        X_spec     = df_special[FEATURES]
        y_spec     = df_special['class']
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rf',     RandomForestClassifier(random_state=42))
        ])
        pipe.fit(X_spec, y_spec)
        importances = pd.Series(
            pipe.named_steps['rf'].feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        print("Top feature importances:")
        print(importances.head(TOP_N_FEATS))

print(f"\nAnalysis of all clusters complete. Plots in '{OUTPUT_DIR}'")
