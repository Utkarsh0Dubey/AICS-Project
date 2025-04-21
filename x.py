import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# === Configuration ===
DATA_PATH = "kaggle_phishing_dataset.csv"  # path to your dataset
OUTPUT_DIR = "validation_results"          # folder to save outputs
K = 5                                       # number of clusters for KMeans
RANDOM_BASELINE_TRIALS = 5                 # how many random permutations to test

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and preprocess data ---
df = pd.read_csv(DATA_PATH)
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)

# Separate features and target
target = 'class'
FEATURES = [c for c in df.columns if c != target]

df_phish = df[df[target] == 1].reset_index(drop=True)
df_legit = df[df[target] == -1].reset_index(drop=True)

# Prepare feature matrix for clustering
scaler_clust = StandardScaler()
X_phish = df_phish[FEATURES].values
iX_phish_scaled = scaler_clust.fit_transform(X_phish)

# Classification pipeline (rescale + RF)
def make_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

# Compute global CV accuracy on entire dataset
def compute_global_cv(df_phish, df_legit):
    df_all = pd.concat([df_phish, df_legit], ignore_index=True)
    X_all = df_all[FEATURES].values
    y_all = df_all[target].values
    pipe = make_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_all, y_all, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Compute weighted cluster-specific CV accuracy
def compute_clustered_cv(X_phish_scaled, df_phish, df_legit, labels):
    total_phish = len(df_phish)
    weighted_acc = 0.0
    for cid in np.unique(labels):
        mask = (labels == cid)
        df_cluster = df_phish[mask]
        size = len(df_cluster)
        if size == 0:
            continue
        # combine cluster phish with all legit
        df_spec = pd.concat([df_cluster, df_legit], ignore_index=True)
        X_spec = df_spec[FEATURES].values
        y_spec = df_spec[target].values
        pipe = make_pipeline()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X_spec, y_spec, cv=skf, scoring='accuracy', n_jobs=-1)
        weighted_acc += (size / total_phish) * scores.mean()
    return weighted_acc

# Main
if __name__ == '__main__':
    # Global baseline
    global_acc = compute_global_cv(df_phish, df_legit)
    print(f"Global RF accuracy (5-fold CV): {global_acc:.4f}")

    # KMeans clustering
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(iX_phish_scaled)
    cluster_acc = compute_clustered_cv(iX_phish_scaled, df_phish, df_legit, labels)
    print(f"KMeans (k={K}) weighted cluster-specific accuracy: {cluster_acc:.4f}")

    # Random baseline: shuffle labels
    rand_accs = []
    for i in range(RANDOM_BASELINE_TRIALS):
        np.random.seed(42 + i)
        rand_labels = np.random.permutation(labels)
        acc = compute_clustered_cv(iX_phish_scaled, df_phish, df_legit, rand_labels)
        rand_accs.append(acc)
        print(f"Random baseline trial {i+1}: {acc:.4f}")
    print(f"Random baseline mean accuracy: {np.mean(rand_accs):.4f}\n")

    # Save results
    results = {
        'global_acc': global_acc,
        'cluster_acc': cluster_acc,
        'random_baseline_mean': np.mean(rand_accs)
    }
    pd.Series(results).to_csv(os.path.join(OUTPUT_DIR, 'clustering_validation.csv'))
    print(f"Results saved to '{OUTPUT_DIR}/clustering_validation.csv'")
