# weighted_kmeans_rf.py
"""
Feature-weighted k-Means clustering + per-cluster RF evaluation.

Usage:
    python weighted_kmeans_rf.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------- Configuration --------
DATA_PATH = "kaggle_phishing_dataset.csv"
KS        = [3, 5]      # values of k to evaluate
N_EST     = 100         # number of trees in RandomForest
SEED      = 42
# --------------------------------

def evaluate_clusters(X_tr, y_tr, X_te, y_te, labels_tr, labels_te, k):
    """Train & evaluate per-cluster RF on split and CV."""
    # Map full-length cluster IDs (-1 marks legitimate)
    cid_tr = np.full(len(X_tr), -1)
    cid_tr[y_tr == 1] = labels_tr
    cid_te = np.full(len(X_te), -1)
    cid_te[y_te == 1] = labels_te

    rf_base = RandomForestClassifier(n_estimators=N_EST, random_state=SEED)
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for cid in range(k):
        # build cluster-specific subsets
        tr_mask = (cid_tr == cid) | (y_tr == -1)
        te_mask = (cid_te == cid) | (y_te == -1)
        X_sub_tr, y_sub_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_sub_te, y_sub_te = X_te[te_mask], y_te[te_mask]

        # train/test split evaluation
        rf = RandomForestClassifier(n_estimators=N_EST, random_state=SEED)
        rf.fit(X_sub_tr, y_sub_tr)
        y_pred = rf.predict(X_sub_te)
        split_acc  = accuracy_score(y_sub_te, y_pred)
        split_prec = precision_score(y_sub_te, y_pred, pos_label=1)
        split_rec  = recall_score(y_sub_te, y_pred, pos_label=1)
        split_f1   = f1_score(y_sub_te, y_pred, pos_label=1)

        # 5-fold CV evaluation
        cv = cross_validate(
            rf_base, X_sub_tr, y_sub_tr, cv=skf,
            scoring={'acc':'accuracy','prec':'precision',
                     'rec':'recall','f1':'f1'},
            n_jobs=-1)
        cv_acc  = cv['test_acc'].mean()
        cv_prec = cv['test_prec'].mean()
        cv_rec  = cv['test_rec'].mean()
        cv_f1   = cv['test_f1'].mean()

        # display results
        print(f"Cluster {cid}  (train={len(X_sub_tr)}, test={len(X_sub_te)})")
        print(f"  70/30 split:   Acc {split_acc:.4f}, Prec {split_prec:.4f}, Rec {split_rec:.4f}, F1 {split_f1:.4f}")
        print(f"  5-fold CV  :   Acc {cv_acc:.4f}, Prec {cv_prec:.4f}, Rec {cv_rec:.4f}, F1 {cv_f1:.4f}\n")


def main():
    # load and split
    df = pd.read_csv(DATA_PATH)
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)
    X = df.drop('class', axis=1).values
    y = df['class'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )

    # train baseline RF to get feature importances
    rf_global = RandomForestClassifier(n_estimators=N_EST, random_state=SEED)
    rf_global.fit(X_tr, y_tr)
    importances = rf_global.feature_importances_
    weights = np.sqrt(importances)

    for k in KS:
        print(f"\n=== Weighted k-Means (k={k}) ===\n")
        # scale phishing subset
        scaler = StandardScaler()
        mask_ph_tr = (y_tr == 1)
        X_ph_tr = scaler.fit_transform(X_tr[mask_ph_tr])
        X_ph_tr_w = X_ph_tr * weights
        km = KMeans(n_clusters=k, random_state=SEED).fit(X_ph_tr_w)
        labels_tr = km.labels_

        mask_ph_te = (y_te == 1)
        X_ph_te = scaler.transform(X_te[mask_ph_te])
        X_ph_te_w = X_ph_te * weights
        labels_te = km.predict(X_ph_te_w)

        evaluate_clusters(X_tr, y_tr, X_te, y_te, labels_tr, labels_te, k)


if __name__ == "__main__":
    main()
