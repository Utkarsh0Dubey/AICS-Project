import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Load dataset
df = pd.read_csv("kaggle_phishing_dataset.csv")
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)

X = df.drop('class', axis=1).values
y = df['class'].values

# 70/30 stratified split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

results = []
for k in [3, 5]:
    # Scale and cluster phishing TRAIN subset
    scaler = StandardScaler()
    mask_ph_tr = (y_tr == 1)
    X_ph_tr = scaler.fit_transform(X_tr[mask_ph_tr])
    km = KMeans(n_clusters=k, random_state=42).fit(X_ph_tr)

    # Predict cluster IDs on train/test
    labels_tr = km.labels_
    mask_ph_te = (y_te == 1)
    X_ph_te = scaler.transform(X_te[mask_ph_te])
    labels_te = km.predict(X_ph_te)

    # Build full-length cluster ID arrays
    cid_tr = np.full(len(X_tr), -1)
    cid_tr[mask_ph_tr] = labels_tr
    cid_te = np.full(len(X_te), -1)
    cid_te[mask_ph_te] = labels_te

    # Per-cluster evaluation
    for cid in range(k):
        train_size = (cid_tr == cid).sum()
        test_size  = (cid_te == cid).sum()

        # Prepare subsets: cluster + all legit
        tr_mask = (cid_tr == cid) | (y_tr == -1)
        te_mask = (cid_te == cid) | (y_te == -1)
        X_sub_tr, y_sub_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_sub_te, y_sub_te = X_te[te_mask], y_te[te_mask]

        # Train/test split evaluation
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_sub_tr, y_sub_tr)
        y_pred = rf.predict(X_sub_te)
        split_acc  = accuracy_score(y_sub_te, y_pred)
        split_prec = precision_score(y_sub_te, y_pred, pos_label=1)
        split_rec  = recall_score(y_sub_te, y_pred, pos_label=1)
        split_f1   = f1_score(y_sub_te, y_pred, pos_label=1)

        # 5‑fold CV on the same subset
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv = cross_validate(
            rf, X_sub_tr, y_sub_tr, cv=skf,
            scoring={'acc':'accuracy',
                     'prec':'precision',
                     'rec':'recall',
                     'f1':'f1'},
            n_jobs=-1
        )
        cv_acc  = cv['test_acc'].mean()
        cv_prec = cv['test_prec'].mean()
        cv_rec  = cv['test_rec'].mean()
        cv_f1   = cv['test_f1'].mean()

        results.append({
            'k': k,
            'cluster': cid,
            'train_size': train_size,
            'test_size': test_size,
            'split_acc': round(split_acc, 4),
            'split_prec': round(split_prec, 4),
            'split_rec': round(split_rec, 4),
            'split_f1': round(split_f1, 4),
            'cv_acc': round(cv_acc, 4),
            'cv_prec': round(cv_prec, 4),
            'cv_rec': round(cv_rec, 4),
            'cv_f1': round(cv_f1, 4),
        })

# Display as a DataFrame
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
