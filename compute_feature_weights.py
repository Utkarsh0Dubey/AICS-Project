# compute_feature_weights.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = "kaggle_phishing_dataset.csv"
OUTPUT_PATH = "feature_weights.csv"
SEED = 42
N_EST = 100


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)
    X = df.drop('class', axis=1)
    y = df['class']

    # 70/30 split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )

    # Train global RF
    rf = RandomForestClassifier(n_estimators=N_EST, random_state=SEED)
    rf.fit(X_train, y_train)

    # Extract importances and compute weights
    importances = rf.feature_importances_
    weights = np.sqrt(importances)

    # Build DataFrame
    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances,
        'weight': weights
    }).sort_values('importance', ascending=False)

    # Print to console
    print(feat_df.to_string(index=False))

    # Save to CSV
    feat_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved feature importances and weights to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()
