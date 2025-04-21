import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# Load the data
df = pd.read_csv("kaggle_phishing_dataset.csv")
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)
X = df.drop('class', axis=1)
y = df['class']

# Model and CV setup
rf = RandomForestClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy','precision','recall','f1']

# Run CV
scores = cross_validate(rf, X, y, cv=skf, scoring=scoring, n_jobs=-1)

# Print fold‑wise scores and the aggregate
for metric in scoring:
    vals = scores[f'test_{metric}']
    print(f"\n{metric.capitalize()} per fold: {np.round(vals,4)}")
    print(f"Mean {metric}: {np.mean(vals):.4f}, Std: {np.std(vals):.4f}")
