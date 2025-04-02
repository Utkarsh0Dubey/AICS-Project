import os
import pandas as pd
import numpy as np

# Import model training and evaluation libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.utils import class_weight
from joblib import dump

# Create directories for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ----------------------------
# 1. Load the Enhanced Dataset
# ----------------------------
# This CSV should include the original two columns (url, type) plus the engineered features.
df = pd.read_csv("enhanced_unified_dataset.csv")

# ----------------------------
# 2. Define Features and Target
# ----------------------------
features = ['url_length', 'special_char_count', 'subdomain_count', 'suspicious_keyword_count']
X = df[features]
y = df['type']

# ----------------------------
# 3. Split Data into Training and Test Sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 4. Compute Class Weights and Scale for Boosting Models
# ----------------------------
# For models that support class_weight:
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# For LightGBM and XGBoost, calculate scale_pos_weight as ratio of negatives to positives
n_negative = sum(y_train == 0)
n_positive = sum(y_train == 1)
scale_pos_weight = n_negative / n_positive

# ----------------------------
# 5. Define Multiple Models with Configuration
# ----------------------------
models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),  # Note: KNN does not support class_weight.
    "LightGBM": lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False,
                                 eval_metric='logloss', random_state=42)
}

# ----------------------------
# 6. Train Models, Evaluate, and Save Results
# ----------------------------
results = {}  # To store evaluation metrics for each model

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    # Store the results in the dictionary
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Confusion Matrix": conf_mat
    }

    # Save the trained model to disk
    model_filename = os.path.join("models", f"{name}.pkl")
    dump(model, model_filename)
    print(f"{name} model saved to {model_filename}\n")

# ----------------------------
# 7. Display and Save Evaluation Results
# ----------------------------
# Convert the results dictionary to a DataFrame for easier viewing

results_df = pd.DataFrame({
    k: {**v, "Confusion Matrix": str(v["Confusion Matrix"])} for k, v in results.items()
}).T

print("Evaluation Results:")
print(results_df)
results_df.to_csv(os.path.join("results", "model_evaluation_results.csv"), index=True)

# Optionally, print full classification reports for each model
for name, model in models.items():
    print(f"Classification Report for {name}:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("\n")
