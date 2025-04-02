# Step 1: Import required libraries and create a results folder
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
import shap
import warnings
warnings.filterwarnings('ignore')

# For LIME explanations
import lime
import lime.lime_tabular

# Create results folder if it doesn't exist
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Step 2: Load the dataset from the project folder
df = pd.read_csv('phishing.csv')
print("Dataset head:")
print(df.head())

# If an 'Index' column exists, drop it for cleaner processing
if 'Index' in df.columns:
    df.drop('Index', axis=1, inplace=True)

# Step 3: Basic EDA - Check dataset information and missing values
print("\nDataset info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 4: Visualize and save the distribution of the target variable
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title('Distribution of Classes')
plt.xlabel('Class (0: Legitimate, 1: Phishing)')
plt.ylabel('Count')
plt.savefig(os.path.join(results_dir, "class_distribution.png"))
plt.show()

# Step 5: Correlation heatmap to visualize relationships between features
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"))
plt.show()

# Step 6: Split the dataset into features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Step 7: Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Baseline Model: RandomForest
# -----------------------------

# Step 8: Train a RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 9: Evaluate the RandomForest model
y_pred = rf_model.predict(X_test)
print("\nClassification Report for RandomForest:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for RandomForest
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - RandomForest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(results_dir, "confusion_matrix_rf.png"))
plt.show()

# ROC Curve for RandomForest
y_probs = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RandomForest')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, "roc_curve_rf.png"))
plt.show()

# Interactive ROC Curve using Plotly for RandomForest
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig.update_layout(title=f'Interactive ROC Curve - RandomForest (AUC={roc_auc:.2f})',
                  xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig.write_html(os.path.join(results_dir, "interactive_roc_rf.html"))
fig.show()

# Step 10: Use SHAP to explain the RandomForest model predictions using the new SHAP API
# Convert X_test to a NumPy array for consistency
X_test_np = X_test.values

# Create a SHAP Explainer using the unified API (works well for tree models)
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test_np)

# Verify the shapes for debugging
print("X_test_np shape:", X_test_np.shape)
print("SHAP values shape:", shap_values.values.shape)

# Generate the SHAP summary plot and save the figure
shap.summary_plot(shap_values.values, X_test_np, feature_names=X_test.columns, show=False)
plt.savefig(os.path.join(results_dir, "shap_summary.png"), bbox_inches="tight")
plt.close()

# ------------------------------------
# Ensemble Model: Voting Classifier
# ------------------------------------

# Step 11: Build an ensemble model using VotingClassifier
# Define individual classifiers
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)

# Combine them into a soft voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('gb', gb),
    ('svc', svc)
], voting='soft')
ensemble_model.fit(X_train, y_train)

# Step 12: Evaluate the ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)
print("\nClassification Report for Ensemble Model:")
print(classification_report(y_test, y_pred_ensemble))

# Confusion Matrix for Ensemble Model
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6,4))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(results_dir, "confusion_matrix_ensemble.png"))
plt.show()

# ROC Curve for Ensemble Model
y_probs_ensemble = ensemble_model.predict_proba(X_test)[:, 1]
fpr_e, tpr_e, thresholds_e = roc_curve(y_test, y_probs_ensemble)
roc_auc_e = auc(fpr_e, tpr_e)
plt.figure(figsize=(6,4))
plt.plot(fpr_e, tpr_e, label=f'ROC curve (AUC = {roc_auc_e:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Ensemble Model')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, "roc_curve_ensemble.png"))
plt.show()

# Interactive ROC Curve for Ensemble Model using Plotly
fig_e = go.Figure()
fig_e.add_trace(go.Scatter(x=fpr_e, y=tpr_e, mode='lines', name='ROC curve'))
fig_e.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig_e.update_layout(title=f'Interactive ROC Curve - Ensemble Model (AUC={roc_auc_e:.2f})',
                    xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig_e.write_html(os.path.join(results_dir, "interactive_roc_ensemble.html"))
fig_e.show()

# ------------------------------------
# Additional Original Ideas
# ------------------------------------

# 1. Adversarial Robustness Testing:
# Generate adversarial examples by adding a small Gaussian noise to the test set
noise_factor = 0.01  # Adjust noise factor as needed
# Cast X_test_np to float64 to allow addition of noise
X_test_adv = X_test_np.copy().astype(np.float64)
X_test_adv += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test_adv.shape)

# Evaluate RandomForest on adversarial examples
y_pred_adv = rf_model.predict(X_test_adv)
print("\nClassification Report for RandomForest on Adversarial Examples:")
print(classification_report(y_test, y_pred_adv))

# Evaluate Ensemble Model on adversarial examples
y_pred_adv_ensemble = ensemble_model.predict(X_test_adv)
print("\nClassification Report for Ensemble Model on Adversarial Examples:")
print(classification_report(y_test, y_pred_adv_ensemble))

# Save adversarial example results to text files
with open(os.path.join(results_dir, "rf_adversarial_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_adv))
with open(os.path.join(results_dir, "ensemble_adversarial_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_adv_ensemble))

# 2. t-SNE Visualization of the Test Data:
# Reduce the test data to 2 dimensions and visualize using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_test_embedded = tsne.fit_transform(X_test_np)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_test_embedded[:, 0], y=X_test_embedded[:, 1], hue=y_test, palette='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Test Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(title='Class', loc='best')
plt.savefig(os.path.join(results_dir, "tsne_visualization.png"))
plt.show()

# 3. LIME Explanations:
# Explain a single prediction using LIME for further interpretability
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Legitimate', 'Phishing'],
    discretize_continuous=True
)

# Pick an instance from the test set to explain
i = 0  # Change the index to examine different instances if desired
exp = explainer_lime.explain_instance(
    X_test.iloc[i].values,
    rf_model.predict_proba,
    num_features=10
)
# Save the explanation to an HTML file for viewing
lime_explanation_path = os.path.join(results_dir, 'lime_explanation.html')
exp.save_to_file(lime_explanation_path)
print(f"\nLIME explanation for test instance saved to {lime_explanation_path}")

# -----------------------------
# End of Experiment Workflow
# -----------------------------
print("Experiment complete! All results saved in the 'results' folder.")
