import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def main():
    # Load the dataset
    df = pd.read_csv("kaggle_phishing_dataset.csv")
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    # Separate features and target label (assumed column 'class' with phishing = 1, legitimate = -1)
    X = df.drop('class', axis=1)
    y = df['class']

    # Split into training and test sets (70% train, 30% test) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Define the classifiers to evaluate
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    # Evaluate each classifier
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...\n{'-' * 50}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Print the full classification report
        print(f"Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred))

        # Calculate additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
