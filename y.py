import pandas as pd

# Load the CSV
df = pd.read_csv("kaggle_phishing_dataset.csv")

# Count each class label
class_counts = df['class'].value_counts()
print("Class distribution:")
print(class_counts)

# If you want to see how many zeros appear in each feature column:
zero_counts = (df == 0).sum()
print("\nCount of zeros in each column:")
print(zero_counts)
