import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.parse import urlparse
import matplotlib.ticker as mticker

# Create folder for EDA results if it doesn't exist
os.makedirs("EDA Result", exist_ok=True)

# ----------------------------------
# 1. Read the Unified Dataset
# ----------------------------------
df = pd.read_csv("unified_dataset.csv")  # Adjust path if necessary

# ----------------------------------
# 2. Quick Overview
# ----------------------------------
print("Dataset Shape:", df.shape)
print("Column Info:")
print(df.info())
print("\nPreview of Data:\n", df.head())
print("\nClass Distribution (0=Legitimate, 1=Phishing):\n", df['type'].value_counts())

# ----------------------------------
# 3. Feature Engineering for EDA
# ----------------------------------

# 3.1 URL Length
df['url_length'] = df['url'].astype(str).apply(len)

# 3.2 Special Character Count
special_chars = ['@', '-', '_', '?', '=', '&', '%', '.', '/']
def count_special_chars(url):
    return sum(url.count(char) for char in special_chars)
df['special_char_count'] = df['url'].astype(str).apply(count_special_chars)

# 3.3 Subdomain Count (Count of '.' in hostname)
def count_subdomains(url):
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return 0
        return hostname.count('.')
    except Exception:
        return 0
df['subdomain_count'] = df['url'].astype(str).apply(count_subdomains)

# 3.4 Suspicious Keyword Count
suspicious_keywords = ["login", "secure", "account", "update", "verify", "wp-admin"]
def count_suspicious_keywords(url):
    url_lower = url.lower()
    return sum(keyword in url_lower for keyword in suspicious_keywords)
df['suspicious_keyword_count'] = df['url'].astype(str).apply(count_suspicious_keywords)

# Split data into Legitimate (type=0) and Phishing (type=1)
df_legit = df[df['type'] == 0]
df_phish = df[df['type'] == 1]

# ----------------------------------
# 4. Plotting and Saving Histograms for Each Feature Separately
# ----------------------------------

# Function to enforce integer tick labels on the y-axis
def set_integer_yticks(ax):
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# 4.1 URL Length Distribution with Outlier Detection (IQR Method)
# Compute IQR and determine the upper threshold
Q1 = df['url_length'].quantile(0.25)
Q3 = df['url_length'].quantile(0.75)
IQR = Q3 - Q1
upper_threshold = Q3 + 1.5 * IQR

# Identify outliers based on URL length
df_outliers = df[df['url_length'] > upper_threshold]
if not df_outliers.empty:
    outlier_file = os.path.join("EDA Result", "url_length_outliers.txt")
    # Write only the url, type, and url_length columns to the file
    df_outliers[['url', 'type', 'url_length']].to_csv(outlier_file, sep='\t', index=False)
    print(f"Outliers saved to {outlier_file}")

# Filter non-outliers for plotting
df_non_outliers = df[df['url_length'] <= upper_threshold]

# For separate class plots, filter each for non-outliers
df_legit_non_outliers = df_legit[df_legit['url_length'] <= upper_threshold]
df_phish_non_outliers = df_phish[df['url_length'] <= upper_threshold]

# Define bins with gap 40 up to the upper_threshold
bins_url_length = np.arange(0, upper_threshold + 40, 40)

# Legitimate URLs
plt.figure(figsize=(10,5))
plt.hist(df_legit_non_outliers['url_length'], bins=bins_url_length, edgecolor='black')
plt.xlabel("URL Length")
plt.ylabel("Frequency")
plt.title("URL Length Distribution (Legitimate URLs, Non-Outliers)")
plt.xlim(0, upper_threshold)
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "url_length_distribution_legitimate.png"))
plt.close()

# Phishing URLs
plt.figure(figsize=(10,5))
plt.hist(df_phish_non_outliers['url_length'], bins=bins_url_length, edgecolor='black')
plt.xlabel("URL Length")
plt.ylabel("Frequency")
plt.title("URL Length Distribution (Phishing URLs, Non-Outliers)")
plt.xlim(0, upper_threshold)
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "url_length_distribution_phishing.png"))
plt.close()

# 4.2 Special Character Count Distribution
bins_special = np.arange(0, 61, 1)  # 0 to 60
# Legitimate URLs
plt.figure(figsize=(10,5))
plt.hist(df_legit['special_char_count'], bins=bins_special, edgecolor='black')
plt.xlabel("Count of Special Characters")
plt.ylabel("Frequency")
plt.title("Special Character Count (Legitimate URLs)")
plt.xlim(0,60)
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "special_char_count_legitimate.png"))
plt.close()

# Phishing URLs
plt.figure(figsize=(10,5))
plt.hist(df_phish['special_char_count'], bins=bins_special, edgecolor='black')
plt.xlabel("Count of Special Characters")
plt.ylabel("Frequency")
plt.title("Special Character Count (Phishing URLs)")
plt.xlim(0,60)
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "special_char_count_phishing.png"))
plt.close()

# 4.3 Subdomain Count Distribution (Number of '.' in hostname)
bins_subdomain = np.arange(0, 8, 1)  # 0 to 7
# Legitimate URLs
plt.figure(figsize=(10,5))
plt.hist(df_legit['subdomain_count'], bins=bins_subdomain, edgecolor='black')
plt.xlabel("Number of '.' in Hostname")
plt.ylabel("Frequency")
plt.title("Subdomain Count (Legitimate URLs)")
plt.xlim(0,7)
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "subdomain_count_legitimate.png"))
plt.close()

# Phishing URLs
plt.figure(figsize=(10,5))
plt.hist(df_phish['subdomain_count'], bins=bins_subdomain, edgecolor='black')
plt.xlabel("Number of '.' in Hostname")
plt.ylabel("Frequency")
plt.title("Subdomain Count (Phishing URLs)")
plt.xlim(0,7)
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "subdomain_count_phishing.png"))
plt.close()

# 4.4 Suspicious Keyword Count Distribution
max_keyword = int(df['suspicious_keyword_count'].max())
bins_keyword = np.arange(0, max_keyword + 2, 1)  # +2 to include maximum count
# Legitimate URLs
plt.figure(figsize=(10,5))
plt.hist(df_legit['suspicious_keyword_count'], bins=bins_keyword, edgecolor='black')
plt.xlabel("Suspicious Keywords in URL")
plt.ylabel("Frequency")
plt.title("Suspicious Keyword Count (Legitimate URLs)")
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "suspicious_keyword_count_legitimate.png"))
plt.close()

# Phishing URLs
plt.figure(figsize=(10,5))
plt.hist(df_phish['suspicious_keyword_count'], bins=bins_keyword, edgecolor='black')
plt.xlabel("Suspicious Keywords in URL")
plt.ylabel("Frequency")
plt.title("Suspicious Keyword Count (Phishing URLs)")
ax = plt.gca()
set_integer_yticks(ax)
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "suspicious_keyword_count_phishing.png"))
plt.close()

# ----------------------------------
# 5. Correlation Analysis
# ----------------------------------
numeric_cols = ['url_length', 'special_char_count', 'subdomain_count', 'suspicious_keyword_count', 'type']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Matrix (Numeric Features vs. Type)")
plt.tight_layout()
plt.savefig(os.path.join("EDA Result", "correlation_matrix.png"))
plt.close()

# ----------------------------------
# 6. Summaries & Next Steps
# ----------------------------------
print("\nDescriptive Statistics of Numeric Features:\n", df[numeric_cols].describe())
print("\nCorrelation Matrix:\n", corr_matrix)
