import pandas as pd
from urllib.parse import urlparse

# Load the original unified dataset which has only 'url' and 'type'
df = pd.read_csv("unified_dataset.csv")

# Compute URL Length
df['url_length'] = df['url'].astype(str).apply(len)

# Compute Special Character Count
special_chars = ['@', '-', '_', '?', '=', '&', '%', '.', '/']
df['special_char_count'] = df['url'].astype(str).apply(lambda url: sum(url.count(c) for c in special_chars))

# Compute Subdomain Count (Count of dots in the hostname)
df['subdomain_count'] = df['url'].astype(str).apply(
    lambda url: urlparse(url).hostname.count('.') if urlparse(url).hostname is not None else 0
)

# Compute Suspicious Keyword Count
suspicious_keywords = ["login", "secure", "account", "update", "verify", "wp-admin"]
df['suspicious_keyword_count'] = df['url'].astype(str).apply(
    lambda url: sum(keyword in url.lower() for keyword in suspicious_keywords)
)

# Save the enhanced dataset to a new CSV file
df.to_csv("enhanced_unified_dataset.csv", index=False)

print("Enhanced dataset saved as 'enhanced_unified_dataset.csv'")
