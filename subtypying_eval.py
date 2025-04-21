import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

# Create folder for saving plots
output_folder = "new_plots"
os.makedirs(output_folder, exist_ok=True)


def plot_tsne_save(X, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def plot_cluster_distribution_save(labels, title, filename):
    counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def evaluate_cluster_cv(X_special, y_special, cv_splits=5):
    clf = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {'accuracy': 'accuracy',
               'precision': 'precision',
               'recall': 'recall',
               'f1': 'f1'}

    scores = cross_validate(clf, X_special, y_special, cv=skf, scoring=scoring, n_jobs=-1)
    avg_accuracy = np.mean(scores['test_accuracy'])
    avg_precision = np.mean(scores['test_precision'])
    avg_recall = np.mean(scores['test_recall'])
    avg_f1 = np.mean(scores['test_f1'])
    return avg_accuracy, avg_precision, avg_recall, avg_f1


def compute_cluster_statistics(X_scaled, labels):
    sil_vals = silhouette_samples(X_scaled, labels)
    clusters = np.unique(labels)
    stats = {}
    for cl in clusters:
        if len(X_scaled[labels == cl]) > 1:
            avg_silhouette = np.mean(sil_vals[labels == cl])
        else:
            avg_silhouette = 0.0
        cluster_data = X_scaled[labels == cl]
        var_per_feature = np.var(cluster_data, axis=0)
        avg_variance = np.mean(var_per_feature)
        stats[cl] = {"avg_silhouette": avg_silhouette, "avg_variance": avg_variance, "count": len(cluster_data)}
    return stats


def run_kmeans(X_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels


def main():
    # Load dataset
    df = pd.read_csv("kaggle_phishing_dataset.csv")
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    # Assume column 'class': 1 for phishing, -1 for legitimate
    df_phish = df[df['class'] == 1].copy()
    df_legit = df[df['class'] == -1].copy()

    # Extract phishing features and scale them
    X_phish = df_phish.drop('class', axis=1)
    scaler = StandardScaler()
    X_phish_scaled = scaler.fit_transform(X_phish)

    # Try k-means for k = 3, 5, 7
    for k in [3, 5, 7]:
        print(f"\n\n=== Running k-means for k = {k} ===")
        kmeans_labels = run_kmeans(X_phish_scaled, k)
        # Save visualizations
        plot_tsne_save(X_phish_scaled, kmeans_labels, f"t-SNE of K-means Clusters (k={k})", f"tsne_kmeans_k{k}.png")
        plot_cluster_distribution_save(kmeans_labels, f"Cluster Distribution for k-means (k={k})",
                                       f"bar_kmeans_k{k}.png")
        print(f"Plots for k={k} saved.")

        # Compute cluster statistics
        stats = compute_cluster_statistics(X_phish_scaled, kmeans_labels)
        for cl, stat in stats.items():
            print(
                f"Cluster {cl} (Count: {stat['count']}): Avg Silhouette = {stat['avg_silhouette']:.4f}, Avg Variance = {stat['avg_variance']:.4f}")

        # Evaluate specialized classifier performance via cross-validation for each cluster
        print(f"\nEvaluation for k-means clustering with k={k}:")
        unique_clusters = sorted(np.unique(kmeans_labels))
        for cl in unique_clusters:
            # Extract phishing samples in this cluster and reset index for alignment
            df_cluster = df_phish[kmeans_labels == cl].copy().reset_index(drop=True)
            if df_cluster.empty:
                continue
            # Combine with all legitimate samples
            df_special = pd.concat([df_cluster, df_legit], ignore_index=True)
            X_special = df_special.drop('class', axis=1)
            y_special = df_special['class']
            avg_acc, avg_prec, avg_rec, avg_f1 = evaluate_cluster_cv(X_special, y_special)
            print(f"\nCluster {cl} (Count: {len(df_cluster)} phishing samples):")
            print(f"  Average Accuracy:  {avg_acc:.4f}")
            print(f"  Average Precision: {avg_prec:.4f}")
            print(f"  Average Recall:    {avg_rec:.4f}")
            print(f"  Average F1-Score:  {avg_f1:.4f}")

        # Identify largest cluster by count for re-clustering
        largest_cluster = max(stats, key=lambda x: stats[x]["count"])
        print(
            f"\nLargest Cluster in k={k} is Cluster {largest_cluster} with {stats[largest_cluster]['count']} samples.")
        # Reset index in the largest cluster DataFrame for proper alignment
        df_cluster = df_phish[kmeans_labels == largest_cluster].copy().reset_index(drop=True)
        X_largest = scaler.transform(df_cluster.drop('class', axis=1))

        print("Re-clustering the largest cluster with k=2...")
        recluster_labels = run_kmeans(X_largest, k=2)
        new_stats = compute_cluster_statistics(X_largest, recluster_labels)
        for cl, stat in new_stats.items():
            print(
                f"Re-cluster {cl} (Count: {stat['count']}): Avg Silhouette = {stat['avg_silhouette']:.4f}, Avg Variance = {stat['avg_variance']:.4f}")

        # Evaluate specialized classifier on re-clustered groups
        for cl in np.unique(recluster_labels):
            # Use the reclustered array, which aligns with the reset-index DataFrame
            df_recluster = df_cluster.iloc[np.where(recluster_labels == cl)[0]].copy()
            if df_recluster.empty:
                continue
            df_spec_recluster = pd.concat([df_recluster, df_legit], ignore_index=True)
            X_spec_recluster = df_spec_recluster.drop('class', axis=1)
            y_spec_recluster = df_spec_recluster['class']
            avg_acc_r, avg_prec_r, avg_rec_r, avg_f1_r = evaluate_cluster_cv(X_spec_recluster, y_spec_recluster)
            print(f"\nRe-cluster {cl} (Count: {len(df_recluster)} phishing samples):")
            print(f"  Re-cluster Average Accuracy:  {avg_acc_r:.4f}")
            print(f"  Re-cluster Average Precision: {avg_prec_r:.4f}")
            print(f"  Re-cluster Average Recall:    {avg_rec_r:.4f}")
            print(f"  Re-cluster Average F1-Score:  {avg_f1_r:.4f}")

    print("\nCompleted cross-validation evaluation for k-means clustering at k = 3, 5, and 7.\n")


if __name__ == "__main__":
    main()
"""
Subtype Cluster Performance Analysis
To investigate why the largest k‑means clusters exhibit lower specialized classification performance, we conducted a detailed analysis of cluster size, internal cohesion, and the effect of re‑clustering. The key findings are summarized below:

Performance Degradation in Largest Clusters
Across all evaluated values of 
𝑘
k (3, 5, and 7), the most populous cluster consistently yielded slightly lower average cross‑validated accuracy than its smaller counterparts. For example, at 
𝑘
=
3
k=3, the largest cluster (Cluster 0, 4390 samples) attained an average accuracy of 97.04%, whereas Clusters 1 and 2—each containing fewer samples—achieved 99.79% and 99.74%, respectively. A similar pattern holds for 
𝑘
=
5
k=5 (largest cluster accuracy 97.89% vs. >99.80% for smaller clusters) and 
𝑘
=
7
k=7 (largest cluster accuracy 97.91% vs. >99.80%). This trend indicates that increasing cluster size alone does not guarantee improved classifier performance.

Cluster Heterogeneity as the Primary Driver
We measured each cluster’s internal cohesion via its average silhouette score and feature‑space variance. Although the large clusters sometimes exhibited moderate silhouette scores (e.g. 0.3280 at 
𝑘
=
3
k=3), their within‑cluster variance was comparatively high. In contrast, smaller clusters with lower variance consistently supported near‑perfect classification. These observations imply that the largest clusters encompass a wider diversity of URL features—i.e., they are more heterogeneous—making it more difficult for a single classifier to learn a consistent decision boundary.

Re‑clustering Validates Heterogeneity Hypothesis
To confirm that heterogeneity, rather than mere sample count, drives the reduced performance, we re‑clustered the largest group (e.g. Cluster 0 at 
𝑘
=
3
k=3) into two sub‑clusters. Each re‑clustered subgroup achieved approximately 98.0% accuracy—an appreciable increase from the original 97.04%. The similar sizes of these two subgroups paired with their improved performance indicate that partitioning a heterogeneous cluster into more homogeneous segments enhances classification accuracy.

Implications for Phishing Subtyping
These results demonstrate that the “catch‑all” nature of the largest cluster masks important subtypes of phishing URLs. By identifying and splitting heterogeneous clusters, we can create specialized classifiers that better capture subtle URL variations. Future work should explore alternative clustering metrics (e.g., intra‑cluster dispersion) or methods (e.g., hierarchical clustering) to further refine subtype discovery and thereby improve detection robustness.

In summary, the slight performance drop observed in the largest k‑means clusters is principally attributable to their internal heterogeneity. Re‑clustering these groups into more consistent subtypes mitigates this effect and underscores the value of clustering‑based subtyping in phishing URL detection.
"""