import matplotlib.pyplot as plt

# Define nodes and their y-positions
nodes = [
    "Data Preparation",
    "Baseline Model Comparison",
    "Random-Forest Cross-Validation",
    "Cluster-Specific Modeling",
    "Cluster Feature Analysis",
    "Feature-Weighted Clustering",
    "Clustering Validation"
]
ys = [1.0 - i*0.12 for i in range(len(nodes))]

# Create plot
fig, ax = plt.subplots(figsize=(6, 8))
for y, label in zip(ys, nodes):
    ax.text(0.5, y, label, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

# Draw arrows
for i in range(len(ys)-1):
    ax.annotate('', xy=(0.5, ys[i+1]+0.03), xytext=(0.5, ys[i]-0.03),
                arrowprops=dict(arrowstyle='->', lw=1))

ax.set_xlim(0, 1)
ax.set_ylim(min(ys)-0.1, max(ys)+0.1)
ax.axis('off')
plt.show()
