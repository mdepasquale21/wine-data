
import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
#import data
f = open("./pc1-pc2-completetn-aromagroups.txt")
x = np.loadtxt(f, delimiter='\t', skiprows=1)
# create np array for data points
data = np.array(x).astype("float")

#data[i][j], i varies the row (chooses the coordinates [pc1, pc2] at row i)
#data[i][j], j varies the column (chooses between pc1 and pc2 respectively 0 or 1)

X = data
# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Affinity propagation estimated number of clusters: %d' % n_clusters_)
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig('affinity-prop-clusters-wine-data.png', dpi = 250)

