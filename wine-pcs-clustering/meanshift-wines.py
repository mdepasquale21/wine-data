
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
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
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Mean Shift estimated number of clusters: %d' % n_clusters_)
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig('mean-shift-clusters-wine-data.png', dpi = 250)

