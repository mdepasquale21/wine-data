
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

## DBSCAN not very good for this little quantity of wine tasting data!

# #############################################################################
#import data
f = open("./pc1-pc2-completetn-aromagroups.txt")
x = np.loadtxt(f, delimiter='\t', skiprows=1)
# create np array for data points
data = np.array(x).astype("float")

#data[i][j], i varies the row (chooses the coordinates [pc1, pc2] at row i)
#data[i][j], j varies the column (chooses between pc1 and pc2 respectively 0 or 1)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=1.0, min_samples=4).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))
# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('DBSCAN Estimated number of clusters: %d' % n_clusters_)
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig('dbscan-clusters-wine-data.png', dpi = 250)

