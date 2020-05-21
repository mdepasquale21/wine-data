# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#import data
f = open("./pc1-pc2-completetn-aromagroups.txt")
x = np.loadtxt(f, delimiter='\t', skiprows=1)
# create np array for data points
data = np.array(x).astype("float")
#data[i][j], i varies the row (chooses the coordinates [pc1, pc2] at row i)
#data[i][j], j varies the column (chooses between pc1 and pc2 respectively 0 or 1)

print('data imported')

#prepare for clustering
points = data

# import KMeans
from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans

###########################################################################################################################
# create kmeans object
kmeans = KMeans(k=4)
# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
print('K-means cluster centroids:\n', kmeans.cluster_centers_)
###########################################################################################################################
# create kmedians object
kmedians = KMedians(k=4)
# fit kmeans object to data
kmedians.fit(points)
# print location of clusters learned by kmedians object
print('K-medians cluster centroids:\n', kmedians.cluster_centers_)
###########################################################################################################################
# create kmeansf (fuzzy) object
kmeansf = FuzzyKMeans(k=4)
# fit kmeans object to data
kmeansf.fit(points)
# print location of clusters learned by kmeans fuzzy object
print('K-means fuzzy cluster centroids:\n', kmeansf.cluster_centers_)
###########################################################################################################################

fig = plt.figure()
colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#8B0000']

objects = (kmeans, kmedians, kmeansf)
X = points

for i, obj in enumerate(objects):
    ax = fig.add_subplot(1, len(objects), i + 1)
    for k, col in zip(range(obj.k), colors):
        my_members = obj.labels_ == k
        cluster_center = obj.cluster_centers_[k]
        plt.xlim(-4.5, 4)
        plt.ylim(-6.5, 5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='o')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                        markeredgecolor='k', markersize=7)
    ax.set_title(obj.__class__.__name__)

print('clusters computed with K-means, K-medians and Fuzzy K-means')

plt.show()

