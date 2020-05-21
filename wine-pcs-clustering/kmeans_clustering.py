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
from sklearn.cluster import KMeans

# create kmeans object
kmeans = KMeans(n_clusters=4)
# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
print('K-means cluster centroids:\n', kmeans.cluster_centers_)

# save new clusters for chart
y_km = kmeans.fit_predict(points)

#k means ++ plot
plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='red')
plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='black')
plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='blue')
plt.scatter(points[y_km ==3,0], points[y_km == 3,1], s=100, c='cyan')
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-means++ wine clustering')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
fontpar = FontProperties()
fontpar.set_size('small')
plt.legend(['C1','C2', 'C3', 'C4'], loc = 2, prop=fontpar, bbox_to_anchor = (-0.01, 1.15), ncol=4)
plt.savefig('K-means-clusters-wine-data-4.png', dpi = 250)
plt.clf()

print('4 clusters computed')

# create kmeans object
kmeans5 = KMeans(n_clusters=5)
# fit kmeans object to data
kmeans5.fit(points)
# print location of clusters learned by kmeans object
print('K-means cluster centroids:\n', kmeans5.cluster_centers_)

# save new clusters for chart
y_km5 = kmeans5.fit_predict(points)

#k means ++ plot
plt.scatter(points[y_km5 ==0,0], points[y_km5 == 0,1], s=100, c='red')
plt.scatter(points[y_km5 ==1,0], points[y_km5 == 1,1], s=100, c='black')
plt.scatter(points[y_km5 ==2,0], points[y_km5 == 2,1], s=100, c='blue')
plt.scatter(points[y_km5 ==3,0], points[y_km5 == 3,1], s=100, c='cyan')
plt.scatter(points[y_km5 ==4,0], points[y_km5 == 4,1], s=100, c='green')
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-means++ wine clustering')
#plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
fontpar = FontProperties()
fontpar.set_size('small')
plt.legend(['C1','C2', 'C3', 'C4', 'C5'], loc = 2, prop=fontpar, bbox_to_anchor = (-0.01, 1.15), ncol=5)
plt.savefig('K-means-clusters-wine-data-5.png', dpi = 250)
plt.clf()

print('5 clusters computed')
