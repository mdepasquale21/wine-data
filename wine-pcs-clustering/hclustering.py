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

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
#dendrogram plot
plt.title('Hierarchical clustering')
plt.savefig('hclustering-dendrogram.png', dpi = 250)
plt.clf()

print('dendrogram computed')

# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)
#clusters plot
plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=100, c='black')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='blue')
plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='cyan')
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Hierarchical clustering data')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
fontpar = FontProperties()
fontpar.set_size('small')
plt.legend(['C1','C2', 'C3', 'C4'], loc = 2, prop=fontpar, bbox_to_anchor = (-0.01, 1.12), ncol=2)
plt.savefig('hclusters-wine-data.png', dpi = 250)
plt.clf()

print('clusters computed')
