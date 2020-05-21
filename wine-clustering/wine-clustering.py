import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# DATA IMPORT AND MANIPULATION ##################################################################################################
all_data = load_wine(return_X_y=False)

wine_features = all_data.feature_names
wine_data = all_data.data

wine_df = pd.DataFrame(wine_data, columns = wine_features)

wine_df.info()

# PRINCIPAL COMPONENT ANALYSIS OF WINE DATA #####################################################################################

dp = wine_df.loc[:, wine_features].values
dp = StandardScaler().fit_transform(dp)

k = 3;
pca= decomposition.PCA(n_components=k)
pca.fit(dp)
dp = pca.transform(dp)

#print("\nWINE DATA SCORES")
#print(dp)

print("\nGENERATE WINE PCs PLOT")
fig = plt.figure()
#ax = fig.add_subplot(111)
#ax = Axes3D(fig, elev=48, azim=134)
ax = Axes3D(fig, elev=10, azim=85)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], color='c', marker='o', edgecolor='b')
plt.savefig('WINE-scatterPCA-3d.png', dpi = 250)
plt.clf()

print("EXPLAINED VARIANCE BY FIRST " + str(k) + " PRINCIPAL COMPONENTS")
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

# CLUSTERING OF WINE DATA ########################################################################################################

print("\nINITIALIZE CLUSTERING")
points = dp

# create kmeans object
clusters = 3
kmeans = KMeans(n_clusters=clusters)
print("COMPUTING " + str(clusters) + " CLUSTERS")

# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
print('K-means cluster centroids:\n', kmeans.cluster_centers_)

# save new clusters for chart
y_km = kmeans.fit_predict(points)

#k means plot
fig = plt.figure()
ax = Axes3D(fig, elev=10, azim=85)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.scatter(points[y_km ==0,0], points[y_km == 0,1], points[y_km == 0,2], s=50, c='red')
ax.scatter(points[y_km ==1,0], points[y_km == 1,1], points[y_km == 1,2], s=50, c='black')
ax.scatter(points[y_km ==2,0], points[y_km == 2,1], points[y_km == 2,2], s=50, c='blue')
#ax.scatter(points[y_km ==3,0], points[y_km == 3,1], points[y_km == 3,2], s=50, c='cyan')
plt.title('K-means wine clustering')
#plt.legend(['C1','C2', 'C3', 'C4'], loc = 2, bbox_to_anchor = (0.1, 0.93), ncol=4)
plt.legend(['C0','C1', 'C2'], loc = 2, bbox_to_anchor = (0.1, 0.93), ncol=3)
plt.savefig('WINE-K-means-clusters-3.png', dpi = 250)
plt.clf()

print('Clusters computed')

##############################################################################################################################################################################

# 2-D WINE DATA ########################################################################################################

print("\nGENERATE 2-D WINE PLOT")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.scatter(dp[:, 0], dp[:, 1], color='c', marker='o', edgecolor='b')
plt.savefig('2d-WINE-scatterPCA-3d.png', dpi = 250)
plt.clf()

#k means plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.scatter(points[y_km ==0,0], points[y_km == 0,1], s=50, c='red')
ax.scatter(points[y_km ==1,0], points[y_km == 1,1], s=50, c='black')
ax.scatter(points[y_km ==2,0], points[y_km == 2,1], s=50, c='blue')
#ax.scatter(points[y_km ==3,0], points[y_km == 3,1], s=50, c='cyan')
plt.title('K-means wine clustering')
#plt.legend(['C1','C2', 'C3', 'C4'], loc = 2, bbox_to_anchor = (0.1, 0.93), ncol=4)
plt.legend(['C0','C1', 'C2'], loc = 2, bbox_to_anchor = (0.01, 0.98), ncol=3)
plt.savefig('2d-WINE-K-means-clusters-3.png', dpi = 250)
plt.clf()


