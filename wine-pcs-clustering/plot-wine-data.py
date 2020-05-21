import numpy as np
import matplotlib.pyplot as plt

#import data
f = open("./pc1-pc2-completetn-aromagroups.txt")
x = np.loadtxt(f, delimiter='\t', skiprows=1)
# create np array for data points
data = np.array(x).astype("float")
#data[i][j], i varies the row (chooses the coordinates [pc1, pc2] at row i)
#data[i][j], j varies the column (chooses between pc1 and pc2 respectively 0 or 1)

pc1 = data[:, :-1]
pc2 = data[:,-1]
# create scatter plot
plt.scatter(pc1, pc2, cmap='viridis')
plt.xlim(-6,6)
plt.ylim(-8,6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Wine data PC')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.savefig('wine-data-scatter.png', dpi = 250)
plt.clf()

