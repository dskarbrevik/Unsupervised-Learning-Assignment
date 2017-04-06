
%matplotlib inline

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from matplotlib.colors import LogNorm

os.chdir("C:\\Users\\skarb\\Desktop\\Github\\W207_P3\\")

feature_names = []
with open('mushroom.map') as fmap:
    for line in fmap:
        [index, name, junk] = line.split()
        feature_names.append(name)

print ('Loaded feature names:', len(feature_names))

X, Y = [], []

with open('mushroom.data') as fdata:
    for line in fdata:
        items = line.split()
        Y.append(int(items.pop(0)))
        x = np.zeros(len(feature_names))
        for item in items:
            feature = int(item.split(':')[0])
            x[feature] = 1
        X.append(x)

# Convert these lists to numpy arrays.
X = np.array(X)
Y = np.array(Y)

# Split into train and test data.
train_data, train_labels = X[:7000], Y[:7000]
test_data, test_labels = X[7000:], Y[7000:]

# Check that the shapes look right.
print (train_data.shape, test_data.shape)


###############################################################################
# Problem 1
###############################################################################

n_comp = 50
pca_mod = PCA(n_components = n_comp)
pca_mod.fit(train_data)

varRatio = pca_mod.explained_variance_ratio_
sumVarRatio = np.cumsum(np.concatenate(([0], varRatio)))

plt.plot(range(n_comp+1),sumVarRatio)
plt.xlabel('number components')
plt.ylabel('sum of explained variance ratios')
plt.ylim([0,1])
plt.title("PCA: Explained Variance in first 50 components", 
          fontsize=12, fontweight="bold")
plt.show()


###############################################################################
# Problem 2
###############################################################################


n_comp = 2
pca_mod = PCA(n_components = n_comp)
pca_dat = pca_mod.fit_transform(train_data)

plt.figure(figsize=(10,10))
colormap = {0: 'red', 1: 'blue'}
colors = [colormap[x] for x in train_labels]
plt.scatter(pca_dat[:,0], pca_dat[:,1], color=colors, edgecolors="black")
plt.title("Mushroom Data in 2 Dimensions", fontsize=14, fontweight="bold")
plt.show()

###############################################################################
# Problem 3
###############################################################################

km = KMeans (n_clusters=16, init='k-means++')
clstrs = km.fit (pca_dat)
print (clstrs.cluster_centers_.shape)
print (clstrs.cluster_centers_)






























