
#%matplotlib inline

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from matplotlib.colors import LogNorm

os.chdir("/Users/Admiral/Desktop/W207_P3/")

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

fig = plt.figure(figsize=(10, 30))
for i in range(16):

    model1 = KMeans(n_clusters=i+1)
    model1.fit(pca_dat)
    labels = model1.labels_
    centers = model1.cluster_centers_
    x_centers = [centers[x][0] for x in range(len(centers))]
    y_centers = [centers[x][1] for x in range(len(centers))]
    predicted_cluster = model1.predict(pca_dat)
    max_dist = np.zeros(len(centers))
    for j in range(len(pca_dat)):
        rel_cluster = centers[predicted_cluster[j]]
        euclid_x2 = (pca_dat[j][0]-rel_cluster[0])**2
        euclid_y2 = (pca_dat[j][1]-rel_cluster[1])**2
        euclid_dist = (euclid_x2 + euclid_y2)**(.5)
        #print euclid_dist #why the hell is this always 1
        if euclid_dist > max_dist[predicted_cluster[j]]:
            max_dist[predicted_cluster[j]]=euclid_dist
    sub_plot = fig.add_subplot(8,2,i+1)
    sub_plot.scatter(pca_dat[:,0], pca_dat[:,1], c = labels)
    title_i = str(i+1) + " Clusters"
    for k in range(len(centers)):
        cir = plt.Circle(centers[k], radius = max_dist[k], color = 'g', fill = False)
        sub_plot.add_patch(cir)
    plt.title(title_i)
plt.show()

###############################################################################
# Problem 4
###############################################################################

model = GMM(n_components=2, covariance_type="full")
model.fit(pca_dat)




# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = model.score_samples(XX)
Z = Z.reshape(X.shape)
type(XX)


model.score_samples(XX)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(pca_dat[:, 0], pca_dat[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
















