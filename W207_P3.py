
#%matplotlib inline

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from matplotlib.colors import LogNorm
from sklearn.grid_search import GridSearchCV


os.chdir("\\Users\\skarb\\Desktop\\Github\\W207_P3\\")

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

fig = plt.figure(figsize=(10, 5))
count = 1
for i in [1,16]:

    model1 = KMeans(n_clusters=i)
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
    sub_plot = fig.add_subplot(1,2,count)
    sub_plot.scatter(pca_dat[:,0], pca_dat[:,1], c = labels)
    title_i = str(i) + " Clusters"
    for k in range(len(centers)):
        cir = plt.Circle(centers[k], radius = max_dist[k], color = 'g', fill = False)
        sub_plot.add_patch(cir)
        sub_plot.scatter(model1.cluster_centers_[k][0], model1.cluster_centers_[k][1], color = "black")
    plt.title(title_i)
    count+=1
plt.show()

###############################################################################
# Problem 4
###############################################################################

pos_labels = np.where(train_labels == 1)
n_comp = 2
pca_mod = PCA(n_components = n_comp)
pca_dat = pca_mod.fit_transform(train_data)
pos_pca = pca_dat[pos_labels]
model = GMM(n_components=4, covariance_type="spherical")
model.fit(pos_pca)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = model.score_samples(XX)
#Z = Z.reshape(X.shape)

test = np.empty([2500,])
for i in range(len(Z[0])):
    test[i] = Z[0][i]
test = test.reshape(X.shape)
test = -(test)
fig = plt.figure(figsize=(10, 5))
CS = plt.contour(X, Y, test, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(pca_dat[:, 0], pca_dat[:, 1])
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
#plt.axes.set_xlim([-10,10])
#plt.axes.set_ylim([-10,10])
plt.show()

###############################################################################
# Problem 5
###############################################################################

pos_labels = np.where(train_labels == 1)
n_comp = 2
pca_mod1 = PCA(n_components = n_comp)
pca_dat1 = pca_mod1.fit_transform(train_data)
pos_pca = pca_dat[pos_labels]
model1 = GMM(n_components=4, covariance_type="full")
model1.fit(pos_pca)

neg_labels = np.where(train_labels == 0)
n_comp = 2
pca_mod2 = PCA(n_components = n_comp)
pca_dat2 = pca_mod2.fit_transform(train_data)
neg_pca = pca_dat2[neg_labels]
model2 = GMM(n_components=4, covariance_type="full")
model2.fit(neg_pca)


pca_mod3 = PCA(n_components = n_comp)
pca_test_data = pca_mod3.fit_transform(test_data)

pos_score = model1.score(pca_test_data)
neg_score = model2.score(pca_test_data)


gmm_results = np.zeros([1124,])
for i in range(pos_score.shape[0]):
    if pos_score[i] > neg_score[i]:
        gmm_results[i] = 1.0
    elif pos_score[i] < neg_score[i]:
        gmm_results[i] = 0.0
    elif pos_score[i] == neg_score[i]:
        print("50/50 situation faced")
        gmm_results[i] = 1.0

metrics.accuracy_score(test_labels, gmm_results)

###############################################################################
# Problem 6
###############################################################################

n_comp = 2

# subset the training labels
pos_labels = np.where(train_labels == 1)
neg_labels = np.where(train_labels == 0)

pca_mod3 = PCA(n_components = n_comp)
pca_test_data = pca_mod3.fit_transform(test_data)

for covar in ["spherical","diag","tied","full"]:

    #positive data class
    pca_mod1 = PCA(n_components = n_comp)
    pca_dat1 = pca_mod1.fit_transform(train_data)
    pos_pca = pca_dat[pos_labels]
    model1 = GMM(n_components=4, covariance_type=covar)
    model1.fit(pos_pca)
    
    # negative data class
    pca_mod2 = PCA(n_components = n_comp)
    pca_dat2 = pca_mod2.fit_transform(train_data)
    neg_pca = pca_dat2[neg_labels]
    model2 = GMM(n_components=4, covariance_type=covar)
    model2.fit(neg_pca)




pos_score = model1.score(pca_test_data)
neg_score = model2.score(pca_test_data)


gmm_results = np.zeros([1124,])
for i in range(pos_score.shape[0]):
    if pos_score[i] > neg_score[i]:
        gmm_results[i] = 1.0
    elif pos_score[i] < neg_score[i]:
        gmm_results[i] = 0.0
    elif pos_score[i] == neg_score[i]:
        print("50/50 situation faced")
        gmm_results[i] = 1.0

metrics.accuracy_score(test_labels, gmm_results)














