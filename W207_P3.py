
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


#os.chdir("\\Users\\skarb\\Desktop\\Github\\W207_P3\\")
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

# project data across N-dimensions
n_comp = 50
pca_mod = PCA(n_components = n_comp)
pca_mod.fit(train_data)

# cummulative ratio of explained variance
varRatio = pca_mod.explained_variance_ratio_
sumVarRatio = np.cumsum(np.concatenate(([0], varRatio)))

# plot the data to visualize
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

# project data to two dimensions
n_comp = 2
pca_mod = PCA(n_components = n_comp)
pca_dat = pca_mod.fit_transform(train_data)

# plot 2-D data with color code (blue=poisonous, red = not poisnous)
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

# setup KMeans model and separate data
for i in [1,16]:
    model = KMeans(n_clusters=i)
    model.fit(pca_dat)
    labels = model.labels_
    centers = model.cluster_centers_
    predicted_cluster = model.predict(pca_dat)
    distanceMax = np.zeros(len(centers))
    
    # calc distance between points to decide furthest point
    for j in range(len(pca_dat)):
        clusters = centers[predicted_cluster[j]]
        distanceX = (pca_dat[j][0]-clusters[0])**2
        distanceY = (pca_dat[j][1]-clusters[1])**2
        total_distance = (distanceX + distanceY)**(.5)
        if total_distance > distanceMax[predicted_cluster[j]]:
            distanceMax[predicted_cluster[j]]=total_distance
    plots = fig.add_subplot(1,2,count)
    
    # plot data points
    plots.scatter(pca_dat[:,0], pca_dat[:,1], c = labels)
    
    # plot circles and center points
    for k in range(len(centers)):
        cir = plt.Circle(centers[k], radius = distanceMax[k], color = "black", fill = False)
        plots.add_patch(cir)
        plots.scatter(model.cluster_centers_[k][0], model.cluster_centers_[k][1], color = "red")
    plt.title(str(i) + " Clusters")
    count+=1
plt.show()

###############################################################################
# Problem 4
###############################################################################

#n_comp = 2
#
## subset the training labels
#pos_labels = np.where(train_labels == 1)
#neg_labels = np.where(train_labels == 0)
#
#pca_mod3 = PCA(n_components = n_comp)
#pca_test_data = pca_mod3.fit_transform(test_data)
#
#scores = []
#score_details = []
#
#
#for gmm_n_comp in range(4):
#    for covar in ["spherical","diag","tied","full"]:
#
#        #positive data class
#        pca_mod1 = PCA(n_components = n_comp)
#        pca_dat1 = pca_mod1.fit_transform(train_data)
#        pos_pca = pca_dat[pos_labels]
#        model1 = GMM(n_components=(gmm_n_comp+1), covariance_type=covar)
#        model1.fit(pos_pca)
#        model1.get_params
#        # negative data class
#        pca_mod2 = PCA(n_components = n_comp)
#        pca_dat2 = pca_mod2.fit_transform(train_data)
#        neg_pca = pca_dat2[neg_labels]
#        model2 = GMM(n_components=(gmm_n_comp+1), covariance_type=covar)
#        model2.fit(neg_pca)


fig = plt.figure(figsize=(12, 30))
count=1

# get positive data
pos_labels = np.where(train_labels == 1)
n_comp = 2
pca_mod = PCA(n_components = n_comp)
pca_dat = pca_mod.fit_transform(train_data)
pos_pca = pca_dat[pos_labels]

for gmm_n_comp in range(4):
    for covar in ["spherical","diag","tied","full"]:

        model = GMM(n_components=gmm_n_comp+1, covariance_type=covar)
        model.fit(pos_pca)
        
        # display predicted scores by the model as a contour plot
        x = np.linspace(-20., 30.)
        y = np.linspace(-20., 40.)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = model.score_samples(XX)
        
        # get from tuple to ndarray
        ZZ = np.empty([2500,])
        for i in range(len(Z[0])):
            ZZ[i] = Z[0][i]
        ZZ = ZZ.reshape(X.shape)
        ZZ = -(ZZ)
        
        #plot contour map
        plots = fig.add_subplot(8,2,count)
        CS = plots.contour(X, Y, ZZ, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
        #CB = plt.colorbar(CS, shrink=0.8, extend='both')
        plots.scatter(pca_dat[:, 0], pca_dat[:, 1])
        plots.axis('tight')
        title = "-log likelihood from GMM; comp = " + str(gmm_n_comp+1) + ", cov = " + covar
        plt.title(title)
        count+=1
plt.show()

###############################################################################
# Problem 5
###############################################################################

# get positive data
pos_labels = np.where(train_labels == 1)
n_comp = 2
pca_mod1 = PCA(n_components = n_comp)
pca_dat1 = pca_mod1.fit_transform(train_data)
pos_pca = pca_dat1[pos_labels]
model1 = GMM(n_components=4, covariance_type="full")
model1.fit(pos_pca)

# get negative data
neg_labels = np.where(train_labels == 0)
n_comp = 2
pca_mod2 = PCA(n_components = n_comp)
pca_dat2 = pca_mod2.fit_transform(train_data)
neg_pca = pca_dat2[neg_labels]
model2 = GMM(n_components=4, covariance_type="full")
model2.fit(neg_pca)

# get 2-d test data
pca_mod3 = PCA(n_components = n_comp)
pca_test_data = pca_mod3.fit_transform(test_data)

# test GMM 
pos_score = model1.score(pca_test_data)
neg_score = model2.score(pca_test_data)

# apply labels to probability results
gmm_results = np.zeros([1124,])
for i in range(pos_score.shape[0]):
    if pos_score[i] > neg_score[i]:
        gmm_results[i] = 1.0
    elif pos_score[i] < neg_score[i]:
        gmm_results[i] = 0.0
    elif pos_score[i] == neg_score[i]:
        print("50/50 situation faced")
        gmm_results[i] = 1.0

score = (metrics.accuracy_score(test_labels, gmm_results))*100

print("Accuracy for 4 comp, full covariance matrix GMM on 2-D data = {0:.2f}%"
      .format(score))

###############################################################################
# Problem 6
###############################################################################

n_comp = 2

# subset the training labels
pos_labels = np.where(train_labels == 1)
neg_labels = np.where(train_labels == 0)

pca_mod3 = PCA(n_components = n_comp)
pca_test_data = pca_mod3.fit_transform(test_data)

scores = []
score_details = []

# iterate through the GMM parameters
for gmm_n_comp in range(4):
    for covar in ["spherical","diag","tied","full"]:

        #positive data class
        pca_mod1 = PCA(n_components = n_comp)
        pca_dat1 = pca_mod1.fit_transform(train_data)
        pos_pca = pca_dat[pos_labels]
        model1 = GMM(n_components=(gmm_n_comp+1), covariance_type=covar)
        model1.fit(pos_pca)
        model1.get_params
        # negative data class
        pca_mod2 = PCA(n_components = n_comp)
        pca_dat2 = pca_mod2.fit_transform(train_data)
        neg_pca = pca_dat2[neg_labels]
        model2 = GMM(n_components=(gmm_n_comp+1), covariance_type=covar)
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

        scores.append(metrics.accuracy_score(test_labels, gmm_results))
        score_details.append([gmm_n_comp, covar])

# look at best paramters    
best_gmm = score_details[scores.index(max(scores))][0]
best_covar = score_details[scores.index(max(scores))][1]
        
print("The highest accuracy achieved was: {0:.2f}%".format(max(scores)*100))
print("This was with {0} PCA components, {1} GMM components, and a {2} covariance type matrix."
      .format(n_comp, best_gmm, best_covar))



