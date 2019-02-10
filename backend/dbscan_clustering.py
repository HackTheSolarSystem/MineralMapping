from sklearn.cluster import DBSCAN as dbscan
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from pathlib import Path
from skimage.io import imread, imshow
import numpy.ma as ma
from collections import Counter

standards = pd.read_csv('challenge_data/mineral_standards.csv')
df = pd.read_csv('challenge_data/predicted_percentweight_standard.csv')
df = df.fillna(0)

x = df.values
db = dbscan(eps = 10, min_samples = 20).fit(x)

core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for n_noise_
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)

pca = PCA(n_components = 2)
principleComponents = pca.fit_transform(x)
principalDf= pd.DataFrame(data = principleComponents, columns = ['name1', 'name2'])
finalDf = pd.concat([principalDf, pd.Series(labels)], axis = 1)
final_minerals = pd.concat([principalDf, standards['mineral']], axis = 1)

mn = list(set(standards['mineral'].values))
color_dict = {}
for i, val in enumerate(mn):
    color_dict[val] = i
standards['mineral'].map(color_dict)

labels
finalDf.head()

finalDf['cluster'] = labels
finalDf.head()

range(len(labels))
np.unique(labels)


fig = plt.figure(figsize = (14,9))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Principal Component 1', fontsize = 15)
ax1.set_ylabel('Principal Component 2', fontsize = 15)
ax1.set_title('2 component PCA', fontsize = 20)
for c in np.unique(labels):
    ax1.scatter(finalDf['name1'][finalDf['cluster'] == c], finalDf['name2'][finalDf['cluster'] == c], label = c, s=40)
ax1.legend()
plt.savefig('./images/pca_predicteddata.png')




finalDf['mineral'] = standards['mineral']
finalDf.head()

np.unique(finalDf['mineral'])
fig = plt.figure(figsize = (14,9))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Principal Component 1', fontsize = 15)
ax1.set_ylabel('Principal Component 2', fontsize = 15)
ax1.set_title('2 component PCA', fontsize = 20)
# ax = fig.add_subplot(1,1,1)
for m in np.unique(finalDf['mineral']):
    ax1.scatter(finalDf['name1'][finalDf['mineral'] == m], finalDf['name2'][finalDf['mineral'] == m], label = m, s = 40)
ax1.legend()
plt.savefig('./images/pca_realdata.png')


df_obj2 = pd.read_csv('challenge_data/predicted_percentweight_obj2.csv')
df_obj2 = df_obj2.fillna(0)
df_obj2.drop('Unnamed: 0', axis = 1, inplace = True)

df_obj2.head()


x = df_obj2.values
x[x < 5] = 0
db = dbscan(eps = 3, min_samples = 10).fit(x)

core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
Counter(labels)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
100*2000/len(df_obj2)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        #Black used for n_noise_
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)

pca = PCA(n_components = 2)
principleComponents = pca.fit_transform(x)
principalDf= pd.DataFrame(data = principleComponents, columns = ['name1', 'name2'])
finalDf = pd.concat([principalDf, pd.Series(labels)], axis = 1)

finalDf.head()


finalDf['cluster'] = labels
np.unique(labels)

fig = plt.figure(figsize = (14,9))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Principal Component 1', fontsize = 15)
ax1.set_ylabel('Principal Component 2', fontsize = 15)
ax1.set_title('2 component PCA', fontsize = 20)
for c in np.unique(labels):
    ax1.scatter(finalDf['name1'][finalDf['cluster'] == c], finalDf['name2'][finalDf['cluster'] == c], label = c, s=40)
ax1.legend()

finalDf.shape
finalDf.head()

root = Path("/Users/hellenfellows/OneDrive - AMNH/BridgeUp/HackathonRepo/MineralMapping/challenge_data/")
image_path = root / "dataset_1_opaques"
mask = imread(root / "dataset_1_opaques/obj2_mask.tif")
mask.sum()
mask[mask > 0] = 1
finalDf['cluster'].dtypes
mask = mask.astype('int64')
obj2_cluster = mask.copy()
obj2_cluster[obj2_cluster == 1] = finalDf['cluster'] + 2

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

~mask.astype(bool)

obj2_cluster_masked = ma.masked_array(obj2_cluster, ~mask.astype(bool))
obj2_cluster_masked

len(set(labels))

fig = plt.figure(figsize = (14,9))
plt.imshow(obj2_cluster_masked, cmap = 'tab20')
#plt.imshow(obj2_cluster_masked, cmap = discrete_cmap(len(set(labels)), base_cmap='gnuplot2'))


df_obj2.head()
df_obj2['cluster'] = finalDf['cluster']
df_obj2.head()
df_obj2.to_csv("df_obj2_cluster.csv")
