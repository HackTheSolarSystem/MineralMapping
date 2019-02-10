from sklearn.cluster import DBSCAN as dbscan
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


standards = pd.read_csv('challenge_data/mineral_standards.csv')
df = pd.read_csv('challenge_data/predicted_percentweight_standard.csv')
df = df.fillna(0)

x = df.values
start = time.time()
db = dbscan(eps = 10, min_samples = 20).fit(x)
end = time.time()
print(start - end)

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

c = 0
finalDf['name1'][finalDf['cluster'] == c]

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
