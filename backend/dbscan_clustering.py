from sklearn.cluster import DBSCAN as dbscan
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Principal Component 1', fontsize = 15)
ax1.set_ylabel('Principal Component 2', fontsize = 15)
ax1.set_title('2 component PCA', fontsize = 20)
ax1.scatter(finalDf['name1'], finalDf['name2'], c = labels)
plt.savefig('pca_predicteddata.png')
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(finalDf['name1'], finalDf['name2'], c = standards['mineral'].map(color_dict).values)
plt.savefig('pca_realdata.png')



mn = list(set(standards['mineral'].values))
color_dict = {}
for i, val in enumerate(mn):
    color_dict[val] = i
standards['mineral'].map(color_dict)

ax.scatter(finalDf['name1'], finalDf['name2'], c = labels)
ax.scatter(finalDf['name1'], finalDf['name2'], c = standards['mineral'].map(color_dict).values)




