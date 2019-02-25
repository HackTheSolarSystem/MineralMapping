import argparse
from collections import Counter
import time

import pandas as pd
from pathlib import Path
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from lib import load_standards_df, get_standards_characteristics


def todo():
    # change NA values to 0
    df = df.fillna(0)
    x = df.values
    # run DBSCAN Clustering
    # Need to play with the eps and min_samples parameters to get reasonable results
    # Future solution should loop through various parameter values to find reasonable obj1_minerals
    # In general, to consolidate to fewer clusters, increase the parameters, and vicee versa
    db = DBSCAN(eps = 10, min_samples = 20).fit(x)

    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # See number of clusters and number of unclustered (noise) points
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # create a PCA for visualization
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

    # add a cluster column
    finalDf['cluster'] = labels

    # plot PCA showing the different clusters
    fig = plt.figure(figsize = (14,9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Principal Component 1', fontsize = 15)
    ax1.set_ylabel('Principal Component 2', fontsize = 15)
    ax1.set_title('2 component PCA', fontsize = 20)
    for c in np.unique(labels):
        ax1.scatter(finalDf['name1'][finalDf['cluster'] == c], finalDf['name2'][finalDf['cluster'] == c], label = c, s=40)
    ax1.legend()
    plt.savefig('./images/pca_predicteddata.png')

    # add a mineral column
    finalDf['mineral'] = standards['mineral']

    # plot PCA showing the location of the actual minerals
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

    # Clustering on object 1
    # read csv of predicteed percent weights for object 1
    df_obj1 = pd.read_csv('challenge_data/predicted_percentweight_obj1.csv')
    df_obj1 = df_obj1.fillna(0)
    df_obj1.drop('Unnamed: 0', axis = 1, inplace = True)
    x = df_obj1.values
    # Set cells with a percent weight lower than 5 to 0
    # Could consider what a realistic cleaning threshold is
    x[x < 5] = 0
    # Run dbscan on object 1
    # May need to adjust eps and min_samples for different images
    db = DBSCAN(eps = 3, min_samples = 15).fit(x)

    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # see number of clusters and noise
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
    finalDf['cluster'] = labels

    # visualize clusters in PCA
    fig = plt.figure(figsize = (14,9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Principal Component 1', fontsize = 15)
    ax1.set_ylabel('Principal Component 2', fontsize = 15)
    ax1.set_title('2 component PCA', fontsize = 20)
    for c in np.unique(labels):
        ax1.scatter(finalDf['name1'][finalDf['cluster'] == c], finalDf['name2'][finalDf['cluster'] == c], label = c, s=40)
    ax1.legend()


    root = Path("/Users/hellenfellows/OneDrive - AMNH/BridgeUp/HackathonRepo/MineralMapping/challenge_data/")
    image_path = root / "dataset_1_opaques"
    mask = imread(root / "dataset_1_opaques/obj1_mask.tif")
    mask.sum()
    # make mask binary
    mask[mask > 0] = 1
    mask = mask.astype('int64')
    obj1_cluster = mask.copy()
    # adjust cluster numbers so that nothing is <= 0
    obj1_cluster[obj1_cluster == 1] = finalDf['cluster'] + 2

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

    # mask for object 1
    obj1_cluster_masked = ma.masked_array(obj1_cluster, ~mask.astype(bool))

    # show location of clusters on actual image of object 1
    fig = plt.figure(figsize = (14,9))
    plt.imshow(obj1_cluster_masked, cmap = 'tab10')
    plt.savefig('./images/obj1_cluster_pred.png')

    df_obj1['cluster'] = finalDf['cluster']
    df_obj1.to_csv("df_obj1_cluster.csv")


    # Clustering on object 2

    df_obj2 = pd.read_csv('challenge_data/predicted_percentweight_obj2.csv')
    df_obj2 = df_obj2.fillna(0)
    df_obj2.drop('Unnamed: 0', axis = 1, inplace = True)
    x = df_obj2.values
    x[x < 5] = 0
    # Run dbscan on object 2
    db = DBSCAN(eps = 3, min_samples = 15).fit(x)

    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    pca = PCA(n_components = 2)
    principleComponents = pca.fit_transform(x)
    principalDf= pd.DataFrame(data = principleComponents, columns = ['name1', 'name2'])
    finalDf = pd.concat([principalDf, pd.Series(labels)], axis = 1)
    finalDf['cluster'] = labels

    fig = plt.figure(figsize = (14,9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Principal Component 1', fontsize = 15)
    ax1.set_ylabel('Principal Component 2', fontsize = 15)
    ax1.set_title('2 component PCA', fontsize = 20)
    for c in np.unique(labels):
        ax1.scatter(finalDf['name1'][finalDf['cluster'] == c], finalDf['name2'][finalDf['cluster'] == c], label = c, s=40)
    ax1.legend()


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


    obj2_cluster_masked = ma.masked_array(obj2_cluster, ~mask.astype(bool))


    fig = plt.figure(figsize = (14,9))
    plt.imshow(obj2_cluster_masked, cmap = 'tab10')
    plt.legend(obj2_cluster_masked)
    plt.savefig('./images/obj2_cluster_pred.png')

    # create csv with predicted weights and clusters
    df_obj2['cluster'] = finalDf['cluster']
    df_obj2.to_csv("df_obj2_cluster.csv")


def get_predicted_weights(standards_df, standards_characteristics):
    # create empty data frame to fill with predicted percent weights
    percent_weight_pred = pd.DataFrame(columns=standards_df.columns)
    # reorder coefficients to match order of columns in mineral_standards
    coeffs = pd.DataFrame(index=['coeff'], columns=standards_df.columns[:-1])
    for element, characteristics in standards_characteristics.items():
        coeffs[element] = characteristics["coef"]
    coeffs_mat = np.repeat(np.reciprocal(coeffs.values), len(standards_df), axis=0)

    # apply coefficients from linear regression to pixel intensities from standard
    percent_weight_pred = coeffs_mat * standards_df.drop(['mineral'], axis=1)
    # if the predicted percent weight is over 100, set it to 100
    percent_weight_pred[percent_weight_pred > 100] = 100
    # replace NaN with 0
    percent_weight_pred.fillna(0, inplace=True)

    # add a mineral column
    percent_weight_pred['mineral'] = standards_df['mineral']

    return percent_weight_pred


def main(standards_dir, bits, epsilon):
    # read the csv of mineral standards
    standards_df = load_standards_df(standards_dir, bits)

    # Get standards characteristics for weight to intensity coefficients
    standards_characteristics = get_standards_characteristics(standards_dir, bits)

    # Load DataFrame of predicted weights in standards
    df = get_predicted_weights(standards_df, standards_characteristics)
    x = df.drop(columns="mineral").values

    # Fit using DBSCAN
    print("Running DBSCAN clustering...")
    start = time.time()
    db = DBSCAN(eps=0.1, min_samples=20).fit(x)
    end = time.time()
    print(f"Clustering ran in {end-start} seconds")

    # Plot clustering results
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # See number of clusters and number of unclustered (noise) points
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # create a PCA for visualization
    pca = PCA(n_components = 2)
    principleComponents = pca.fit_transform(x)
    principalDf= pd.DataFrame(data = principleComponents, columns = ['name1', 'name2'])
    finalDf = pd.concat([principalDf, pd.Series(labels)], axis = 1)
    final_minerals = pd.concat([principalDf, standards_df['mineral']], axis = 1)

    mn = list(set(standards_df['mineral'].values))
    color_dict = {}
    for i, val in enumerate(mn):
        color_dict[val] = i
    standards_df['mineral'].map(color_dict)

    # add a cluster column
    finalDf['cluster'] = labels

    # plot PCA showing the different clusters
    fig = plt.figure(figsize = (14,9))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('Principal Component 1', fontsize = 15)
    ax1.set_ylabel('Principal Component 2', fontsize = 15)
    ax1.set_title('2 component PCA', fontsize = 20)
    for c in np.unique(labels):
        ax1.scatter(finalDf['name1'][finalDf['cluster'] == c], finalDf['name2'][finalDf['cluster'] == c], label = c, s=40)
    ax1.legend()
    plt.show()


def parse_args():
    def valid_dir(path_str):
        p = Path(path_str)
        if not p.exists():
            raise argparse.ArgumentTypeError(f"Could not find path {path_str}")
        if not p.is_dir():
            raise argparse.ArgumentTypeError(f"Path {path_str} is not a directory")
        return p

    description = "Predict mineral content of a meteorite given spectrometer " \
                  "imagery via DBSCAN cluster inference."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--standards-dir", type=valid_dir, default=Path("."),
                        help="path to directory containing standards")
    parser.add_argument("--bits", type=int, choices=[8, 32], default=32,
                        help="image bit-depth to use")
    parser.add_argument("--epsilon", type=float, default=10.,
                        help="epsilon radius to use for DBSCAN")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
