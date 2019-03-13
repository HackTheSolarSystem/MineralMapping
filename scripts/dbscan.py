import argparse
from collections import Counter
import json
import time

print("Importing dependencies...")
import pandas as pd
from pathlib import Path
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from lib import load_images, load_standards_df, get_standards_characteristics


def get_predicted_weights(obj_df, standards_characteristics, calculate_unknown=True):
    # create empty data frame to fill with predicted percent weights
    percent_weight_pred = pd.DataFrame(columns=obj_df.columns)
    # reorder coefficients to match order of columns in mineral_standards
    coeffs = pd.DataFrame(index=['coeff'], columns=obj_df.columns[:-1])
    for element, characteristics in standards_characteristics.items():
        coeffs[element] = characteristics["coef"]
    coeffs_mat = np.repeat(np.reciprocal(coeffs.values), len(obj_df), axis=0)

    # apply coefficients from linear regression to pixel intensities from standard
    percent_weight_pred = coeffs_mat * obj_df
    # if the predicted percent weight is over 100, set it to 100
    percent_weight_pred[percent_weight_pred > 100] = 100
    # replace NaN with 0
    percent_weight_pred.fillna(0, inplace=True)

    if calculate_unknown:
        # add column for unknown weight percent
        percent_weight_pred['unknown'] = np.ones(len(percent_weight_pred)) - \
                percent_weight_pred.sum(axis=1)
        percent_weight_pred['unknown'] = np.maximum(percent_weight_pred['unknown'], np.zeros(percent_weight_pred['unknown'].shape))

    return percent_weight_pred


def plot_pca(x, labels):
    # create a PCA for visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=components, columns=['name1', 'name2'])
    principal_df['cluster'] = labels

    # plot PCA showing the different clusters
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Principal Component 1', fontsize=15)
    ax1.set_ylabel('Principal Component 2', fontsize=15)
    ax1.set_title('2 component PCA', fontsize=20)
    for c in np.unique(labels):
        x_pts = principal_df['name1'][principal_df['cluster'] == c]
        y_pts = principal_df['name2'][principal_df['cluster'] == c]
        ax1.scatter(x_pts, y_pts, label=c, s=40)
    ax1.legend()
    plt.savefig("pca.png")


def main(meteorite_dir, standards_dir, bits, epsilon, min_samples, num_components, disable_unknown=False):
    # Load meteorite intensity dataframe
    print("Loading meteorite images...")
    meteorite_df, shape = load_images(meteorite_dir, bits)

    # Get standards characteristics for weight to intensity coefficients
    print("Loading standards characteristics...")
    standards_characteristics = get_standards_characteristics(standards_dir, bits)

    # Load DataFrame of predicted weights in standards
    print("Getting meteorite predicted weight percents...")
    df = get_predicted_weights(meteorite_df, standards_characteristics, calculate_unknown=not disable_unknown)
    x = df.values

    # Fit using DBSCAN
    #print("Running DBSCAN clustering...")
    #start = time.time()
    #db = DBSCAN(eps=epsilon, min_samples=min_samples, n_jobs=-1).fit(x)
    #end = time.time()
    #print(f"Clustering ran in {end-start} seconds")
    #labels = db.labels_

    # Fit using GMM
    print("Running Gaussian Mixture clustering...")
    start = time.time()
    labels = GaussianMixture(covariance_type="diag", n_components=num_components).fit_predict(x)
    end = time.time()
    print(f"Clustering ran in {end-start} seconds")
    print("LABELS ARE:", labels)

    # Get clustering stats
    clusters, counts = np.unique(labels, return_counts=True)
    cluster_counts = {
        str(cluster): int(count)
        for cluster, count in zip(clusters, counts)
    }
    #n_noise = cluster_counts["-1"]
    print(f"Estimated number of clusters: {len(clusters)}")
    #print(f"Estimated number of noise points: {n_noise}/{len(x)} ({n_noise*100/len(x):.2f}%)")
    print(f"Counts per cluster:\n{json.dumps(cluster_counts, indent=4)}")

    # Plot PCA
    plot_pca(x, labels)

    # Show image overlayed on actual meteorite
    fig = plt.figure(figsize=(14,9))
    img_arr = np.reshape(labels, shape) + 1
    plt.imshow(img_arr, cmap="tab10")
    plt.suptitle(f"Epsilon: {epsilon}, Min Samples: {min_samples}")
    plt.savefig("obj.png")


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
    parser.add_argument("--meteorite-dir", type=valid_dir, default=Path("."),
                        help="path to directory containing mineral images")
    parser.add_argument("--standards-dir", type=valid_dir, default=Path("."),
                        help="path to directory containing standards")
    parser.add_argument("--bits", type=int, choices=[8, 32], default=32,
                        help="image bit-depth to use")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="epsilon radius to use for DBSCAN")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="minimum samples per cluster to use for DBSCAN")
    parser.add_argument("--num-components", type=int, default=10,
                        help="number of components to use for GMM")
    parser.add_argument("--disable-unknown", action="store_true",
                        help="don't calculate unknown weight percent column")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
