import os
import sys

# Append path to use modules outside pycharm environment, e.g. terminal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from dataset import dataset_config
from helper_functions import pre_processing


import numpy as np
from assignment1 import dbscan, agglomerative, k_means
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_agglomerative(X_data, reduced_data, X_songname, X_artist):
    """
    Runs the agglomerative clustering algorithm.
    :param X_data: All features gathered from the dataset in form of (n_features, n_songs)
    :param reduced_data: The result of the PCA-reduced data on the original data.
    :param X_songname: Corresponding songname
    :param X_artist: Corresponding artist
    """

    # Clustering based on the original data. Gives more precise results, but can't be visualized
    # agglomerative_y_labels = agglomerative.cluster_with_agglomerative(X_data)
    # print(agglomerative_y_labels)

    # Clustering on PCA-reduced data. Can be visualized, but gives more unprecise results
    X_labels = agglomerative.cluster_with_agglomerative(reduced_data)
    agglomerative.print_agglomerative(reduced_data, X_labels, X_songname, X_artist)

def run_kmeans(X_data, reduced_data, X_songname, X_artist):
    """
    Runs the kmeans clustering algorithm.
    :param X_data: All features gathered from the dataset in form of (n_features, n_songs)
    :param reduced_data: The result of the PCA-reduced data on the original data.
    :param X_songname: Corresponding songname
    :param X_artist: Corresponding artist
    """

    # Clustering based on the original data. Gives more precise results, but can't be visualized
    # kmeans_y_labels = k_means.cluster_with_kmeans_on_original_data(X_data)
    # print(kmeans_y_labels)

    # Clustering on PCA-reduced data. Can be visualized, but gives more unprecise results
    Z, kmeans, x_min, x_max, y_min, y_max, xx, yy, X_labels = k_means.cluster_with_kmeans_on_reduced_data(reduced_data)
    k_means.print_kmeans(Z, xx, yy, reduced_data, X_songname, X_artist, X_labels, kmeans, x_min, x_max, y_min, y_max)

def run_dbscan(X_data, reduced_data, X_songname, X_artist):
    """
    Runs the dbscan clustering algorithm.
    :param X_data: All features gathered from the dataset in form of (n_features, n_songs)
    :param reduced_data: The result of the PCA-reduced data on the original data.
    :param X_songname: Corresponding songname
    :param X_artist: Corresponding artist
    """

    # Clustering based on the original data. Gives more precise results, but can't be visualized
    n_clusters_, labels, core_samples_mask = dbscan.cluster_with_dbscan(X_data)
    # print(labels)

    # Clustering on PCA-reduced data. Can be visualized, but gives more unprecise results
    n_clusters_, labels, core_samples_mask = dbscan.cluster_with_dbscan(reduced_data)
    dbscan.print_dbscan(labels, reduced_data, n_clusters_, core_samples_mask, X_songname, X_artist)


def main():
    """
    Used to run the algorithms.

    """

    # For printing to console
    np.set_printoptions(threshold=np.inf)

    # Getting all audio features from my likeable songs
    audio_features = dataset_config.likeable_songs

    # Remove duplicates
    audio_features = pre_processing.remove_duplicates(audio_features)

    # X_data is the features you have chosen.
    X_data, X_songname, X_artist = pre_processing.pre_process_cluster_data(audio_features)

    # Simply normalize all data
    X_data = StandardScaler().fit_transform(X_data)

    # X_data's dimensions are reduced to n_components.
    reduced_data = PCA(n_components=2).fit_transform(X_data)

    # Run Agglomerative clustering
    run_agglomerative(X_data=X_data, reduced_data=reduced_data,
                      X_songname=X_songname, X_artist=X_artist)

    # # Run KMeans clustering
    # run_kmeans(X_data=X_data, reduced_data=reduced_data,
    #            X_songname=X_songname, X_artist=X_artist)

    # Run DBSCAN clustering
    # run_dbscan(X_data=X_data, reduced_data=reduced_data,
    #            X_songname=X_songname, X_artist=X_artist)




main()

