import os
import sys
from sklearn.preprocessing import StandardScaler

# Append path to use modules outside pycharm environment, e.g. terminal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
import matplotlib.pyplot as plt

from dataset import dataset_config
import features
from helper_functions import pre_processing


def calculate_average(X_data):
    return np.matrix(X_data).mean(0)


def calculate_standard_deviation(X_data):
    return np.matrix(X_data).std(0)


def plot(data, tot_features, avg):
    """
    Plots data
    :param data: data to plot
    :param tot_features: all features you have chosen
    :param avg: flag if it is standard deviation or average data we are plotting
    :return:
    """
    # Finds the length of the features
    len_tot_features = np.arange(len(tot_features))

    # Various plotting parameters
    plt.figure(figsize=(40, 40))
    plt.bar(len_tot_features, data)
    plt.xticks(len_tot_features, tot_features)
    if avg:
        plt.title("Average of the chosen features for all your songs!")
    else:
        plt.title("Standard Deviation of the chosen features for all your songs!")

    for i, v in enumerate(data):
        plt.text(i - 0.2, v + 0.3, "%.2f" % v, color='black', fontweight='bold')

    plt.show()


def run():

    # Get all the features you chose
    tot_features = features.get_features()

    # Get all the likeable songs you chose
    likeable_songs = dataset_config.likeable_songs

    # Get data on the correct format so we can do analysis on it
    X_data, X_songname, X_artist = pre_processing.pre_process_cluster_data(likeable_songs)

    # X_data = StandardScaler().fit_transform(X_data)
    # Find the average of your data
    averages = np.squeeze(np.asarray(calculate_average(X_data)))

    # Plot the average data
    plot(averages, tot_features, True)

    # Calculate the standard deviation of your data
    standard_deviation = np.squeeze(np.asarray(calculate_standard_deviation(X_data)))

    # Plot the standard deviation data
    plot(standard_deviation, tot_features, False)

    # Calculate "funness" of your songs
    funness = 0
    for i, v in enumerate(tot_features):
        if v == 'loudness':
            funness += averages[i]

        elif v == 'tempo':
            funness += averages[i]

        elif v == 'energy':
            funness += averages[i] * 100

        elif v == 'danceability':
            funness += averages[i] * 100

    print("Your funness score is %f. Compare it to the people around you. The higher the score, the more fun your music is!" % funness)

run()
