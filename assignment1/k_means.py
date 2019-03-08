import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_with_kmeans_on_original_data(X_data):
    """
    Runs the kmeans clustering algorithm, but only on the original data
    :param X_data: All features gathered from the dataset
    """

    # TODO: Change parameters for the original data kmeans algorithm here
    k_means = KMeans(n_clusters=1, init='k-means++', n_init=999,
                     max_iter=300, tol=0.0001, precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True, n_jobs=1,
                     algorithm='auto')

    k_means.fit(X=X_data, y=None)

    return k_means.labels_


def cluster_with_kmeans_on_reduced_data(reduced_data):
    """
    Runs the kmeans clustering algorithm on the reduced data.
    :param reduced_data: All features gathered from the dataset
    """

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 0.5, reduced_data[:, 0].max() + 0.5
    y_min, y_max = reduced_data[:, 1].min() - 0.5, reduced_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # TODO: Change parameters for the reduced data kmeans algorithm here
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10,
                    max_iter=300, tol=0.0001, precompute_distances='auto',
                    verbose=0, random_state=None, copy_x=True, n_jobs=1,
                    algorithm='auto')

    kmeans.fit(X=reduced_data, y=None)

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    return Z.reshape(xx.shape), kmeans, x_min, x_max, y_min, y_max, xx, yy, kmeans.labels_


def print_kmeans(Z, xx, yy, reduced_data, X_songname, X_artist, X_labels,  kmeans, x_min, x_max, y_min, y_max):
    """
    Visualize the kmeans clustering algorithm.
    You do not need to do anything in here.
    """

    fig = plt.figure(figsize=(40, 40))

    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    ax = fig.add_subplot(111)
    sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=8, c=X_labels)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.title('K-means clustering on your spotify playlist (PCA-reduced data)\n'
              'Centroids are marked with white cross')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    annot = plt.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="w"),
                         arrowprops=dict(arrowstyle="->"))

    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos

        artist = ""
        for a in X_artist[ind['ind'][0]]:
            artist += a

        songname = X_songname[ind['ind'][0]]

        annot.set_text(artist + ": " + songname)
        annot.get_bbox_patch().set_facecolor((1, 1, 1))
        annot.get_bbox_patch().set_alpha(1)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()
