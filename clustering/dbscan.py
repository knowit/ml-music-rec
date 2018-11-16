import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def cluster_with_dbscan(data):
    """
    Runs the dbscan clustering algorithm.
    :param data: All features gathered from the dataset
    """

    # TODO: Change parameters from here
    db = DBSCAN(eps=0.34, min_samples=15, metric='euclidean',
                metric_params=None, algorithm='auto', leaf_size=30,
                p=None, n_jobs=1).fit(data)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return n_clusters_, labels, core_samples_mask


def print_dbscan(labels, reduced_data, n_clusters_, core_samples_mask, X_songname, X_artist):
    """
    Visualize the dbscan clustering algorithm.
    You do not need to do anything in here.
    """

    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111)

    sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

        xy = reduced_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=2)

    plt.title('DBSCAN: Estimated number of clusters: %d' % n_clusters_)

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

