import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


def cluster_with_agglomerative(data):
    """
    Runs the hierarchical agglomerative clustering algorithm.
    :param data: All features gathered from the dataset
    """

    # TODO: Change parameters from here
    agglomerative = AgglomerativeClustering(n_clusters=1, affinity='euclidean', memory=None,
                                            connectivity=None, compute_full_tree='auto',
                                            linkage='ward', pooling_func=np.mean)

    agglomerative.fit(X=data, y=None)

    return agglomerative.labels_

def print_agglomerative(reduced_data, X_labels, X_songname, X_artist):
    """
    Visualize the hierarchical agglomerative clustering algorithm.
    You do not need to do anything in here.
    """

    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111)

    sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                    c=X_labels, s=8)

    ax.set_title('Agglomerative Clustering')

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
    plt.colorbar(sc)
    plt.show()
