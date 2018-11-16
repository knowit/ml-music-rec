from features import get_features
import numpy as np


def label_songs(X, label):
    """

    :param X: Input features samples
    :param label: 1 for likeable songs and 0 for annoying songs
    :return:
    """
    return [label for _ in X]


def filter_features(tracks):
    """

    :param tracks: List of track dictionaries containing audio features
    :return: List X containing feature vectors for each sample, and list Y with labels 0 or 1
    """
    features = get_features()

    X = []

    for track_obj in tracks:
        X.append([track_obj['audio_features'][b] for b in track_obj['audio_features']
                       if not isinstance(track_obj['audio_features'][b], str) and b in features])

    return X


def split_dataset(data, labels, validation_proportion, test_proportion):
    """

    :param data:
    :param labels:
    :param validation_proportion: Proportion of data to use for validation. Value range: [0, 1]. 1 means 100%
    :param test_proportion: Proportion of data to use for test. Value range: [0, 1]. 1 means 100%
    :return:
    """
    np.random.seed(4)
    # shuffle and split the data into a training set, validation set and test set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    nb_validation_samples = int(validation_proportion * data.shape[0])
    nb_test_samples = int(test_proportion * data.shape[0])

    x_train = data[:-nb_validation_samples - nb_test_samples]
    y_train = labels[:-nb_validation_samples - nb_test_samples]

    x_val = data[-nb_validation_samples - nb_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_samples - nb_test_samples:-nb_test_samples]

    x_test = data[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def remove_duplicates(tracks):
    """
    Removes duplicate tracks
    :param tracks: List of tracks
    :return: List of tracks without duplicates
    """
    tracks_without_duplicates = []
    duplicates = 0
    for track_obj in tracks:
        duplicate_found = False
        for t in tracks_without_duplicates:
            if track_obj['artists'] == t['artists'] and track_obj['name'].lower() == t['name'].lower():
                duplicate_found = True
                duplicates += 1

        if not duplicate_found:
            tracks_without_duplicates.append(track_obj)

    # print("Found %i duplicates" % duplicates)
    return tracks_without_duplicates


def remove_training_set_tracks(training_set_list, world_songs):
    tracks_without_overlap = []
    track_overlaps = 0
    for track_obj in world_songs:
        duplicate_found = False
        for t in training_set_list:
            if track_obj['artists'] == t['artists'] and track_obj['name'] == t['name']:
                duplicate_found = True
                track_overlaps += 1

        if not duplicate_found:
            tracks_without_overlap.append(track_obj)

    print("Number of training set tracks in world songs: %s" % track_overlaps)

    return tracks_without_overlap


def pre_process_cluster_data(audio_features):
    """
    Preprocesses the data.
    :param audio_features: All audio_features gathered. List of dictionaries
    :return: Three lists. The first one is the features, the two next ones are the corresponding songname and artist.
    """
    X_data, X_songname, X_artist = [], [], []

    total_features = get_features()

    for a in audio_features:

        new_list = [a['audio_features'][b] for b in a['audio_features'] if not isinstance(a['audio_features'][b], str)
                    and b in total_features]

        X_data.append(new_list)
        X_songname.append(a['name'])
        X_artist.append(a['artists'])

    return X_data, X_songname, X_artist

