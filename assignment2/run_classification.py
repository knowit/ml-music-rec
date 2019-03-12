import os
import sys

# Append path to use modules outside pycharm environment, e.g. terminal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from dataset.dataset_config import likeable_songs, annoying_songs
from helper_functions.pre_processing import filter_features, split_dataset, label_songs, remove_duplicates, remove_training_set_tracks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from features import get_features
from helper_functions.helpers import load_pickle, plot_training_history
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn import svm
import pickle


np.set_printoptions(threshold=np.nan)

np.random.seed(6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)

'''
DATA PRE-PROCESSING
'''
# Remove duplicates
x_likeable = remove_duplicates(likeable_songs)
x_annoying = remove_duplicates(annoying_songs)

print("Number of likeable songs in training dataset %i" % len(x_likeable))
print("Number of annoying songs in training dataset %i\n" % len(x_annoying))

# Filter out chosen features and omit metadata such as song name and artist
x_likeable_data = filter_features(x_likeable)
x_annoying_data = filter_features(x_annoying)

# Assign labels
y_likeable = label_songs(x_likeable_data, 1)
y_annoying = label_songs(x_annoying, 0)

# Create list containing entire dataset input
x_data = np.array(x_likeable_data + x_annoying_data)
y_labels = np.array(y_likeable + y_annoying)

# Normalize input training data
x_data = StandardScaler().fit_transform(x_data)

# Randomize data and split into training, validation and test set
x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_data, y_labels, 0.1, 0.1)


def train_nn(save_model=False):
    '''
    Train neural network model
    :param save_model: True if model should be saved to file
    '''

    ### TRAINING ###
    # Create model

    # TODO: Here you can modify the architecture of the neural network model and experiment with different parameters
    model = Sequential()
    model.add(Dense(1,  # TODO: Number of hidden layer neurons
                    input_dim=len(get_features()),
                    activation='relu'))

    # TODO: Possible to add additional neural network layers here
    # TODO: Use model.add(Dense(number of hidden layer neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # TODO: Optional; add early stopping as callback
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=[x_val, y_val],
                        batch_size=50,  # Number of data samples to run through network before parameter update
                        epochs=1,  # TODO: Number of times to run entire training set through network
                        shuffle=True,
                        callbacks=[]
                        ).history

    score = model.evaluate(x_test, y_test, batch_size=50)  # Evaluate model on test set
    print('Test loss:%f' % (score[0]))
    print('Test accuracy:%f' % (score[1]))

    if save_model:
        model.save('./models/nn_model.h5')
        print("Model saved")

    plot_training_history(history)


def music_recommendation_nn(model_path='./models/nn_model.h5'):
    model = load_model(model_path)
    world_songs = load_pickle('../playlist_features/Mega List/world_songs.pkl')

    # Pre-process songs, as was done for training set
    world_songs_without_duplicates = remove_duplicates(world_songs)

    # Must also remove training set songs from world songs
    training_set = x_likeable + x_annoying
    world_songs_without_duplicates = remove_training_set_tracks(training_set, world_songs_without_duplicates)

    x_data = StandardScaler().fit_transform(np.array(filter_features(world_songs_without_duplicates)))
    y_pred = model.predict(x_data, batch_size=50)

    # Find 30 songs, which are most likeable
    song_pred_mapping = [{
        'name': world_songs_without_duplicates[i]['name'],
        'artists': world_songs_without_duplicates[i]['artists'],
        'likeability_score': y_pred[i],
    } for i in range(len(y_pred))]

    # Sort songs by likeability score
    song_pred_mapping_sorted_by_likeability = sorted(song_pred_mapping, key=lambda x: x['likeability_score'])

    # Extract 30 most likeable songs and 30 most annoying songs
    recommendations = song_pred_mapping_sorted_by_likeability[-30:]
    thirty_most_annoying_songs = song_pred_mapping_sorted_by_likeability[:30]

    # Print most likeable and annoying songs
    print("\n------Your music recommendations-------")
    print(pd.DataFrame(recommendations)[['artists', 'name', 'likeability_score']])

    print("\n-------Songs you will definitely hate-------")
    print(pd.DataFrame(thirty_most_annoying_songs)[['artists', 'name', 'likeability_score']])
    print("")


def train_svm():
    # TODO: Instantiate SVM object using Sklearn framework
    clf = None

    # TODO: Train SVM by calling fit() and passing training set + labels ass arguments

    # TODO: Output accuracy on validation set by calling score() and passing validation set + label as arguments
    val_score = ""

    # TODO: Output accuracy on validation set by calling score() and passing validation set + label as arguments
    test_score = ""

    print('Validation accuracy of SVM: ' + str(val_score))
    print('Test accuracy of SVM: ' + str(test_score))

    filehandler = open('./models/svm_model.pkl', 'wb')
    pickle.dump(clf, filehandler)


def music_recommendation_svm():
    world_songs = load_pickle('../playlist_features/Mega List/world_songs.pkl')

    # Pre-process songs, as was done for training set
    world_songs_without_duplicates = remove_duplicates(world_songs)

    # Must also remove training set songs from world songs
    training_set = x_likeable + x_annoying
    world_songs_without_duplicates = remove_training_set_tracks(training_set, world_songs_without_duplicates)

    filehandler = open('./models/svm_model.pkl', 'rb')
    clf = pickle.load(filehandler)

    all_data = StandardScaler().fit_transform(np.array(filter_features(world_songs_without_duplicates)))
    y_pred_probability = clf.predict_proba(all_data)
    y_pred = clf.predict(all_data)

    # Find 30 songs, which are most likeable
    song_pred_mapping = [{
        'name': world_songs_without_duplicates[i]['name'],
        'artists': world_songs_without_duplicates[i]['artists'],
        'likeability_score': y_pred_probability[i][1],
    } for i in range(len(y_pred))]

    # Sort songs by likeability score
    song_pred_mapping_sorted_by_likeability = sorted(song_pred_mapping, key=lambda x: x['likeability_score'])

    # Extract 30 most likeable songs and 30 most annoying songs
    recommendations = list(reversed(song_pred_mapping_sorted_by_likeability[-30:]))
    thirty_most_annoying_songs = song_pred_mapping_sorted_by_likeability[:30]

    # Print most likeable and annoying songs
    print("\n------Your music recommendations-------")
    print(pd.DataFrame(recommendations)[['artists', 'name', 'likeability_score']])

    print("\n-------Songs you will definitely hate-------")
    print(pd.DataFrame(thirty_most_annoying_songs)[['artists', 'name', 'likeability_score']])
    print("")


# TODO: Comment in appropriate methods for training neural network/svm

# SVM
# train_svm()
# music_recommendation_svm()

# Neural network
# train_nn(save_model=True)
# music_recommendation_nn()
