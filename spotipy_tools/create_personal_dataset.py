import sys
import os
# Append path to use modules outside pycharm environment, e.g. terminal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from spotipy_tools.spotipy_helpers import save_audio_features_for_list_of_playlists
from helper_functions.helpers import load_pickle


# Add playlist URIs for custom datasets,
# then run this file and uncomment likeable_songs and annoying_songs lines in dataset_config

# TODO: E.g: "spotify:user:jooney:playlist:60FUjmTcUrnewNksBNFVWX"
spotify_uris_likeable_songs = [

]

# TODO: F. eks: "spotify:user:spotifycharts:playlist:37i9dQZEVXbJvfa0Yxg7E7"
spotify_uris_annoying_songs = [

]

save_audio_features_for_list_of_playlists(spotify_uris_likeable_songs, "../playlist_features/likeable_songs")
save_audio_features_for_list_of_playlists(spotify_uris_annoying_songs, "../playlist_features/annoying_songs")

print("")

