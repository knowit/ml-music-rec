import os
import sys
# Append path to use modules outside pycharm environment, e.g. terminal
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

from constants import DIR_PLAYLIST_FEATURES
from helper_functions.helpers import save_pickle
from spotipy_tools.secrets import CLIENT_ID, CLIENT_SECRET


def get_user_playlists(sp, username):
    """
    Get all playlists by user
    :param sp: Spotipy object with client_id and client_secret
    :param username:
    :return: List with dicts, containing name and spotify uri of objects.
    """
    playlists = sp.user_playlists(username)
    filtered_playlists = []
    for playlist in playlists['items']:
        if playlist['owner']['id'] == username:
            print(playlist)
            filtered_playlists.append({'name': playlist['name'], 'uri': playlist['uri']})

    return filtered_playlists


def get_audio_analysis(sp, track):
    """
    Get audio analysis for a track
    :param sp: Spotipy
    :param track: Spotify URI
    :return: Return
    """
    return sp.audio_analysis(track)


def get_audio_features_for_tracks(sp, track_list):
    """
    Get audio features for a list of track URIs
    :type sp: spotipy.Spotify

    :param sp: Spotipy
    :param track_list: List of track URIs
    :return: Return a list of dicts, for each track, containing audio features, artists, song uri and name of song
    """
    # Some playlists have local tracks, which need to be avoided ('spotify:local')
    track_uris = [track['uri'] for track in track_list if track['uri'].strip().startswith('spotify:track')]
    features = []

    full_batch_size = 50
    num_full_batches = len(track_uris) // full_batch_size

    # Print vars
    total = num_full_batches*full_batch_size
    for i in range(0, total, full_batch_size):
        track_uris_subset = track_uris[i:i+full_batch_size]
        track_list_subset = track_list[i:i+full_batch_size]

        try:
            audio_features = sp.audio_features(track_uris_subset)
            audio_features_and_info = [{**track_list_subset[j], 'audio_features': audio_features[j]} for j in range(len(track_list_subset))]
            features += audio_features_and_info
        except Exception as e:
            print(track_uris)
            print(e)

        # print_progress(i, total)

    remainder = len(track_uris) % full_batch_size
    if remainder != 0:
        audio_features = sp.audio_features(track_uris[-remainder:])
        audio_features_and_info = [{**track_list[-remainder:][j], 'audio_features': audio_features[j]} for j in
                                   range(len(track_list[-remainder:]))]
        features += audio_features_and_info

    return features


def get_audio_analysis_for_tracks(sp, track_list):
    """
    Get audio analysis for list of tracks
    :type sp: spotipy.Spotify

    :param sp: Spotipy
    :param track_list: List containing dict for each track. Must contain uri key.
    :return: list of dicts with track info and audio analysis
    """
    audio_analysis = []

    for track in track_list:
        audio_analysis.append({**track, 'audio_analysis': get_audio_analysis(sp, track['uri'])})

    return audio_analysis


def get_playlist_track_info(sp, playlist_uri):
    """
    Get track information for a playlist
    :type sp: spotipy.Spotify

    :param sp: Spotipy
    :param playlist_uri: Spotify playlist URI
    :return:
    List containing dictionary for each track.
    Each dict contains tracl info: {artists: List, name: string (?), uri: string (?)}
    """
    username = get_user_name_from_spotify_uri(playlist_uri)
    playlist = sp.user_playlist(username, playlist_uri)
    track_info = []

    tracks_batch = playlist['tracks']
    track_info += _get_track_info_from_tracks_batch(tracks_batch)

    while tracks_batch['next']:
        tracks_batch = sp.next(tracks_batch)
        track_info += _get_track_info_from_tracks_batch(tracks_batch)

    return track_info


def _get_track_info_from_tracks_batch(tracks_batch):
    track_info = []

    for trackObj in tracks_batch['items']:
        track = trackObj['track']
        if track is None:
            continue
        else:
            artists = [artist['name'] for artist in track['artists']]
            track_info.append({'artists': artists, 'name': track['name'], 'uri': track['uri']})

    return track_info


def initSpotipy(client_id, client_secret, token=None):
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id,
                                                          client_secret=client_secret)
    if token:
        sp = spotipy.Spotify(auth=token, client_credentials_manager=client_credentials_manager)
    else:
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp


def get_user_name_from_spotify_uri(spotify_uri):
    uri_split = spotify_uri.split(':')
    return uri_split[2]


def save_playlist_audio_features(playlist_uri, dir_path):
    sp = initSpotipy(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    username = get_user_name_from_spotify_uri(playlist_uri)
    playlist_name = sp.user_playlist(username, playlist_uri)['name']
    track_list = get_playlist_track_info(sp,
                                  playlist_uri=playlist_uri)

    audio_features = get_audio_features_for_tracks(sp, track_list)

    save_pickle(dir_path + playlist_name, audio_features)


def save_audio_features_for_list_of_playlists(playlist_uri_list, save_path):
    sp = initSpotipy(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    all_playlists_track_info = [get_playlist_track_info(sp, playlist_uri=playlist_uri)
                                for playlist_uri in playlist_uri_list]

    audio_features_by_playlist = [get_audio_features_for_tracks(sp, track_list) for track_list in all_playlists_track_info]

    # Flatten previous list
    all_songs_audio_features = [song for playlist in audio_features_by_playlist for song in playlist]

    print("\nTotal number of songs in %s: %i" % (save_path, len(all_songs_audio_features)))
    save_pickle(save_path, all_songs_audio_features)


def parse_all_playlists(file_path='./playlists.txt'):
    genre_dir = None
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('spotify:user'):
                genre = line.strip()
                genre_dir = DIR_PLAYLIST_FEATURES + genre + '/'
                os.mkdir(genre_dir)
                print("Mkdir %s" % genre_dir)
            else:
                save_playlist_audio_features(line.strip(), genre_dir)
