import os

from constants import DIR_PLAYLIST_FEATURES
from helper_functions.helpers import load_pickle
from helper_functions.helpers import print_progress


# TODO: Finish
def save_track_lists():
    genres = os.listdir(DIR_PLAYLIST_FEATURES)
    for genre_name in genres:
        genre_dir = DIR_PLAYLIST_FEATURES + genre_name + '/'
        with open(genre_dir + 'track_list.txt', 'w') as f:
            f.write('Genre: %s\n' % genre_name)
            playlist_file_names = os.listdir(genre_dir)
            for playlist_name in playlist_file_names:
                if playlist_name != 'track_list.txt':
                    f.write('\n\nPlaylist name: %s\n\n' % playlist_name.strip('.pkl'))
                    playlist_path = genre_dir + playlist_name
                    tracks = load_pickle(playlist_path)

                    # print('"%s",' % playlist_path)
                    for track_obj in tracks:
                        for artist in track_obj['artists']:
                            f.write('%s, ' % artist)
                        f.write("- %s\n" % track_obj['name'])


def load_songs_from_playlists(playlist_file_paths):
    '''

    :param playlist_file_paths: List of file paths (without extension) to playlists

    :type playlist_file_paths: List
    :return: List of songs dicts, with from provided playlists
    '''

    total_number_of_songs = 0
    result = []
    count = 0
    print("Unpickling playlists")
    for playlist_path in playlist_file_paths:
        playlist = load_pickle(playlist_path)
        total_number_of_songs += len(playlist)
        result += playlist

        count += 1
        print_progress(count, len(playlist_file_paths))

    return result

print('')