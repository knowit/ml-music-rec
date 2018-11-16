# Music Recommendations Using Machine Learning

## Initial setup
1. Install Python 3.6
2. Execute `pip install -r requirements.txt` to install
required Python packages. **Optional**: Use virtualenv to create a project-specific environment for the packages: https://virtualenv.pypa.io/en/stable/.
This is recommended to keep package dependencies local to the project instead of being global on your machine.
3. If you want to use spotipy to fetch features for your own playlists,
create a `secrets.py` file, in `spotipy_tools folder, containing CLIENT_ID and CLIENT_SECRET constants (to use with Spotify API).
Otherwise you can just construct the dataset from the playlists in the repo (`dataset_config.py`).

Note: The file names of downloaded playlists (pkl files) are parsed from playlist names and Windows users may have issues with these. If that is the case, just rename *.pkl file or skip file.

## Spotify Web API
https://developer.spotify.com/web-api/

Get CLIENT_ID and CLIENT_SECRET by registering at https://developer.spotify.com/my-applications/


## Spotipy
Python package which implements Spotify Web API,
which is used to extract features:
http://spotipy.readthedocs.io/en/latest/

## Framework Documentation
- sklearn: http://scikit-learn.org/stable/documentation.html
- keras: https://keras.io/

