import spotipy
from spotipy.oauth2 import SpotifyOAuth
import hdbscan
import numpy as np

def initialize_spotipy(client_id, client_secret, redirect_uri):
    return spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                     client_secret=client_secret,
                                                     redirect_uri=redirect_uri,
                                                     scope="playlist-read-collaborative"))

def fetch_playlist_tracks(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)
    return results['items']

def extract_audio_features(sp, tracks):
    track_ids = [track['track']['id'] for track in tracks]
    return sp.audio_features(track_ids)

def perform_clustering(features_list):
    X = np.array([[feature[x] for x in ['danceability', 'energy', 'tempo']] for feature in features_list])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    clusterer.fit(X)
    return clusterer.labels_

def get_recommendations(sp, labels, track_ids):
    recommendations = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip the noise cluster
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        seed_tracks = [track_ids[i] for i in indices]
        recs = sp.recommendations(seed_tracks=seed_tracks, limit=5)
        recommendations[cluster_id] = [rec['name'] for rec in recs['tracks']]
    return recommendations

def main():
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    redirect_uri = "YOUR_REDIRECT_URI"
    playlist_id = "YOUR_PLAYLIST_ID"

    sp = initialize_spotipy(client_id, client_secret, redirect_uri)
    tracks = fetch_playlist_tracks(sp, playlist_id)
    features_list = extract_audio_features(sp, tracks)
    labels = perform_clustering(features_list)
    recommendations = get_recommendations(sp, labels, [track['track']['id'] for track in tracks])

    for cluster, recs in recommendations.items():
        print(f"Recommendations for cluster {cluster}: {recs}")

if __name__ == "__main__":
    main()
