import os

import click
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()


def initialize_spotipy() -> Spotify:
    return Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            redirect_uri=os.getenv("REDIRECT_URI"),
            scope="playlist-read-collaborative",
        ),
    )


def fetch_playlist_tracks(sp: Spotify, playlist_id: str) -> list:
    results = sp.playlist_tracks(playlist_id)
    return results["items"]


def extract_audio_features(sp: Spotify, tracks: list) -> list:
    track_ids = [track["track"]["id"] for track in tracks]
    return sp.audio_features(track_ids)


def perform_clustering(features_list: list, n_clusters: int = 3) -> np.ndarray:
    X = np.array(features_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels


def get_recommendations(sp: Spotify, labels: np.ndarray, track_ids: list) -> list:
    recommendations = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip the noise cluster
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        seed_tracks = [track_ids[i] for i in indices]
        recs = sp.recommendations(seed_tracks=seed_tracks, limit=5)
        recommendations[cluster_id] = [rec["name"] for rec in recs["tracks"]]
    return recommendations


@click.command(
    help="This script clusters tracks in a given Spotify playlist and provides song recommendations for each cluster.",
)
@click.option("--playlist_id", prompt="Playlist ID", help="The Spotify Playlist ID.")
def main(playlist_id: str) -> None:
    sp = initialize_spotipy()
    tracks = fetch_playlist_tracks(sp, playlist_id)
    features_list = extract_audio_features(sp, tracks)
    labels = perform_clustering(features_list)
    recommendations = get_recommendations(
        sp,
        labels,
        [track["track"]["id"] for track in tracks],
    )

    for cluster, recs in recommendations.items():
        print(f"Recommendations for cluster {cluster}: {recs}")


if __name__ == "__main__":
    main()
