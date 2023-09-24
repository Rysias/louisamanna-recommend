import os
import typing as T
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from dotenv import load_dotenv
from joblib import Memory
from loguru import logger
from sklearn.cluster import KMeans
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

memory = Memory(".joblib_cache", verbose=0)

load_dotenv()


class TrackInfo(T.NamedTuple):
    name: str
    artists: str


def get_track_info(track: dict) -> TrackInfo:
    title = track["name"]
    artists = ", ".join([artist["name"] for artist in track["artists"]])
    return TrackInfo(title, artists)


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


def tracks_per_user(
    tracks: list[dict[str, T.Any]],
) -> dict[str, list[dict[str, T.Any]]]:
    user_track_dict = defaultdict(list)

    for track in tracks:
        user_id = track["added_by"]["id"]
        track_features = track["track"]
        user_track_dict[user_id].append(track_features)

    return user_track_dict


def filter_audio_features(features: list[dict[str, float]]) -> list[list[float]]:
    features_to_extract = {
        "danceability",
        "energy",
        "key",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    }
    return [
        [feature[feature_name] for feature_name in features_to_extract]
        for feature in features
    ]


def extract_audio_features(sp: Spotify, tracks: list) -> list:
    track_ids = [track["id"] for track in tracks]
    return filter_audio_features(sp.audio_features(track_ids))


def perform_clustering(features_list: list, n_clusters: int = 3) -> np.ndarray:
    X = np.array(features_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    labels = kmeans.labels_
    return labels


def get_user_recommendation(track_list: list[dict[str, T.Any]], sp: Spotify) -> list:
    features = extract_audio_features(sp, track_list)
    labels = perform_clustering(features)
    return get_recommendations(sp, labels, [track["id"] for track in track_list])


def get_recommendations(sp: Spotify, labels: np.ndarray, track_ids: list) -> list:
    recommendations = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip the noise cluster
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        seed_tracks = [track_ids[i] for i in indices][:5]
        recs = sp.recommendations(seed_tracks=seed_tracks, limit=5)
        recommendations[cluster_id] = [get_track_info(rec) for rec in recs["tracks"]]
    return recommendations


@memory.cache
def get_user_name(sp: Spotify, user_id: str) -> str:
    user_data = sp.user(user_id)
    return user_data.get("display_name", None)


@click.command(
    help="This script clusters tracks in a given Spotify playlist and provides song recommendations for each cluster.",
)
@click.option("--playlist_id", prompt="Playlist ID", help="The Spotify Playlist ID.")
def main(playlist_id: str) -> None:
    logger.info(f"Starting program with playlist_id: {playlist_id}")

    logger.info("Initializing Spotipy...")
    sp = initialize_spotipy()

    logger.info("Fetching playlist tracks...")
    tracks = fetch_playlist_tracks(sp, playlist_id)

    logger.info("Grouping tracks by user...")
    user_track_dict = tracks_per_user(tracks)

    logger.info("Generating recommendations...")
    OUTPUT_PATH = Path("output/recommendations.md")
    with OUTPUT_PATH.open("w", encoding="utf-8") as md_file:
        for user_id, track_list in user_track_dict.items():
            user_name = get_user_name(sp, user_id)
            md_file.write(f"## Recommendations for {user_name}\n")
            logger.info(f"Generating for {user_id=}")
            recommendations = get_user_recommendation(track_list, sp)
            for cluster_id, tracks in recommendations.items():
                md_file.write(f"### Cluster {cluster_id}\n")
                for track in tracks:
                    md_file.write(f"- {track}\n")


if __name__ == "__main__":
    logger.add("logs/recommend.log", rotation="1 day")
    main()
