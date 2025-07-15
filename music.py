import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist

# --- LOAD DATA ---
data = pd.read_csv("data.csv")

# --- FEATURES USED FOR RECOMMENDATION ---
number_cols = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]

# --- CLUSTER SONGS ---
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, n_init='auto'))
])
song_cluster_pipeline.fit(data[number_cols])
data['cluster_label'] = song_cluster_pipeline.predict(data[number_cols])


def get_song_data(song, spotify_data):
    try:
        return spotify_data[
            (spotify_data['name'] == song['name']) & (
                spotify_data['year'] == song['year'])
        ].iloc[0]
    except IndexError:
        print(
            f"Warning: {song['name']} ({song['year']}) not found in local data.")
        return None


def get_mean_vector(song_list, spotify_data):
    vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is not None:
            vectors.append(song_data[number_cols].values.astype(float))
    if not vectors:
        raise ValueError("No valid songs found for recommendation.")
    return np.mean(vectors, axis=0)


def recommend_songs(song_list, spotify_data, n_songs=10):
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.named_steps['scaler']
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_center, scaled_data, 'cosine')[0]
    spotify_data = spotify_data.copy()
    spotify_data['distance'] = distances
    input_names = {s['name'] for s in song_list}
    recs = spotify_data[~spotify_data['name'].isin(
        input_names)].nsmallest(n_songs, 'distance')
    return recs[['name', 'year', 'artists']]
# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    input_songs = [
        {'name': 'Come As You Are', 'year': 1991},
    ]
    recommendations = recommend_songs(input_songs, data)
    print("Recommended Songs:")
    if recommendations is not None and len(recommendations) > 0:
        print(recommendations.to_string(index=False))
    else:
        print("No recommendations found.")
print(type(recommendations))
