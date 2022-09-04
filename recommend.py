import pandas as pd
import random
import authorization # this is the script we created earlier
import numpy as np
from numpy.linalg import norm
from face import get_VAD

def recommend(mood_vec, ref_df, n_recs=5):
    track_moodvec = mood_vec

    # Compute distances to all reference tracks
    ref_df["distances"] = ref_df["mood_vec"].apply(lambda x: norm(track_moodvec - np.array(x)))
    ref_df_sorted = ref_df.sort_values(by="distances", ascending=True)

    # Return n recommendations
    return ref_df_sorted.iloc[:n_recs]

df = pd.read_csv("muse_v3.csv")

# df["mood_vec"] = df[["valence", "energy"]].values.tolist()
df["mood_vec"] = df[["valence_tags","arousal_tags", "dominance_tags"]].values.tolist()

# Valence, arousal, dominance
mood_vec = get_VAD("../data/fear3.jpg")  # [1.1338692880875, 6.037996638576, 2.4078040275840005]
result = (recommend(mood_vec, ref_df = df, n_recs = 15))

# print(result[['track','spotify_id']])
# print(np.array(result[['lastfm_url', 'track', 'spotify_id', 'artist','distances']]))

final_results = result.dropna().iloc[0]['spotify_id']
output = "https://open.spotify.com/track/" #67rLkvNELkXfa64EdpYT1A"
print(output+final_results)
