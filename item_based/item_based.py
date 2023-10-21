import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity
"""
N (int) : function will return N similar videos
user_watched_indices (int list) : history watched video id list
watched_proportions (float list) : watched proportion list (0.0 ~ 1.0)
data (...?) : all videos data labels

recommend N videos given user's history watch list
"""
def find_n_videos_by_histories(N, user_watched_indices, watched_proportions, data):
    # 
    user_watched_data = data.iloc[user_watched_indices]
    user_watched_data = user_watched_data.reset_index(drop=True)

    # weigthed sum the data with watching proportion
    for ind, row in user_watched_data.iterrows():
        row = row * watched_proportions[ind]
        user_watched_data.loc[ind] = row
    sum = user_watched_data.sum()

    # Compute cosine similarity
    cosine_similarities = cosine_similarity([sum], data)

    # Get indices of videos sorted by similarity (excluding the user's watched videos)
    video_indices = sorted(range(len(cosine_similarities[0])), key=lambda i: cosine_similarities[0][i], reverse=True)
    recommended_indices = [index for index in video_indices if index not in user_watched_indices]

    # Recommend the top N videos
    top_n_recommendations = recommended_indices[:N]  # Replace N with the desired number of recommendations

    # Display or use the recommended indices
    print("Recommended Video Indices:", top_n_recommendations)

    return top_n_recommendations

    for rec in top_n_recommendations:
        search_tag(rec)