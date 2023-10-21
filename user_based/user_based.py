import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_user(N, quried_user_id, users, data):
    """
    N (int) : function will return N similar users' ids
    quried_user_id (int) : the recommending user's id 
    users (user list)  : the list of user data (include history_watch_list, and watch_proportion)
    data (...?) : all videos data labels

    return : return N similar users' ids
    """
    # weigthed the data with watching proportion
    quried_user = users[quried_user_id]
    user_watched_data = data.iloc[quried_user['watched_indices']]
    quried_vec = user_watched_data.reset_index(drop=True).astype(float)

    for ind, row in quried_vec.iterrows():
        row = row * quried_user['watched_proportion'][ind]
        quried_vec.loc[ind] = row
    quried_vec = quried_vec.sum().to_frame().T

    users_vecs = None

    for user in users:
        user_watched_data = data.iloc[user['watched_indices']]
        feature_vec = user_watched_data.reset_index(drop=True).astype(float)
        for ind, row in feature_vec.iterrows():
            row = row * user['watched_proportion'][ind]
            feature_vec.loc[ind] = row
        feature_vec = feature_vec.sum().to_frame().T.astype(float)
        users_vecs = pd.concat([users_vecs, feature_vec], axis=0)

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(quried_vec, users_vecs)

    # Get indices of users sorted by similarity
    user_indices = sorted(range(len(
        cosine_similarities[0])), key=lambda i: cosine_similarities[0][i], reverse=True)
    similar_user_indices = [
        index for index in user_indices if index != quried_user_id]

    # Recommend the top N videos
    top_n_similar_users_indices = similar_user_indices[:N]

    return top_n_similar_users_indices


def find_similar_video_from_users(N, quried_user_id, similar_user_ids, users, data):
    """
    N (int) : function will return N similar videos' ids
    quried_user_id (int) : the recommending user's id 
    similar_user_ids (int list) : the similar users' id
    users (user list)  : the list of user data (include history_watch_list, and watch_proportion)
    data (...?) : all videos data labels

    return : return N similar users' ids
    """
    video_indices_from_users = []

    for user_id in similar_user_ids:
        video_indices_from_users = np.concatenate(
            [video_indices_from_users, users[user_id]['watched_indices']])

    video_indices_from_users = np.unique(
        video_indices_from_users).astype(float)

    data_from_users = data.iloc[video_indices_from_users]

    quried_user = users[quried_user_id]
    user_watched_data = data.iloc[quried_user['watched_indices']]
    quried_vec = user_watched_data.reset_index(drop=True).astype(float)
    for ind, row in quried_vec.iterrows():
        row = row * quried_user['watched_proportion'][ind]
        quried_vec.loc[ind] = row
    quried_vec = quried_vec.sum().to_frame().T

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(quried_vec, data_from_users)

    # Get indices of videos sorted by similarity (excluding the user's watched videos)
    indices = sorted(range(len(
        cosine_similarities[0])), key=lambda i: cosine_similarities[0][i], reverse=True)
    video_indices = video_indices_from_users[indices]
    recommended_indices = [
        index for index in video_indices if index not in users[quried_user_id]["watched_indices"]]

    # Recommend the top N videos
    top_n_recommendations = recommended_indices[:N]

    return top_n_recommendations
