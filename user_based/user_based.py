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

    return : return N similar videos' ids
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
    top_n_recommendations = []

    for i in range(len(recommended_indices)):
        top_n_recommendations.append(recommended_indices[:len(recommended_indices)][i])
        if (i % 2 == 0 and i != 0):
            random_id = np.random.randint(0, len(data), 1)[0]
            while (random_id in recommended_indices[:N] or random_id in users[quried_user_id]["watched_indices"]):
                random_id = np.random.randint(0, len(data), 1)[0]
            top_n_recommendations.append(random_id)

    return top_n_recommendations[:N]


def Recommend(N, quried_user_id, quried_user_grade, users):
    """
    N (int) : function will return N similar videos' ids
    quried_user_id (int) : the recommending user's id 
    quried_user_grade (int) : the recommending user's grade (education level)
    users (user list)  : the list of user data (include history_watch_list, and watch_proportion)
    return : return N similar videos' ids
    """
    data = pd.read_csv('../data/all_video_label_v4.csv').drop(columns=["video_id"])
    quried_user = users[quried_user_id]
    user_watched_data = data.iloc[quried_user['watched_indices']]
    if (user_watched_data.empty):
        if quried_user_grade == 0:
            filtered_df = data[(data["第一冊"] == 1) | (data["第二冊"] == 1)]
            return filtered_df.sample(N)
        else:
            return data.sample(N)

    similar_user_ids = find_similar_user(N, quried_user_id, users, data)
    recommendations = find_similar_video_from_users(
        N, quried_user_id, similar_user_ids, users, data)
    
    for ind in range(len(recommendations)):
        recommendations[ind] = int(recommendations[ind])

    #print(recommendations)
    return recommendations

def Recommend_given2D(N ,user_portion_2dlist):
    users = []
    for i in range(len(user_portion_2dlist)):
        user = {
            "watched_indices" : [],
            "watched_proportion" :[] 
        }
        for ind in range(len(user_portion_2dlist[0])):
            if user_portion_2dlist[i][ind] > 0:
                user['watched_indices'].append(ind)
                user['watched_proportion'].append(user_portion_2dlist[i][ind])
                users.append(user)
    
    print(json.dumps((Recommend(N, 0, [], users))))

import json
#import sys, json
"""
if __name__ == '__main__':
    N = json.loads(sys.argv[1])
    user_portion_2dlist = json.loads(sys.argv[2])

    Recommend_given2D(10, user_portion_2dlist)
"""


    




