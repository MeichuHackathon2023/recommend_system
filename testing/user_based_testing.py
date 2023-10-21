import sys
sys.path.insert(1, '../user_based')
import user_based
import test_util

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# read in datas
data = pd.read_csv("../data/all_video_label.csv").drop(columns=['video_id']).drop(columns=['第四冊', '第三冊','第二冊','第一冊',])

# create random users
random_user_count = 10
users = []
for i in range(random_user_count):
    random_watched_indices_array = np.random.randint(1, 200, 10)
    random_watched_proportion_array = np.random.uniform(0, 1, 10)
    users.append({
         'watched_indices' : random_watched_indices_array,
         'watched_proportion' : random_watched_proportion_array
    })

# find similar users
similar_users_id = user_based.find_similar_user(3, 0, users, data)
print(similar_users_id)

# print out feature info of similar users
print("========== similar users ==========")
for i in similar_users_id:
    test_util.print_feature_vec_user(3, i, users, data)


# print out feature info of quried users
print("========== quried users ==========")
test_util.print_feature_vec_user(5, 0, users, data)

# find videos from similar users
video_indices = user_based.find_similar_video_from_users(10, 0, similar_users_id, users, data)
print(video_indices)

# print out video tags of recommend videos
for i in video_indices:
    test_util.search_tag(int(i), data)