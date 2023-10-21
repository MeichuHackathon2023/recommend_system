def print_feature_vec_user(N, user_id, users, data):
    """
    N : print out top N genre the quried user like
    user_id (int) : quried user id
    users (user list) : a list of user data (include history_watch_list, and watch_proportion)
    data (df) : a labeled data list
    """
    quried_user = users[user_id]
    user_watched_data = data.iloc[quried_user['watched_indices']]
    quried_vec = user_watched_data.reset_index(drop=True)
    for ind, row in quried_vec.iterrows():
        row = row * quried_user['watched_proportion'][ind]
        quried_vec.loc[ind] = row
    quried_vec = quried_vec.sum().to_frame().T
    result = quried_vec.loc[0]
    print(result[result != 0].sort_values(ascending=False)[:3])

def search_tag(id, data):
    """
    id : the videos id of the searched video
    data : all labeled data list
    """
    target = data.iloc[id]
    target = target[target != 0]
    print("the tag of id = " + str(id))
    for tag in target.index:
        print(tag)