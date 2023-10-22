import sys
sys.path.insert(1, '../user_based/user_based.py')
import user_based

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

N = 10

user_portion_2d_list = np.zeros((10, 198))
for i in range(10):
    for j in np.random.randint(1, 198, 5):
        portion = np.random.uniform(0, 1, 1)[0]
        user_portion_2d_list[i][j] = portion

user_based.Recommend_given2D(N, user_portion_2d_list)