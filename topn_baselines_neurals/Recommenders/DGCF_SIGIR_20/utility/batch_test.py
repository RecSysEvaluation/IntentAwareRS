'''
Created on Oct 10, 2019
Tensorflow Implementation of Disentangled Graph Collaborative Filtering (DGCF) model in:
Wang Xiang et al. Disentangled Graph Collaborative Filtering. In SIGIR 2020.
Note that: This implementation is based on the codes of NGCF.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''

import numpy as np
import tensorflow as tf
from  tqdm import tqdm

#  import m
from topn_baselines_neurals.Recommenders.DGCF_SIGIR_20.utility.accuracy_measures import *
measure_Recall_NDCG = dict()


for i in [1, 5, 10, 20, 50, 100]:
    measure_Recall_NDCG["Recall_"+str(i)] = Recall(i)
    measure_Recall_NDCG["NDCG"+str(i)] = NDCG(i)


def model_testing(sess, model, users_to_test, test_data_dic = None, ITEM_NUM = None, BATCH_SIZE = 100):
    
    # batch size for test data.... we keep it small becuase otherwise it will raise memory concerns
    u_batch_size = 5
    test_users = users_to_test
    n_test_users = len(users_to_test)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

        item_batch = range(0, ITEM_NUM)
        rate_batch = sess.run(model.batch_ratings, {model.users: user_batch, model.pos_items: item_batch})

        for i in range(len(user_batch)):
            user_id = user_batch[i]
            user_item_score =  np.argsort(rate_batch[i])[-100:][::-1]
            for key in measure_Recall_NDCG:
                measure_Recall_NDCG[key].add(set(test_data_dic[user_id]), user_item_score.copy())

    measure_Recall_NDCG_temp = dict()
    for key, _ in measure_Recall_NDCG.items():
        measure_Recall_NDCG_temp[key] = measure_Recall_NDCG[key].getScore()
    return measure_Recall_NDCG_temp








