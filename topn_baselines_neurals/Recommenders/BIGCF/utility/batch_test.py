import torch
import numpy as np
from topn_baselines_neurals.Recommenders.BIGCF.utility.accuracy_measures import *

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def Recall_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    return recall


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, topks):
    sorted_items = X[0].numpy() # get the ground turth value.....
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    recall, ndcg = [], []
    for k in topks:
        recall.append(Recall_ATk(groundTrue, r, k))
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def eval_PyTorch(model, data_generator, Ks):
    
    recall_NDCG_dict = dict()
    for i in Ks:
        recall_NDCG_dict["Recall@"+str(i)] = Recall(i)
        recall_NDCG_dict["NDCG@"+str(i)] = NDCG(i)

    test_users = list(data_generator.test_set.keys())
    u_batch_size = data_generator.batch_size

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        rate_batch = model.predict(user_batch)

        exclude_index = []
        exclude_items = []
        ground_truth = []
        for i in range(len(user_batch)):
            train_items = list(data_generator.train_items[user_batch[i]])
            exclude_index.extend([i] * len(train_items))
            exclude_items.extend(train_items)
            ground_truth.append(list(data_generator.test_set[user_batch[i]]))

        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        rate_batch = rate_batch_k.cpu()

        for i in range(len(ground_truth)):
            predicted_items = list(np.array(rate_batch[i]))
            for key in recall_NDCG_dict:
                recall_NDCG_dict[key].add(set(ground_truth[i]), predicted_items)
    return recall_NDCG_dict

def sparse_eval_PyTorch(model, data_generator, Ks):

    test_users_1 = list(data_generator.test_set_sparse_1.keys())
    test_users_2 = list(data_generator.test_set_sparse_2.keys())
    test_users_3 = list(data_generator.test_set_sparse_3.keys())

    user_list = [test_users_1, test_users_2, test_users_3]

    final_results = []

    for k in range(3):
        result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
        u_batch_size = data_generator.batch_size

        n_test_users = len(user_list[k])
        n_user_batchs = n_test_users // u_batch_size + 1

        batch_rating_list = []
        ground_truth_list = []
        count = 0
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = user_list[k][start: end]
            rate_batch = model.predict(user_batch)

            exclude_index = []
            exclude_items = []
            ground_truth = []
            for i in range(len(user_batch)):
                train_items = list(data_generator.train_items[user_batch[i]])
                exclude_index.extend([i] * len(train_items))
                exclude_items.extend(train_items)
                ground_truth.append(list(data_generator.test_set[user_batch[i]]))
            rate_batch[exclude_index, exclude_items] = -(1 << 20)
            _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
            batch_rating_list.append(rate_batch_k.cpu())
            ground_truth_list.append(ground_truth)

        X = zip(batch_rating_list, ground_truth_list)
        batch_results = []
        for x in X:
            batch_results.append(test_one_batch(x, Ks))
        for batch_result in batch_results:
            result['recall'] += batch_result['recall'] / n_test_users
            result['ndcg'] += batch_result['ndcg'] / n_test_users

        assert count == n_test_users
        final_results.append(result)

    return final_results