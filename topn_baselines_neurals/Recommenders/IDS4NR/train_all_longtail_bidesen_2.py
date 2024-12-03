import pandas as pd
import numpy as np
import random
import time
import pickle
import argparse
import scipy.sparse as sp
from data_utils import UserItemRatingDataset, instance_a_train_loader, instance_a_eval_loader, instance_u_eval_loader_all, instance_a_train_loader_weight
from metric import getR, getNDCG, calculate_ndcg, getCC, getILD, getNovel
from ncf_bi_longtail import LFM_Bi_Disen8, NCF_Bi_Disen8
from cml_bi_longtail import CML_Bi_Disen8
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from accuracy_measures import *



parser = argparse.ArgumentParser()
# Train Test
parser.add_argument('--n_epochs',
                    type=int,
                    default=1,
                    help="Number of epoch during training")
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help="Batch size in one iteration")
parser.add_argument('--num_neg',
                    type=int,
                    default=4,
                    help="Number of Negative Sampling")
#parser.add_argument('--test_neg',
#                    type=int,
#                    default=200,
#                    help="Number of Test Ranking")
# Model
parser.add_argument('--dim',
                    type=int,
                    default=300,
                    help="Dimension for embedding")
parser.add_argument('--lamda',
                    type=float,
                    default=0.01,
                    help="extra regulation coeffcient")
# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help="Learning rate")
parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help="Weight decay factor")
parser.add_argument('--test_all',
                    type=bool,
                    default=True)
args = parser.parse_args()
print(args)

def evaluate(user_pred, test_user_list, train_user_list):
    lst_hr_5, lst_hr_10, lst_ndcg_5, lst_ndcg_10 = [], [], [], []
    coverage_5, coverage_10, novel_5, novel_10 = set(), set(), [], []
    for u in user_pred.keys():
        if len(test_user_list[u]) == 0 or len(train_user_list[u]) == 0:
            continue
        u_pred = sorted(user_pred[u], key=lambda x:x[1], reverse=True)
        u_pred_5 = [u[0] for u in u_pred[:5]]
        u_pred_10 = [u[0] for u in u_pred[:10]]
        hr_5, ndcg_5 = getR(u_pred_5, test_user_list[u]), calculate_ndcg(test_user_list[u], u_pred_5, 10)#getNDCG(u_pred_5, test_user_list[u])
        hr_10, ndcg_10 = getR(u_pred_10, test_user_list[u]), calculate_ndcg(test_user_list[u], u_pred_10, 20)#getNDCG(u_pred_10, test_user_list[u])
        lst_hr_5.append(hr_5)
        lst_hr_10.append(hr_10)
        lst_ndcg_5.append(ndcg_5)
        lst_ndcg_10.append(ndcg_10)

        for i in u_pred_5:
            coverage_5.add(i)
        for i in u_pred_10:
            coverage_10.add(i)
        novel_5.append(getNovel(u_pred_5, cold_items, tail_items))
        novel_10.append(getNovel(u_pred_10, cold_items, tail_items))

    return sum(lst_hr_5)/len(lst_hr_5), sum(lst_ndcg_5)/len(lst_ndcg_5), sum(lst_hr_10)/len(lst_hr_10), sum(lst_ndcg_10)/len(lst_ndcg_10), len(coverage_5)/item_num, len(coverage_10)/item_num, sum(novel_5)/len(novel_5), sum(novel_10)/len(novel_10)

def cal_item_novel(train_item_list):
    theta_i = np.zeros(item_num)
    for i in range(item_num):
        len_interaction_i = len(train_item_list[i])
        if len_interaction_i > 0:
            theta_i[i] = np.log(user_num / len_interaction_i)
    for i in range(item_num):
        if theta_i[i] == 0:     #冷启动的item novelty设置最大
            theta_i[i] = max(theta_i)
    return theta_i

def cal_user_novel(theta_i, train_user_list):
    theta_u, sigma_u = np.zeros(user_num), np.zeros(user_num)
    for u in range(user_num):
        tmp_lst = list(train_user_list[u])
        if len(tmp_lst) > 0:
            theta_ilist = theta_i[tmp_lst]
            tmp_theta = np.mean(theta_ilist)
            tmp_sigma = np.sqrt(np.sum((theta_ilist - tmp_theta) ** 2) / len(tmp_lst))
            theta_u[u] = tmp_theta
            sigma_u[u] = tmp_sigma
    return theta_u, sigma_u

def rbf_kernel(t_u, s_u, t_i, lamda=2):
    return np.exp(-abs(t_u-t_i)**2 / (2 * lamda * (s_u+1e-6)**2))

def cal_pointwise_weight(batch_u, batch_i, theta_u, sigma_u, theta_i):
    batch_theta_u, batch_sigma_u = theta_u[batch_u], sigma_u[batch_u]
    batch_theta_i = theta_i[batch_i]
    batch_pai_ui = rbf_kernel(batch_theta_u, batch_sigma_u, batch_theta_i)
    weight = (batch_pai_ui - min(batch_pai_ui)) / (max(batch_pai_ui) - min(batch_pai_ui))
    #weight = np.where(weight > 0, weight, 0)
    #weight = np.nan_to_num(weight, posinf=0, neginf=0)
    return weight

def func_cal_sim_item(item_all_feat_arr, item_num, feat_num):
    row_idx, col_idx = [], []
    for i in range(item_all_feat_arr.shape[0]):
        for each_feat in item_all_feat_arr[i]:
            if each_feat != 0:
                row_idx.append(i)
                col_idx.append(each_feat)
    item_all_feat_sp = sp.csr_matrix((np.ones_like(row_idx), (row_idx, col_idx)), (item_num, feat_num))
    item_item_sp = item_all_feat_sp.dot(item_all_feat_sp.T)
    item_item_nonzero = item_item_sp.nonzero()

    item_item_sim = [[] for i in range(item_num)]
    for i,j in zip(item_item_nonzero[0], item_item_nonzero[1]):
        item_item_sim[i].append(j)
    return item_item_sim

if __name__ == '__main__':
    #random.seed(2019)
    #np.random.seed(2019)
    
    accuracy_List = [5, 10]
    recall_dictionary = dict()
    coverage_dictionary = dict()
    for i in accuracy_List:
        recall_dictionary["recall@"+str(i)] = Recall(i)
        coverage_dictionary["cov@"+str(i)] = Coverage(i)
        
        
    
    
    

    f_data = '../movielens_100k/movielens100k_longtail_data.pkl'
    #f_data = '../music/music_longtail_data.pkl'
    #f_data = '../beauty/beauty_longtail_data.pkl'

    with open(f_data, 'rb') as f:
        dataset = pickle.load(f)
        user_num, item_num = dataset['user_size'], dataset['item_size']
        # user_size -- number of users data type: int
        # item_size -- number of items data type: int
        # list of cold_items, list of head items, list of tail items
        cold_items, head_items, tail_items = dataset['cold_items'], dataset['head_items'], dataset['tail_items']
        # cold_items (Extreme tail items): select cold items data type: list
        # head_items: selected head items (top 20% popular items) datatype: list
        # tail_items:selected tail items data type: list
        
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']

        hot_items = np.zeros([item_num])
        hot_items[np.array(head_items)] = 1

        coldstart_items = np.zeros([item_num])
        coldstart_items[np.array(cold_items)] = 1

        feat_num = dataset['feat_size']
        item_all_feat_dict = dataset['item_all_feat_dict']
        item_all_feat_arr = list(item_all_feat_dict.values())
        max_len_item_feat = max([len(lst) for lst in item_all_feat_arr])
        item_all_feat_arr = [lst+[0]*(max_len_item_feat-len(lst)) for lst in item_all_feat_arr]
        item_all_feat_arr = np.array(item_all_feat_arr, dtype=int)

        user_feat_num = dataset['user_feat_size']
        user_all_feat_dict = dataset['user_all_feat_dict']
        user_all_feat_arr = np.array(list(user_all_feat_dict.values()), dtype=int)

        train_item_list = [[] for i in range(item_num)]
        for u, i in train_pair:
            train_item_list[i].append(u)

    

    theta_i = cal_item_novel(train_item_list)
    norm_theta_i = (theta_i - min(theta_i)) / (max(theta_i) - min(theta_i))
    theta_u, sigma_u = cal_user_novel(theta_i, train_user_list)

    item_item_sim = func_cal_sim_item(item_all_feat_arr[:, 1:], item_num, feat_num)
    # NCF model
    #model = NCF_Bi_Disen8(user_num, item_num, user_feat_num, feat_num, item_all_feat_arr.shape[1], args.dim)
    # LFM model
    model = LFM_Bi_Disen8(user_num, item_num, user_feat_num, feat_num, item_all_feat_arr.shape[1], args.dim)
    
    #model.cuda()
    loss_function = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    
    for i in train_user_list:
        if (len(i)) == 0:
            continue
        for key in coverage_dictionary:
            coverage_dictionary[key].cal_unique_items(i)
            
    for epoch in tqdm(range(50)):
        print('#' * 80)
        print('Epoch {} starts !'.format(epoch))
        start_t = time.time()
        train_loader = instance_a_train_loader(user_num, item_num, train_user_list, args.num_neg, args.batch_size, cold_items)
        
        curr_item_not_sim = []
        for i in range(item_num):
            candidate_comp_neg = list(set(range(1, item_num)) - set(item_item_sim[i]))
            if len(candidate_comp_neg) > 0:
                comp_neg = np.random.choice(list(set(range(1, item_num)) - set(item_item_sim[i])))
            else:
                comp_neg = np.random.choice(range(1, item_num))
            curr_item_not_sim.append(comp_neg)
        curr_item_not_sim = np.array(curr_item_not_sim, dtype=int)

        model.train()
        total_loss = 0
        #--------------------------train
        for batch_id, batch in enumerate(tqdm(train_loader)):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            # batch_w = batch[3].cuda()
            h_item = torch.FloatTensor(hot_items[item])
            rating = rating.float()
            batch_w = torch.FloatTensor(norm_theta_i[item.cpu().numpy()])

            optimizer.zero_grad()
            batch_item_feat = torch.LongTensor(item_all_feat_arr[item])
            batch_user_feat = torch.LongTensor(user_all_feat_arr[user])
            batch_comp_neg = curr_item_not_sim[item]

            ratings_pred_hot, ratings_pred_cold, ratings_pred_joint, item_vae_loss = model(user, item, batch_user_feat, batch_item_feat, torch.LongTensor(batch_comp_neg))

            loss = F.binary_cross_entropy(ratings_pred_hot.view(-1), Variable(rating), weight=1 - batch_w) + F.binary_cross_entropy(ratings_pred_cold.view(-1),  Variable(rating), weight=batch_w) + args.lamda * (item_vae_loss / user.shape[0]) #①
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print( 'loss: %f' % (total_loss / (batch_id+1)))
        #--------------------------test
        model.eval()
        model.zero_grad()
       
        
    for u in range(1, user_num):
        if len(test_user_list[u]) == 0:
            continue

        pred = []
        user = torch.LongTensor(np.array([u]))
        item = torch.LongTensor(np.array(range(item_num)))

        c_item = coldstart_items[item]
        batch_item_feat = torch.LongTensor(item_all_feat_arr[item])
        batch_user_feat = torch.LongTensor([user_all_feat_arr[user]])

        ratings_pred_hot, _, _ = model.predict(user, item, batch_user_feat, batch_item_feat, torch.FloatTensor(c_item))
        ratings_pred_hot = (ratings_pred_hot.view(-1)).cpu().detach().numpy()
        
        # remove already seen items
        idx = np.zeros_like(ratings_pred_hot, dtype=bool)
        for i_idx in train_user_list[u]:
            idx[i_idx] = True
        ratings_pred_hot[idx] = -np.inf
        ratings_pred_hot[0] = -np.inf
        pred = np.argsort(-ratings_pred_hot)

        
        for key in recall_dictionary:
            recall_dictionary[key].add(test_user_list[u], pred)
            
        
        for key in coverage_dictionary:
            coverage_dictionary[key].unique_items_appear_retriveList(pred)
        
        
        
    for key in recall_dictionary:
        print(    str(key) +" = "                 +str(recall_dictionary[key].score()))
        
    for key in coverage_dictionary:
        print(    str(key) +" = "                 +str(coverage_dictionary[key].score()))