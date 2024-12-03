# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:35:03 2024

@author: shefai
"""

import numpy as np
import pickle
import scipy.sparse as sp

class Data_:
    def __init__(self, data = "Beauty", validation = True):
        with open(data, 'rb') as f:
            dataset = pickle.load(f)
        self.user_num, self.item_num = dataset['user_size'], dataset['item_size']   
        self.cold_items, head_items, _ = dataset['cold_items'], dataset['head_items'], dataset['tail_items']
        self.train_user_list, self.test_user_list = dataset['train_user_list'], dataset['test_user_list']
        self.train_pair = dataset['train_pair']
        
        if validation == True:
            self.train_user_list, self.test_user_list, self.train_pair = self.validation_splitting(self.train_user_list, 0.10)
        self.hot_items = np.zeros([self.item_num])
        self.hot_items[np.array(head_items)] = 1

        self.coldstart_items = np.zeros([self.item_num])
        self.coldstart_items[np.array(self.cold_items)] = 1

        self.feat_num = dataset['feat_size']
        item_all_feat_dict = dataset['item_all_feat_dict']
        self.item_all_feat_arr = list(item_all_feat_dict.values())
        max_len_item_feat = max([len(lst) for lst in self.item_all_feat_arr])
        self.item_all_feat_arr = [lst+[0]*(max_len_item_feat-len(lst)) for lst in self.item_all_feat_arr]
        self.item_all_feat_arr = np.array(self.item_all_feat_arr, dtype=int)

        self.user_feat_num = dataset['user_feat_size']
        user_all_feat_dict = dataset['user_all_feat_dict']
        self.user_all_feat_arr = np.array(list(user_all_feat_dict.values()), dtype=int)

        train_item_list = [[] for i in range(self.item_num)]
        for u, i in self.train_pair:
            train_item_list[i].append(u)

        theta_i = self.cal_item_novel(train_item_list, self.item_num, self.user_num)
        self.norm_theta_i = (theta_i - min(theta_i)) / (max(theta_i) - min(theta_i))
        self.item_item_sim = self.func_cal_sim_item(self.item_all_feat_arr[:, 1:], self.item_num, self.feat_num)
        
    
    def validation_splitting(self, train_user_list, validation_portion):
        train_tr, validation_tr = list(), list()
        train_tr.append(set())
        validation_tr.append(set())
        
        for item_set in train_user_list:
            if len(item_set) == 0:
                continue
            item_list = list(item_set)
            train_tr.append(  set(  item_list[:-2]   )  )
            validation_tr.append(  set(  item_list[-2:]   )  )
            
        new_train_pairs = self.make_training_pairs(train_tr)
        return train_tr, validation_tr, new_train_pairs
            
    
    def make_training_pairs(self,train_tr):
        
        new_train_pairs = list()
        user = 1
        for item_set in train_tr:
            if len(item_set) == 0:
                continue
            
            for item_ in item_set:
                new_train_pairs.append(  (user, item_)  )
                
            user+=1   
        return new_train_pairs

    def cal_item_novel(self, train_item_list, item_num, user_num):
        theta_i = np.zeros(item_num)
        for i in range(item_num):
            len_interaction_i = len(train_item_list[i])
            if len_interaction_i > 0:
                theta_i[i] = np.log(user_num / len_interaction_i)
        for i in range(item_num):
            if theta_i[i] == 0:     #冷启动的item novelty设置最大
                theta_i[i] = max(theta_i)
        return theta_i


    def func_cal_sim_item(self,item_all_feat_arr, item_num, feat_num):
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
    