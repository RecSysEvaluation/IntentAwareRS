import pandas as pd
import numpy as np
import random
import time
import pickle
from pathlib import Path
import argparse
import scipy.sparse as sp
from topn_baselines_neurals.Recommenders.IDS4NR.data_utils import UserItemRatingDataset, instance_a_train_loader, instance_a_eval_loader, instance_u_eval_loader_all, instance_a_train_loader_weight
from topn_baselines_neurals.Recommenders.IDS4NR.LFM_Bi_Disen import LFM_Bi_Disen
from topn_baselines_neurals.Recommenders.IDS4NR.NCF_Bi_Disen import NCF_Bi_Disen
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from topn_baselines_neurals.Recommenders.IDS4NR.accuracy_measures import *
from topn_baselines_neurals.Recommenders.IDS4NR.INS4NR_dataprocessing import Data_
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from functools import partial
import time
from skopt import forest_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from skopt.learning.gaussian_process.kernels import RBF, ConstantKernel as C

import os
import sys
current_module = sys.modules[__name__]
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IDSNR_model:
    def __init__(self, model = "NCF",  data = "Beauty", path = "",  validation = True, cutt_off_values = [5, 10, 20], optimization_matric = "Recall", optimization_cutt_off = 10, optimization_calls = 35, random_states_calls = 5):
        
        self.model = model
        self.validation = validation
        self.evaluation_obj = Data_(data = data, validation = validation) # data MovieLens, Music, Beauty
        
        if self.validation:
            self.optimization_matric = optimization_matric
            self.optimization_cutt_off = optimization_cutt_off
            self.optimization_calls = optimization_calls
            self.random_states_calls = random_states_calls
            self.saved_optimized_results = pd.DataFrame()
            
        accuracy_measure = ["Recall", "Precision"]
        self.accuracy_measure_dict = dict()
        for met_ in accuracy_measure:
            for cutt_ in cutt_off_values:
                self.accuracy_measure_dict[met_+"@"+str(cutt_)] = getattr(current_module, met_)(cutt_)
                
        self.beyond_accuracy_measure_dict = dict()        
        beyond_accuracy_measure = ["Coverage", "Novelty"]
        for met_ in beyond_accuracy_measure:
            for cutt_ in cutt_off_values:
                self.beyond_accuracy_measure_dict[met_+"@"+str(cutt_)] = getattr(current_module, met_)(cutt_, self.evaluation_obj.train_user_list)


    def model_tuninig(self, epoch):

        if self.model == "NCF":
            print("Model name:  "+self.model)
            model = NCF_Bi_Disen(self.evaluation_obj.user_num, self.evaluation_obj.item_num, self.evaluation_obj.user_feat_num, self.evaluation_obj.feat_num, self.evaluation_obj.item_all_feat_arr.shape[1], 100)
            learning_rate = 0.001
            negative_sampling = 4
            batch_size = 128
            
        elif self.model == "LFM":
            print("Model name:  LFM")
            model = LFM_Bi_Disen(self.evaluation_obj.user_num, self.evaluation_obj.item_num, self.evaluation_obj.user_feat_num, self.evaluation_obj.feat_num, self.evaluation_obj.item_all_feat_arr.shape[1], 100)
            learning_rate = 0.001
            negative_sampling = 4
            batch_size = 128

        else:
            print("Please write correct name of either NCF or LFM")
        
        model.to(device)
        loss_function = torch.nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        train_loader = instance_a_train_loader(self.evaluation_obj.user_num, self.evaluation_obj.item_num, self.evaluation_obj.train_user_list, negative_sampling, batch_size, self.evaluation_obj.cold_items)
        if self.validation == True:
            best_score, stopped_epoch = self.fit(model, train_loader, epoch, optimizer)
            return best_score, stopped_epoch
        else:
            model = self.fit(model, train_loader, epoch, optimizer)
            return model
        
    def fit(self, model, train_loader, epoch, optimizer):
        if self.validation:
            early_stopping = EarlyStopping()
            
        for epoch in tqdm(range(epoch)):
            curr_item_not_sim = []
            for i in range(self.evaluation_obj.item_num):
                candidate_comp_neg = list(set(range(1, self.evaluation_obj.item_num)) - set(self.evaluation_obj.item_item_sim[i]))
                if len(candidate_comp_neg) > 0:
                    comp_neg = np.random.choice(list(set(range(1, self.evaluation_obj.item_num)) - set(self.evaluation_obj.item_item_sim[i])))
                else:
                    comp_neg = np.random.choice(range(1, self.evaluation_obj.item_num))
                curr_item_not_sim.append(comp_neg)
            curr_item_not_sim = np.array(curr_item_not_sim, dtype=int)
            model.train()
            total_loss = 0
            
            #--------------------------train--------------------
            for batch_id, batch in enumerate(train_loader):
                assert isinstance(batch[0], torch.LongTensor)
                user, item, rating = batch[0], batch[1], batch[2]
                #batch_w = batch[3].cuda()
                rating = rating.float()
                batch_w = torch.FloatTensor(self.evaluation_obj.norm_theta_i[item.cpu().numpy()])

                optimizer.zero_grad()
                batch_item_feat = torch.LongTensor(self.evaluation_obj.item_all_feat_arr[item])
                
                batch_user_feat = torch.LongTensor(self.evaluation_obj.user_all_feat_arr[user]).to(device)
                batch_comp_neg = curr_item_not_sim[item]
                batch_comp_neg = torch.LongTensor(batch_comp_neg).to(device)
                ratings_pred_hot, ratings_pred_cold, ratings_pred_joint, item_vae_loss = model(user.to(device), item.to(device), batch_user_feat, batch_item_feat.to(device), batch_comp_neg)
                loss = F.binary_cross_entropy(ratings_pred_hot.view(-1), Variable(rating.to(device)), weight=1 - batch_w.to(device)) + F.binary_cross_entropy(ratings_pred_cold.view(-1),  Variable(rating.to(device)), weight=batch_w.to(device)) + 0.01 * (item_vae_loss / user.shape[0]) #①
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if self.validation:
                model.eval()
                model.zero_grad()
                for u in range(len(self.evaluation_obj.test_user_list)):
                    if len(self.evaluation_obj.test_user_list[u]) == 0:
                        continue
                    pred = []
                    user = torch.LongTensor(np.array([u]))
                    item = torch.LongTensor(np.array(range(self.evaluation_obj.item_num)))
    
                    c_item = self.evaluation_obj.coldstart_items[item]
                    batch_item_feat = torch.LongTensor(self.evaluation_obj.item_all_feat_arr[item])
                    batch_user_feat = torch.LongTensor([self.evaluation_obj.user_all_feat_arr[user]])
                    
                    ratings_pred_hot, _, _ = model.predict(user.to(device), item.to(device), batch_user_feat.to(device), batch_item_feat.to(device), torch.FloatTensor(c_item).to(device))
                    ratings_pred_hot = (ratings_pred_hot.view(-1)).cpu().detach().numpy()
                    
                    # remove already seen items
                    idx = np.zeros_like(ratings_pred_hot, dtype=bool)
                    for i_idx in self.evaluation_obj.train_user_list[u]:
                        idx[i_idx] = True
                    ratings_pred_hot[idx] = -np.inf
                    ratings_pred_hot[0] = -np.inf
                    pred = np.argsort(-ratings_pred_hot)
                    for key in self.accuracy_measure_dict:
                        self.accuracy_measure_dict[key].add(self.evaluation_obj.test_user_list[u], pred)
                optimize_value = self.accuracy_measure_dict[self.optimization_matric+"@"+str(self.optimization_cutt_off)].getScore()
                early_stopping(optimize_value, epoch+1)
            
                if early_stopping.early_stop:
                    break
            
        if self.validation:    
            return early_stopping.best_score, epoch
        
        # if optimization is not true then simply train a model and returns        
        return model
        
    def predict_(self, model):
        
        model.eval()
        model.zero_grad()
        for u in range(len(self.evaluation_obj.test_user_list)):
            if len(self.evaluation_obj.test_user_list[u]) == 0:
                continue
            pred = []
            user = torch.LongTensor(np.array([u]))
            item = torch.LongTensor(np.array(range(self.evaluation_obj.item_num)))

            c_item = self.evaluation_obj.coldstart_items[item]
            batch_item_feat = torch.LongTensor(self.evaluation_obj.item_all_feat_arr[item])
            batch_user_feat = torch.LongTensor([self.evaluation_obj.user_all_feat_arr[user]])
            
            ratings_pred_hot, _, _ = model.predict(user.to(device), item.to(device), batch_user_feat.to(device), batch_item_feat.to(device), torch.FloatTensor(c_item).to(device))
            ratings_pred_hot = (ratings_pred_hot.view(-1)).cpu().detach().numpy()
            
            # remove already seen items
            idx = np.zeros_like(ratings_pred_hot, dtype=bool)
            for i_idx in self.evaluation_obj.train_user_list[u]:
                idx[i_idx] = True
            ratings_pred_hot[idx] = -np.inf
            ratings_pred_hot[0] = -np.inf
            pred = np.argsort(-ratings_pred_hot)
            
            for key in self.accuracy_measure_dict:
                self.accuracy_measure_dict[key].add(self.evaluation_obj.test_user_list[u], pred.copy())
                
            for key in self.beyond_accuracy_measure_dict:
                self.beyond_accuracy_measure_dict[key].add(pred.copy())
            
            
# class to implement Early stopping only on validation data....
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.stopped_epoch = 0

    def __call__(self, val_accuracy, current_epoch):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = current_epoch
        else:
            self.best_score = score
            self.counter = 0

class Run_experiments_for_IDSNR:

    def __init__(self,model = "NCF", dataset = "MovieLens", NumberOfUsersInTestingData = 943):
        # get best epoch through stopping mechanism
        obj = IDSNR_model(model = model, data = dataset, validation = True)
        best_score, stopped_epoch =  obj.model_tuninig(100)
        stopped_epoch = stopped_epoch + 1
        print("Best Epoch Value    "+ str(stopped_epoch))
        
        start = time.time()
        obj = IDSNR_model(model = model,data = dataset, validation = False)
        model = obj.model_tuninig(stopped_epoch)
        trainingTime = time.time() - start

        start = time.time()
        obj.predict_(model)
        testingTime = time.time() - start
        df = pd.DataFrame()

        for key in obj.accuracy_measure_dict:
            print(key +"  "+  str(obj.accuracy_measure_dict[key].getScore())) 
            df[key] = [obj.accuracy_measure_dict[key].getScore()]
        for key in obj.beyond_accuracy_measure_dict:
            print(key +"  "+  str(obj.beyond_accuracy_measure_dict[key].getScore())) 
            df[key] = [obj.beyond_accuracy_measure_dict[key].getScore()]
        
        df["TrainingTime(s)"] = [trainingTime]
        df["TestingTime(s)"] = [testingTime]
        df["AverageTestingTime(ms)"] = [(testingTime / NumberOfUsersInTestingData) * 1000]
        self.accuracy_values  = df
        





        
        








