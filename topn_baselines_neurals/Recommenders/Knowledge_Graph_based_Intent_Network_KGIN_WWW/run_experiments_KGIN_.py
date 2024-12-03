'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

import torch
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import argparse
from topn_baselines_neurals.Recommenders.Knowledge_Graph_based_Intent_Network_KGIN_WWW.utils.data_loader import load_data
from topn_baselines_neurals.Recommenders.Knowledge_Graph_based_Intent_Network_KGIN_WWW.modules.KGIN import Recommender
from topn_baselines_neurals.Recommenders.Knowledge_Graph_based_Intent_Network_KGIN_WWW.utils.evaluate import model_evaluation


def get_feed_dict(train_entity_pairs, start, end, train_user_set, n_items):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict
def parse_args(dataset, dim, lr, sim_regularity, batch_size, node_dropout, node_dropout_rate, mess_dropout, mess_dropout_rate, gpu_id, context_hops, epoch):
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default=dataset, help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=epoch, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=dim, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=lr, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=sim_regularity, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=node_dropout, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=node_dropout_rate, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=mess_dropout, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=mess_dropout_rate, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=gpu_id, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=context_hops, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()


def run_experiments_KGIN_model(dataset, dim=64, lr= 0.0001, sim_regularity=0.0001, batch_size=1024,
                      node_dropout=True, node_dropout_rate=0.5, mess_dropout=True, mess_dropout_rate=0.1, gpu_id=0, context_hops=3, 
                      epoch = 60, lastFMDataLeakage = False, datasetName = None):
    
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    global args, device

    args = parse_args(dataset, dim, lr, sim_regularity, batch_size, node_dropout, node_dropout_rate, mess_dropout, mess_dropout_rate, gpu_id, context_hops, epoch)
    """read args"""
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    """build dataset"""
    
    train_cf, test_cf, user_dict, n_params, graph, mat_list, userWithDataLeakage = load_data(args, dataset, lastFMDataLeakage, datasetName)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    
    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("start training ...")
    print("Number of epoch:  "+str(args.epoch))
    start = time.time()
    for epoch in tqdm(range(args.epoch)):
        print("Current epoch:  "+str(epoch))
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        loss, s, cor_loss = 0, 0, 0
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'], n_items)
            batch_loss, _, _, batch_cor = model(batch)
            
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size

    trainingTime = time.time() - start
    start = time.time()
    result_dict = model_evaluation(model, user_dict, n_params, userWithDataLeakage, lastFMDataLeakage)
    testingTime = time.time() - start

    result_df = pd.DataFrame()
    for key in result_dict:
            print(result_dict[key].getScore())
            result_df[key] = [result_dict[key].getScore()]

    result_df["TrainingTime"] = [trainingTime]
    result_df["TestingTimeforAllUser"] = [testingTime]
    result_df["AverageTestingPerUser"] = [testingTime  / len(user_dict["test_user_set"])]

    return result_df         

            

            