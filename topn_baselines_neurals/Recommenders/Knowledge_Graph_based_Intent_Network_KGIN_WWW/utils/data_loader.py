import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list

def read_cf_avoid_dataLeakage(pathtrain, pathtest):
    inter_mat = list()
    train_dictionary = dict()
    test_dictionary = dict()
    lines = open(pathtrain, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        train_dictionary[u_id] = set(pos_ids)
    
    lines = open(pathtest, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        test_dictionary[u_id] = set(pos_ids)
    keys_with_dataLeakage = [key for key, item in train_dictionary.items() if len(test_dictionary[key].intersection(train_dictionary[key])) > 0]

    keys_itemLisDataLeakage = dict()
    for key in keys_with_dataLeakage: # combine the interactions of users where we observe a data leakage issue
        keys_itemLisDataLeakage[key] = train_dictionary[key].union(test_dictionary[key])
    
    keyToRemove  = [key for key, items in keys_itemLisDataLeakage.items() if len(items) < 2]
    for key in keyToRemove:
        del keys_itemLisDataLeakage[key]

    new_train, new_test = dataSplitingDataLeakage(keys_itemLisDataLeakage) 
    ############
    for key, _ in new_train.items():
        train_dictionary[key] = new_train[key]
        test_dictionary[key] = new_test[key]

    inter_mat_train = list()
    for u_id, pos_ids in train_dictionary.items():
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat_train.append([u_id, i_id])
    
    inter_mat_test = list()
    for u_id, pos_ids in test_dictionary.items():
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat_test.append([u_id, i_id])

    return np.array(inter_mat_train), np.array(inter_mat_test), keyToRemove
# data spliting.....
def dataSplitingDataLeakage(keys_itemLisDataLeakage):
    new_train, new_test = dict(), dict()
    for key, items in keys_itemLisDataLeakage.items():
        temp_list = list(items)
        if len(temp_list) < 5:
           new_train[key] =  set(temp_list[:-1])
           new_test[key] =   set([temp_list[-1]])
        else:
            selectedRatio = int(len(temp_list) * 0.2)
            new_train[key] =  set(temp_list[:-selectedRatio])
            new_test[key] =   set(temp_list[-selectedRatio:])
    return new_train, new_test

    
def load_data(model_args, datapath, lastFMDataLeakage, datasetName):
    global args
    args = model_args
    directory = datapath
    userWithDataLeakage = None
    print('reading train and test user-item set ...')
    if lastFMDataLeakage == True and datasetName == "lastFm":
        train_cf, test_cf, userWithDataLeakage = read_cf_avoid_dataLeakage(directory / 'train.txt', directory / 'test.txt')
    else:

        train_cf = read_cf(directory / 'train.txt')
        test_cf = read_cf(directory / 'test.txt')

    remap_item(train_cf, test_cf)
    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory / 'kg_final.txt')
    
    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }

    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }
    
    return train_cf, test_cf, user_dict, n_params, graph, [adj_mat_list, norm_mat_list, mean_mat_list], userWithDataLeakage
    

