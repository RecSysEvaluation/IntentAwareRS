import torch.optim as optim
import random
import logging
import datetime
import os
from topn_baselines_neurals.Recommenders.BIGCF.utility.parser import parse_args
from topn_baselines_neurals.Recommenders.BIGCF.utility.batch_test import *
from topn_baselines_neurals.Recommenders.BIGCF.utility.load_data import *
from topn_baselines_neurals.Recommenders.BIGCF.BIGCF import *
from tqdm import tqdm
import time
from copy import deepcopy

args = parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.epoch = 0
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epoch = epoch
            self.counter = 0

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)


def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list

def model_tuningAndTraining(dataset_name = "gowalla", path = "", validation = False, epoch = 500, ssl_reg = 0.4, ks = [20, 40], NumberOfUserInTestingData = 50000):
    if not os.path.exists('log'):
        os.mkdir('log')
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler('log/{}.log'.format(args.dataset), 'a', encoding='utf-8')
    logfile.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    args.dataset = dataset_name
    args.epoch = epoch
    args.data_path = path
    args.ssl_reg = ssl_reg
    args.Ks = ks

    """
    *********************************************************
    Prepare the dataset
    """
    data_generator = Data(args, validation = validation)
    logger.info(data_generator.get_statistics())

    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    logger.info(args)
    print("************************************************************************************")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the adj matrix
    """
    plain_adj = data_generator.get_adj_mat()
    all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
    config['plain_adj'] = plain_adj
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list

    """
    *********************************************************
    Prepare the model and start training
    """
    _model = BIGCF(config, args).cuda()
    optimizer = optim.Adam(_model.parameters(), lr=args.lr)

    print("Start Training")
    if validation == True:
        print("Start Early Stopping mechanism to get best epoch values")
        earlystopping = EarlyStopping(patience=args.patience)

    start = time.time()
    for epoch in range(args.epoch):
        print("Epoch number: "+str(epoch))
        n_samples = data_generator.uniform_sample()
        n_batch = int(np.ceil(n_samples / args.batch_size))

        _model.train()
        loss, mf_loss, emb_loss, cen_loss, cl_loss = 0., 0., 0., 0., 0.

        for idx in tqdm(range(n_batch)):
            optimizer.zero_grad()
            users, pos_items, neg_items = data_generator.mini_batch(idx)
            batch_mf_loss, batch_emb_loss, batch_cen_loss, batch_cl_loss = _model(users, pos_items, neg_items)
            batch_loss = batch_mf_loss + batch_emb_loss + batch_cen_loss + batch_cl_loss
            loss += float(batch_loss) / n_batch
            mf_loss += float(batch_mf_loss) / n_batch
            emb_loss += float(batch_emb_loss) / n_batch
            cen_loss += float(batch_cen_loss) / n_batch
            cl_loss += float(batch_cl_loss) / n_batch

            batch_loss.backward()
            optimizer.step()
        
        if validation == True:

            with torch.no_grad():
                _model.eval()
                _model.inference()
                final_test_ret = eval_PyTorch(_model, data_generator, eval(args.Ks))
                torch.cuda.empty_cache()
            recall = final_test_ret["Recall@20"].getScore()

            print ("Patience value: ", str(earlystopping.patience), "Counter value: ", str(earlystopping.counter),
                    " Best Previous Recall Score: ",str(earlystopping.best_score), " Current Recall:", str(recall) )
            
            earlystopping(recall, epoch)
            if earlystopping.early_stop:
                return earlystopping.epoch + 1
            
    time_dictionary = dict()
    training_time = time.time() - start

    if validation == True:
        return args.epoch
    else:
        with torch.no_grad():
            start = time.time()
            _model.eval()
            _model.inference()
            final_test_ret = eval_PyTorch(_model, data_generator, eval(args.Ks))
            users = len(list(data_generator.test_set.keys()))
            test_time = time.time() - start
            time_dictionary["trainingTime"] = training_time
            time_dictionary["testingTime"] = test_time
            time_dictionary["AverageTestTimePerUser"] = test_time / NumberOfUserInTestingData

            temp_dict = {}
            for key, value in final_test_ret.items():
                temp_dict[key] = final_test_ret[key].getScore()
            final_test_ret = {**temp_dict, **time_dictionary}
            
        return final_test_ret

    