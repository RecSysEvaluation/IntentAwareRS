
from topn_baselines_neurals.Data_manager.Gowalla_AmazonBook_Tmall_DCCF import Gowalla_AmazonBook_Tmall_DCCF 
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Recommenders.BIGCF.BIGCF_main import *
from pathlib import Path
import pandas as pd
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='gowalla', help="amazonBook / gowalla / tmall")
    parser.add_argument('--epoch', type = int, default=200, help="Number Of epoch")
    parser.add_argument('--ssl_reg', type=float, default=0.2, help='Reg weight for ssl loss')
    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 20, 40, 50, 100]', help='Metrics scale')
    parser.add_argument('--ES', type=bool, default= False, help='Metrics scale')
    args = parser.parse_args()
    
    if args.dataset == "gowalla":
        args.epoch = 150
        args.ssl_reg = 0.4
    elif args.dataset == "amazonBook":
        args.epoch = 200
        args.ssl_reg = 0.4
    elif args.dataset == "tmall":
        args.epoch = 200
        args.ssl_reg = 0.2
    else:
        pass
    dataset_name = args.dataset
    commonFolderName = "results"
    #### DCCF and BIGCF are using same train test splits.....
    model = "BIGCF"
    saved_results = "/".join([commonFolderName, model] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results>>>>>>>>>>>>>>>")
    data_path = Path("data/DCCF/"+dataset_name)
    data_path = data_path.resolve()
    ############### BASELINE MODELS DATA PREPARATION ###############
    dataset_object = Gowalla_AmazonBook_Tmall_DCCF()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path)
    ICM_all = None
    UCM_all = None
    NumberOfUserInTestingData = URM_test.shape[0] - np.sum(np.diff(URM_test.indptr) == 0)
    ############### END #############################################
    if args.ES:
        best_epoch = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=True, epoch = args.epoch, ssl_reg = args.ssl_reg, ks = args.Ks)
        print("Start tuning by Best Epoch Value"+str(best_epoch))
        metrics_dic = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=False, 
                                                            epoch = best_epoch, ssl_reg = args.ssl_reg, ks = args.Ks, NumberOfUserInTestingData = NumberOfUserInTestingData)
    else:
        metrics_dic = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=False, 
                                                            epoch = args.epoch, ssl_reg = args.ssl_reg, ks = args.Ks, NumberOfUserInTestingData = NumberOfUserInTestingData)
    
    expanded_data = [(key, value) for key, value in metrics_dic.items()]
    df = pd.DataFrame(expanded_data, columns=['Measures', 'Values'])
    df.to_csv(saved_results + "/"+args.dataset+"_BIGCF.txt", index = False)
    
    
    