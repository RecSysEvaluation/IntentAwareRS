
from topn_baselines_neurals.Recommenders.DCCF.DCCF_main import *
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Data_manager.Gowalla_AmazonBook_Tmall_DCCF import Gowalla_AmazonBook_Tmall_DCCF 
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
import traceback, os
from pathlib import Path
import argparse
import pandas as pd
import time
import ast


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)
    return recommender_object

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='gowalla', help="tmall / gowalla / amazonBook")
    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 20, 40, 50, 100]', help='Metrics scale')
    args = parser.parse_args()

    dataset_name = args.dataset
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    data_path = Path("data/DCCF/"+dataset_name)
    data_path = data_path.resolve()
    commonFolderName = "results"
    model = "DCCF"
    saved_results = "/".join([commonFolderName, model] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    ############### BASELINE MODELS DATA PREPARATION ###############
    validation_set = False
    dataset_object = Gowalla_AmazonBook_Tmall_DCCF()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path, validation=validation_set)
    ICM_all = None
    UCM_all = None
    NumberOfUserInTestingData = URM_test.shape[0] - np.sum(np.diff(URM_test.indptr) == 0)
    ############### END #############################################
    
    ############### RUN EXPERIMENTS FOR DCCF ########################
    
    best_epoch = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=True, epoch = 500, ks = args.Ks, NumberOfUserInTestingData = 0)
    print("Start tuning by Best Epoch Value"+str(best_epoch))
    metrics_dic = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=False, epoch = 
                                                           best_epoch , ks = args.Ks, NumberOfUserInTestingData = NumberOfUserInTestingData)
    
    for key, value in metrics_dic.items():
        print(str(key)+": "+str(value))
    
    expanded_data = [(key, value) for key, value in metrics_dic.items()]
    df = pd.DataFrame(expanded_data, columns=['Measures', 'Values'])
    df.to_csv(saved_results + "/"+args.dataset+"_DCCF.txt", index = False)
    ############### END ############################################
    
    
    ############### RUN EXPERIMENTS FOR BASELINE MODELS ########################
    total_elements = URM_train.shape[0] * URM_train.shape[1]
    non_zero_elements = URM_train.nnz
    density = non_zero_elements / total_elements
    print("Number of users: %s, Items: %d, Interactions: %d, Density %.5f, Number of users with no test items: %d." % 
          (URM_train.shape[0], URM_train.shape[1], non_zero_elements, density, np.sum(np.diff(URM_test.indptr) == 0)))
    
    recommender_class_list = [
        Random,
        TopPop,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        EASE_R_Recommender
        ]
    
    ##### Best HP values for baseline models.....
    if args.dataset == "gowalla":
        itemkNN_best_HP  = {"topK": 508, "similarity": "cosine"}
        userkNN_best_HP  = {"topK": 146, "similarity": "cosine"}
        RP3alpha_best_HP = {"topK": 777, "alpha": 1.087096950563704, "normalize_similarity": False}
        RP3beta_best_HP  = {"topK": 777, "alpha": 0.5663562161452378, "beta": 0.001085447926739258, "normalize_similarity": True}
        
    elif args.dataset == "tmall":
        itemkNN_best_HP = {"topK": 516, "similarity": "cosine"}
        userkNN_best_HP = {"topK": 454, "similarity": "cosine"}
        RP3alpha_best_HP = {"topK": 100, "alpha": 1, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 350, "alpha": 0.7681732734954694, "beta": 0.4181395996963926, "normalize_similarity": True}
        
    elif args.dataset == "amazonBook":
        itemkNN_best_HP = {"topK": 125, "similarity": "cosine"}
        userkNN_best_HP = {"topK": 454, "similarity": "cosine"}
        RP3alpha_best_HP = {"topK": 496, "alpha": 0.41477903655656115, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 496, "alpha": 0.44477903655656115, "beta": 0.5968193614337285, "normalize_similarity": True}

        recommender_class_list = [
        Random,
        TopPop,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender
        ]
        

    evaluator = EvaluatorHoldout(URM_test, [1, 5, 10, 20, 40, 50, 100], exclude_seen=True)
    for recommender_class in recommender_class_list:
        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15}

            if isinstance(recommender_object, ItemKNNCFRecommender):
                fit_params = {"topK": itemkNN_best_HP["topK"], "similarity": itemkNN_best_HP["similarity"]}

            elif isinstance(recommender_object, UserKNNCFRecommender):
                fit_params = {"topK": userkNN_best_HP["topK"],  "similarity": userkNN_best_HP["similarity"]}
            
            elif isinstance(recommender_object, P3alphaRecommender):
                fit_params = {"topK": RP3alpha_best_HP["topK"], "alpha": RP3alpha_best_HP["alpha"], "normalize_similarity": RP3alpha_best_HP["normalize_similarity"]}
            
            elif isinstance(recommender_object, RP3betaRecommender):
                fit_params = {"topK": RP3beta_best_HP["topK"], "alpha": RP3beta_best_HP["alpha"], "beta": RP3beta_best_HP["beta"], "normalize_similarity": RP3beta_best_HP["normalize_similarity"]}
            else: # get defaut parameters...........
                fit_params = {}

            # measure training time.....
            start = time.time()
            recommender_object.fit(**fit_params)
            training_time = time.time() - start

            # testing for all records.....
            start = time.time()
            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
            testing_time = time.time() - start
            averageTestingForOneRecord = testing_time / len(URM_test.getnnz(axis=1) > 0) # get number of non-zero rows in test data
        
            results_run_1["TrainingTime(s)"] = [training_time] + [0 for i in range(results_run_1.shape[0] - 1)]
            results_run_1["TestingTimeforRecords(s)"] = [testing_time] + [0 for i in range(results_run_1.shape[0] - 1)]
            results_run_1["AverageTestingTimeForOneRecord(s)"] = [averageTestingForOneRecord] + [0 for i in range(results_run_1.shape[0] - 1)]

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            results_run_1["cuttOff"] = results_run_1.index
            results_run_1.insert(0, 'cuttOff', results_run_1.pop('cuttOff'))
            results_run_1.to_csv(saved_results+"/"+args.dataset+"_"+recommender_class.RECOMMENDER_NAME+".txt", sep = "\t", index = False)


        except Exception as e:
            traceback.print_exc()
            
    ################################ END ##################################
    
    


    


    


    


