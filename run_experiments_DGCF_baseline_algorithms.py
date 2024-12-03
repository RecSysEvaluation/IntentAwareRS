
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
import traceback, os
from pathlib import Path
import numpy as np
import time
import argparse
from topn_baselines_neurals.Data_manager.Gowalla_Yelp_Amazon_DGCF import Gowalla_Yelp_Amazon_DGCF 
def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)
    return recommender_object

if __name__ == '__main__':

    commonFolderName = "experiments_results"
    model = "DGCF"
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='gowalla', help="yelp2018, gowalla, amazonbook")
    args = parser.parse_args()

    dataset_name = args.dataset
    commonFolderName = "results"
    data_path = Path("data/DGCF/")
    data_path = data_path.resolve()
    validation_set = False
    datasetName = args.dataset+".pkl"
    
    model = "DGCF"
    dataset_object = Gowalla_Yelp_Amazon_DGCF()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path = data_path / dataset_name)

    total_elements = URM_train.shape[0] * URM_train.shape[1]
    non_zero_elements = URM_train.nnz + URM_test.nnz
    # Sparsity calculation
    density = non_zero_elements / total_elements
    
    print("Number of users: %s, Items: %d, Interactions: %d, Density %.5f, Number of users with no test items: %d." % 
          (URM_train.shape[0], URM_train.shape[1], non_zero_elements, density, np.sum(np.diff(URM_test.indptr) == 0)))
    
    # If directory does not exist, create
    saved_results = "/".join([commonFolderName,model,args.dataset] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    output_root_path = saved_results+"/"

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
        itemkNN_best_HP = {"topK": 508, "similarity": "cosine"}
        userkNN_best_HP = {"topK": 146, "similarity": "cosine"}
        RP3alpha_best_HP = {"topK": 777, "alpha": 1.087096950563704, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 777, "alpha": 0.5663562161452378, "beta": 0.001085447926739258, "normalize_similarity": True}
        
    elif args.dataset == "yelp2018":
        itemkNN_best_HP = {"topK": 144, "similarity": "cosine"}
        userkNN_best_HP = {"topK": 146, "similarity": "cosine"}
        RP3alpha_best_HP = {"topK": 496, "alpha": 0.7681732734954694, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 350, "alpha": 0.7681732734954694, "beta": 0.4181395996963926, "normalize_similarity": True}
    
    elif args.dataset == "amazonbook":
        itemkNN_best_HP = {"topK": 125, "similarity": "cosine"}
        userkNN_best_HP = {"topK": 146, "similarity": "cosine"}
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
    
    evaluator = EvaluatorHoldout(URM_test, [1, 5,10, 20, 50, 100], exclude_seen=True)
    for recommender_class in recommender_class_list:
        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, None, None)
            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15}

            if isinstance(recommender_object, ItemKNNCFRecommender):
                fit_params = {"topK": itemkNN_best_HP["topK"], "similarity": itemkNN_best_HP["similarity"]}

            elif isinstance(recommender_object, UserKNNCFRecommender):
                fit_params = {"topK": userkNN_best_HP["topK"], "similarity": userkNN_best_HP["similarity"]}

            elif isinstance(recommender_object, P3alphaRecommender):
                fit_params = {"topK": RP3alpha_best_HP["topK"], "alpha": RP3alpha_best_HP["alpha"], "normalize_similarity": RP3alpha_best_HP["normalize_similarity"]}
            
            elif isinstance(recommender_object, RP3betaRecommender):
                fit_params = {"topK": RP3beta_best_HP["topK"], "alpha": RP3beta_best_HP["alpha"], "beta": RP3beta_best_HP["beta"], 
                              "normalize_similarity": RP3beta_best_HP["normalize_similarity"]}
                
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
            results_run_1.to_csv(output_root_path+args.dataset+"_"+recommender_class.RECOMMENDER_NAME+".txt", sep = "\t", index = False)

        except Exception as e:
            traceback.print_exc()
            
