from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.IDS4NR.IDSNR import *
from topn_baselines_neurals.Data_manager.IDS4NR_MovleLensBeautyMusic import IDS4NR_MovleLensBeautyMusic
import os
import argparse

from pathlib import Path
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
    parser.add_argument('--dataset', type = str, default='MovieLens', help="MovieLens/Music/Beauty")
    parser.add_argument('--model', type = str, default='LFM', help="LFM or NCF")
    args = parser.parse_args()
    dataset_name = args.dataset
    
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model LFM
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model LFM
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model LFM
    
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    commonFolderName = "results"
    data_path = Path("data/ID4SNR/")
    data_path = data_path.resolve()
    datasetName = args.dataset+".pkl"
    dataset_object = IDS4NR_MovleLensBeautyMusic()
    URM_train, URM_test, UCM_all = dataset_object._load_data_from_give_files(data_path = data_path / datasetName)
    
    # count statistics of dataset...........
    total_elements = URM_train.shape[0] * URM_train.shape[1]
    non_zero_elements = URM_train.nnz + URM_test.nnz
    density = non_zero_elements / total_elements
    print("Number of users: %s, Items: %d, Interactions: %d, Density %.5f, Number of users with no test items: %d." % 
          (URM_train.shape[0], URM_train.shape[1], non_zero_elements, density, np.sum(np.diff(URM_test.indptr) == 0)))

    resultFolder = "results"
    saved_results = "/".join([resultFolder,"ID4SNR",dataset_name] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    
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
    if args.dataset == "MovieLens":
        itemkNN_best_HP = {"topK": 805, "similarity": "cosine", "shrink": 544, "normalize": True, "feature_weighting": "BM25"}
        userkNN_best_HP = {"topK": 1000, "similarity": "cosine", "shrink": 1000, "normalize": True, "feature_weighting": "BM25"}
        RP3alpha_best_HP = {"topK": 117, "alpha": 0.4490914092362502, "normalize_similarity": True}
        RP3beta_best_HP = {"topK": 479, "alpha": 0.41257885094571617, "beta": 0.6050046063241745, "normalize_similarity": True}
        EASE_BestHP = {"topK": None, "l2_norm":  210.9260335725507, "normalize_matrix": False}
        
    elif args.dataset == "Music":
        itemkNN_best_HP = {"topK": 623, "similarity": "cosine", "shrink": 193, "normalize": True, "feature_weighting": "TF-IDF"}
        userkNN_best_HP = {"topK": 149, "similarity": "cosine", "shrink": 0, "normalize": True, "feature_weighting": "none"}
        RP3alpha_best_HP = {"topK": 85, "alpha": 0.4590018257740354, "normalize_similarity": True}
        RP3beta_best_HP = {"topK": 363, "alpha": 0.45815768400879797, "beta": 0.39431553250158924, "normalize_similarity": False}
        EASE_BestHP = {"topK": None, "l2_norm":  47.11431042076403, "normalize_matrix": False}
          
    elif args.dataset == "Beauty":
        itemkNN_best_HP = {"topK": 1000, "similarity": "cosine", "shrink": 1000, "normalize": False, "feature_weighting": "TF-IDF"}
        userkNN_best_HP = {"topK": 255, "similarity": "cosine", "shrink": 1000, "normalize": False, "feature_weighting": "TF-IDF"}
        RP3alpha_best_HP = {"topK": 431, "alpha": 0.5496778369228872, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 1000, "alpha": 0.5472822005926639, "beta": 0.15218913649100715, "normalize_similarity": False}
        EASE_BestHP = {"topK": None, "l2_norm":  105.27021898818253, "normalize_matrix": False}
        
        # SearchBayesianSkopt: New best config found. Config 0: {'topK': 995, 'shrink': 183, 'similarity': 'dice', 'normalize': True}

    evaluator = EvaluatorHoldout(URM_test, [1, 5, 10, 20, 40, 50, 100], exclude_seen=True)
    for recommender_class in recommender_class_list:
        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, None, UCM_all)
            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15}

            if isinstance(recommender_object, ItemKNNCFRecommender):
                fit_params = {"topK": itemkNN_best_HP["topK"], "similarity": itemkNN_best_HP["similarity"], "shrink": userkNN_best_HP["shrink"], "normalize": userkNN_best_HP["normalize"]}

            elif isinstance(recommender_object, UserKNNCFRecommender):
                fit_params = {"topK": userkNN_best_HP["topK"],  "similarity": userkNN_best_HP["similarity"], "shrink": userkNN_best_HP["shrink"], "normalize": userkNN_best_HP["normalize"]}
            
            elif isinstance(recommender_object, P3alphaRecommender):
                fit_params = {"topK": RP3alpha_best_HP["topK"], "alpha": RP3alpha_best_HP["alpha"], "normalize_similarity": RP3alpha_best_HP["normalize_similarity"]}
            
            elif isinstance(recommender_object, RP3betaRecommender):
                fit_params = {"topK": RP3beta_best_HP["topK"], "alpha": RP3beta_best_HP["alpha"], "beta": RP3beta_best_HP["beta"], "normalize_similarity": RP3beta_best_HP["normalize_similarity"]}
            
            elif isinstance(recommender_object, EASE_R_Recommender):
                fit_params = {"topK": EASE_BestHP["topK"], "l2_norm":  EASE_BestHP["l2_norm"], "normalize_matrix": EASE_BestHP["normalize_matrix"]}
            
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
            pass

    ################# RUN experiments for IDS4NR model ####################################
    obj1 = Run_experiments_for_IDSNR(model = args.model, dataset = data_path / datasetName, NumberOfUsersInTestingData = URM_test.shape[0])
    accuracy_measure = obj1.accuracy_values
    accuracy_measure.to_csv(saved_results+"/"+args.dataset+"__"+args.model+".txt", index = False, sep = "\t")

    


