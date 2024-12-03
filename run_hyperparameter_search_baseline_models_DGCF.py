#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
from topn_baselines_neurals.HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from topn_baselines_neurals.Data_manager.Gowalla_Yelp_Amazon_DGCF import Gowalla_Yelp_Amazon_DGCF 
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from functools import partial
from pathlib import Path
import os, multiprocessing
import argparse
def run_experiments_for_IDSNR_Model():
    """
    This function provides a simple example on how to tune parameters of a given algorithm
    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='yelp2018', help="yelp2018, gowalla, amazonbook")

    args = parser.parse_args()
    dataset_name = args.dataset
    model = "DGCF"
    task = "optimization"
    
    commonFolderName = "results"
    data_path = Path("data/DGCF/")
    data_path = data_path.resolve()
    validation_portion  = 0.2
    validation_set = True
    dataset_object = Gowalla_Yelp_Amazon_DGCF()
    URM_train, URM_test, URM_validation_train, URM_validation_test = dataset_object._load_data_from_give_files(validation=validation_set, data_name = data_path / dataset_name, validation_portion = validation_portion)
    saved_results = "/".join([commonFolderName,model,args.dataset, task] )
    # If directory does not exist, create
    if not os.path.exists(saved_results):
        os.makedirs(saved_results+"/")
    # model to optimiz
    collaborative_algorithm_list = [
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        EASE_R_Recommender
    ]
    from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
    cutoff_list = [20]
    metric_to_optimize = "RECALL"
    cutoff_to_optimize = 20
    n_cases = 100
    n_random_starts = 10

    evaluator_validation = EvaluatorHoldout(URM_validation_test, cutoff_list = cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)
    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_validation_train,
                                                       URM_train_last_test = URM_train,
                                                       metric_to_optimize = metric_to_optimize,
                                                       cutoff_to_optimize = cutoff_to_optimize,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = saved_results,
                                                       resume_from_saved = True,
                                                       parallelizeKNN = False, allow_weighting = True,
                                                       similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]
                                                       )


    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)












if __name__ == '__main__':


    run_experiments_for_IDSNR_Model()
