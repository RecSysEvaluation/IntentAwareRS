#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
from topn_baselines_neurals.HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from topn_baselines_neurals.Data_manager.IDS4NR_MovleLensBeautyMusic import IDS4NR_MovleLensBeautyMusic
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from functools import partial
from pathlib import Path
import os, multiprocessing
import argparse
import numpy as np


def run_experiments_for_INS4NR_Model():
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='MovieLens', help="MovieLens/Music/Beauty")
    args = parser.parse_args()
    dataset_name = args.dataset
    model = "ID4SNR"
    task = "optimization"
    commonFolderName = "results"
    data_path = Path("data/ID4SNR/"+dataset_name+".pkl")
    data_path = data_path.resolve()
    validation_set = True
    validation_portion = 0.1
   
    dataset_object = IDS4NR_MovleLensBeautyMusic()
    URM_train, URM_test, URM_validation_train, URM_validation_test, UCM_all = dataset_object._load_data_from_give_files(data_path, validation=validation_set, validation_portion = validation_portion)
    saved_results = "/".join([commonFolderName,model,dataset_name, task] )
    print("Totla number of users:  "+str(URM_train.shape[0]))
    print("Totla number of items:  "+str(URM_train.shape[1]))
    
    # If directory does not exist, create
    if not os.path.exists(saved_results):
        os.makedirs(saved_results+"/")
    # model to optimize
    
    collaborative_algorithm_list = [
            P3alphaRecommender,
            RP3betaRecommender,
            ItemKNNCFRecommender,
            UserKNNCFRecommender,
            EASE_R_Recommender
    ]

    from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
    cutoff_list = [10]
    metric_to_optimize = "RECALL"
    cutoff_to_optimize = 10
    n_cases = 50
    n_random_starts = 5
    
    evaluator_validation = EvaluatorHoldout(URM_validation_test, cutoff_list = cutoff_list)
    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_validation_train,
                                                       metric_to_optimize = metric_to_optimize,
                                                       cutoff_to_optimize = cutoff_to_optimize,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       output_folder_path = saved_results,
                                                       resume_from_saved = True,
                                                       parallelizeKNN = False, allow_weighting = True,
                                                       similarity_type_list = ['cosine']
                                                       )


    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)



if __name__ == '__main__':
    run_experiments_for_INS4NR_Model()
