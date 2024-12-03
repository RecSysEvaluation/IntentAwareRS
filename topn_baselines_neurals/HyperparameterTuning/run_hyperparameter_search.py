#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

import os, multiprocessing
from functools import partial

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from topn_baselines_neurals.Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
from topn_baselines_neurals.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from topn_baselines_neurals.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from topn_baselines_neurals.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from topn_baselines_neurals.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

######################################################################
from skopt.space import Real, Integer, Categorical
import traceback

from topn_baselines_neurals.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from topn_baselines_neurals.HyperparameterTuning.SearchSingleCase import SearchSingleCase
from topn_baselines_neurals.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
######################################################################

def run_KNNRecommender_on_similarity_type(similarity_type, hyperparameterSearch,
                                          hyperparameter_search_space,
                                          recommender_input_args,
                                          n_cases,
                                          n_random_starts,
                                          resume_from_saved,
                                          save_model,
                                          evaluate_on_test,
                                          max_total_time,
                                          output_folder_path,
                                          output_file_name_root,
                                          metric_to_optimize,
                                          cutoff_to_optimize,
                                          allow_weighting = False,
                                          allow_bias_ICM = False,
                                          allow_bias_URM = False,
                                          recommender_input_args_last_test = None):

    original_hyperparameter_search_space = hyperparameter_search_space

    hyperparameters_range_dictionary = {
        "topK": Integer(5, 1000),
        "shrink": Integer(0, 1000),
        "similarity": Categorical([similarity_type]),
        "normalize": Categorical([True, False]),
    }
    is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]

    if similarity_type == "asymmetric":
        hyperparameters_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparameters_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "euclidean":
        hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
        hyperparameters_range_dictionary["normalize_avg_row"] = Categorical([True, False])
        hyperparameters_range_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])


    if not is_set_similarity:

        if allow_weighting:
            hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

        if allow_bias_ICM:
            hyperparameters_range_dictionary["ICM_bias"] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')

        if allow_bias_URM:
            hyperparameters_range_dictionary["URM_bias"] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')

    local_hyperparameter_search_space = {**hyperparameters_range_dictionary, **original_hyperparameter_search_space}

    hyperparameterSearch.search(recommender_input_args,
                           hyperparameter_search_space= local_hyperparameter_search_space,
                           n_cases = n_cases,
                           n_random_starts = n_random_starts,
                           resume_from_saved = resume_from_saved,
                           save_model = save_model,
                           evaluate_on_test = evaluate_on_test,
                           max_total_time = max_total_time,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root + "_" + similarity_type,
                           metric_to_optimize = metric_to_optimize,
                           cutoff_to_optimize = cutoff_to_optimize,
                           recommender_input_args_last_test = recommender_input_args_last_test)


def runHyperparameterSearch_Collaborative(recommender_class, URM_train, URM_train_last_test = None,
                                          n_cases = None, n_random_starts = None, resume_from_saved = False,
                                          save_model = "best", evaluate_on_test = "best", max_total_time = None,
                                          evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                          metric_to_optimize = None, cutoff_to_optimize = None,
                                          output_folder_path ="result_experiments/", parallelizeKNN = True,
                                          allow_weighting = False, allow_bias_URM=False, allow_dropout_MF = False, similarity_type_list = None):
    
    """
    This function performs the hyperparameter optimization for a collaborative recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param max_total_time:    [None or int] if set stops the hyperparameter optimization when the time in seconds for training and validation exceeds the threshold
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param cutoff_to_optimize:  Integer with the recommendation list length to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param allow_bias_URM:      Boolean value, if True it enables the use of bias to shift the values of the URM
    :param allow_dropout_MF:    Boolean value, if True it enables the use of dropout on the latent factors of MF algorithms
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

    URM_train = URM_train.copy()
    n_users, n_items = URM_train.shape
    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()
    try:
        output_file_name_root = recommender_class.RECOMMENDER_NAME

        hyperparameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)
        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no hyperparameters therefore only one evaluation is needed
            """

            hyperparameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )


            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None


            hyperparameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values={},
                                   metric_to_optimize = metric_to_optimize,
                                   cutoff_to_optimize = cutoff_to_optimize,
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = output_file_name_root,
                                   resume_from_saved = resume_from_saved,
                                   save_model = save_model,
                                   evaluate_on_test = evaluate_on_test,
                                   )


            return



        ##########################################################################################################

        if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

            
            similarity_type = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )


            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None


            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                           recommender_input_args = recommender_input_args,
                                                           hyperparameter_search_space = {},
                                                           hyperparameterSearch = hyperparameterSearch,
                                                           n_cases = n_cases,
                                                           n_random_starts = n_random_starts,
                                                           resume_from_saved = resume_from_saved,
                                                           save_model = save_model,
                                                           evaluate_on_test = evaluate_on_test,
                                                           max_total_time = max_total_time,
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
                                                           metric_to_optimize = metric_to_optimize,
                                                           cutoff_to_optimize = cutoff_to_optimize,
                                                           allow_weighting = allow_weighting,
                                                           allow_bias_URM = allow_bias_URM,
                                                           recommender_input_args_last_test = recommender_input_args_last_test)



            if parallelizeKNN:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
                pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)
                pool.close()
                pool.join()

            else:
                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)
            return



       ##########################################################################################################

        if recommender_class is P3alphaRecommender:

            hyperparameters_range_dictionary = {
                "topK": Integer(5, 1000),
                "alpha": Real(low = 0, high = 2, prior = 'uniform'),
                "normalize_similarity": Categorical([True, False]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparameters_range_dictionary = {
                "topK": Integer(5, 1000),
                "alpha": Real(low = 0, high = 2, prior = 'uniform'),
                "beta": Real(low = 0, high = 2, prior = 'uniform'),
                "normalize_similarity": Categorical([True, False]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )

        if recommender_class is EASE_R_Recommender:

            hyperparameters_range_dictionary = {
                "topK": Categorical([None]),
                "normalize_matrix": Categorical([False]),
                "l2_norm": Real(low = 1e0, high = 1e7, prior = 'log-uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )


        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None
        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        hyperparameterSearch.search(recommender_input_args,
                               hyperparameter_search_space= hyperparameters_range_dictionary,
                               n_cases = n_cases,
                               n_random_starts = n_random_starts,
                               resume_from_saved = resume_from_saved,
                               save_model = save_model,
                               evaluate_on_test = evaluate_on_test,
                               max_total_time = max_total_time,
                               output_folder_path = output_folder_path,
                               output_file_name_root = output_file_name_root,
                               metric_to_optimize = metric_to_optimize,
                               cutoff_to_optimize = cutoff_to_optimize,
                               recommender_input_args_last_test = recommender_input_args_last_test)
    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()




def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
    from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout


    dataset_reader = Movielens1MReader()
    output_folder_path = "result_experiments/SKOPT_test/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data(save_folder_path=output_folder_path + "data/")

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    collaborative_algorithm_list = [
        Random,
        TopPop,
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
    ]



    from Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])


    runHyperparameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 8,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path)

    for recommender_class in collaborative_algorithm_list:

        try:

            runHyperparameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()


if __name__ == '__main__':
    read_data_split_and_search()
