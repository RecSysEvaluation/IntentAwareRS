#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19

@author: Maurizio Ferrari Dacrema
"""


######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from topn_baselines_neurals.Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects



######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from topn_baselines_neurals.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from topn_baselines_neurals.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from topn_baselines_neurals.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from topn_baselines_neurals.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from topn_baselines_neurals.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
# from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_WARP_Cython, MatrixFactorization_SVDpp_Cython, MatrixFactorization_AsySVD_Cython
# from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from topn_baselines_neurals.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
# from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from topn_baselines_neurals.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
# from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
# from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
# from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
# from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
# from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
# from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
# from Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender
