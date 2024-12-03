#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""
import copy
import pandas as pd
import zipfile, shutil
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DataReader_utils import download_from_URL
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
import pickle
from topn_baselines_neurals.Data_manager.split_functions.ieee_transactions_given_train_test_splits import split_train_test_validation


class IDS4NR_MovleLensBeautyMusic(DataReader):

    DATASET_URL = ""
    DATASET_SUBFOLDER = ""
    CONFERENCE_JOURNAL = ""
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]
    
    IS_IMPLICIT = False
    FILE_NAME = "movielens100k_longtail_data.pkl"
    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
    
    
    def _load_data_from_give_files(self, data_path = "MovieLens.pkl", validation = False, validation_portion = 0.1):
        
        try:
            with open(data_path, 'rb') as file:
                data_dictionary = pickle.load(file)
    
                train_data = data_dictionary["train_user_list"][1:]
                test_data = data_dictionary["test_user_list"][1:]
                user_features_dictionary = data_dictionary["user_all_feat_dict"]   
                del user_features_dictionary[0]
                        
        except FileNotFoundError:
            print(f"File not found: {data_path}")

        
        URM_dataframe, UCM_dataframe = self.convert_dictionary_to_dataFrame(train_data, test_data, user_features_dictionary.copy())
        self.count_interactions_per_user_item(URM_dataframe)
        self.checkLeakage(train_data, test_data)
        
        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        dataset_manager.add_UCM(UCM_dataframe, "UCM_all")
        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)
        
        if validation == True:
            URM_train, URM_test, URM_validation_train, URM_validation_test = split_train_test_validation(loaded_dataset, test_data, validation=validation, validation_portion = validation_portion)
            return URM_train, URM_test, URM_validation_train, URM_validation_test, loaded_dataset.AVAILABLE_UCM['UCM_all']
        else:
            URM_train, URM_test = split_train_test_validation(loaded_dataset,test_data,   validation=validation)
            return URM_train, URM_test, loaded_dataset.AVAILABLE_UCM['UCM_all']
        
    def convert_dictionary_to_dataFrame(self, train_list, test_list, user_content_dictionary):

        full_data = dict()
        train_list1 = copy.deepcopy(train_list)
        test_list1 = copy.deepcopy(test_list)
        for i in range(len(train_list1)):
            
            temp = train_list1[i]
            temp.update(test_list1[i])
            full_data[i] = temp
            
        expanded_data = [(key, value) for key, values in full_data.items() for value in values]
        # Create DataFrame
        URM_dataframe = pd.DataFrame(expanded_data, columns=['UserID', 'ItemID'])

        URM_dataframe["Data"] = 1
        URM_dataframe['UserID']= URM_dataframe['UserID'].astype(str)
        URM_dataframe['ItemID']= URM_dataframe['ItemID'].astype(str)
        
        user_list = [ value for key,value in user_content_dictionary.items()]
        temp_user_dict = dict()
        user_list = user_list[1:]
        
        for i in range(len(user_list)):
            temp_user_dict[i] = user_list[i]
            
        
        expanded_user = [(key, value)  for key, values in temp_user_dict.items() for value in values]
        UCM_dataframe = pd.DataFrame(expanded_user, columns=['UserID', 'FeatureID'])
        UCM_dataframe["Data"] = 1
        
        UCM_dataframe['UserID']= UCM_dataframe['UserID'].astype(str)
        UCM_dataframe['FeatureID']= UCM_dataframe['FeatureID'].astype(str)
        return URM_dataframe, UCM_dataframe
    
    def count_interactions_per_user_item(self, df):
        user_interaction = df.groupby("UserID")["ItemID"].count()
        item_interaction = df.groupby("ItemID")["UserID"].count()
        if user_interaction.empty:
            print("No interactions found for users.")
        else:
            print("Interactions per user --> Minimum: %d Maximum: %d" % (min(user_interaction), max(user_interaction)))
        if item_interaction.empty:
            print("No interactions found for items.")
        else:
            print("Interactions per item --> Minimum: %d Maximum: %d" % (min(item_interaction), max(item_interaction)))

    def checkLeakage(self, train_data, test_data):
        checkLeakage = len([i for i in range(len(train_data)) if len(train_data[i].intersection(test_data[i])) > 0])
        if (checkLeakage == 0):
            print("We do not observe data leakage issue")
        else:
            print("Total users: %d, Users with data leakage: %d"% (len(train_data), checkLeakage))
            """
            for i in range(len(train_data)):
                common_items = len(train_data[i].intersection(test_data[i]))
                print("train items: %d, test items: %d, common items: %d" % (len(train_data[i]), len(test_data[i]), common_items))
            """

    def conversion_set_list(self, setDictionary):
        tempDict = {}
        for i in range(len(setDictionary)):
            tempDict[i] = setDictionary[i]

        return tempDict

        











