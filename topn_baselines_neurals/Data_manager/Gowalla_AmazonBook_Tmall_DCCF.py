#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""
import pickle
import pandas as pd
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
from topn_baselines_neurals.Data_manager.split_functions.DCCF_given_train_test_splits import split_train_test_validation


class Gowalla_AmazonBook_Tmall_DCCF(DataReader):

    DATASET_URL = ""
    DATASET_SUBFOLDER = ""
    CONFERENCE_JOURNAL = ""
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]
    IS_IMPLICIT = True
    
    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
    def _load_data_from_give_files(self, datapath, validation = False , validation_portion = 0.1):
        
        # get the provided train-test splits....
        try:
            train_file = datapath / 'train.pkl'
            test_file = datapath / 'test.pkl'

            with open(train_file, 'rb') as f:
                train_mat = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_mat = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {datapath}")

        R = train_mat.todok() # dok --> dictionary of keys....
        train_items, test_set = {}, {} # get list of items each user interacted in training data and list of items each user interacted in test data......
        train_uid, train_iid = train_mat.row, train_mat.col
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in train_items:
                train_items[uid] = [iid]
            else:
                train_items[uid].append(iid)

        test_uid, test_iid = test_mat.row, test_mat.col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in test_set:
                test_set[uid] = [iid]
            else:
                test_set[uid].append(iid)

        train_dictionary = train_items
        test_dictionary = test_set

        self.checkLeakage(train_dictionary.copy(), test_dictionary.copy())
        URM_dataframe = self.convert_dictionary_to_dataframe_DGCF(train_dictionary.copy(), test_dictionary.copy())
        self.count_interactions_per_user_item(URM_dataframe)
        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)
        
        if validation == True:
            URM_train, URM_test, URM_validation_train, URM_validation_test = split_train_test_validation(loaded_dataset, test_dictionary, validation=validation, validation_portion = validation_portion)
            return URM_train, URM_test, URM_validation_train, URM_validation_test
        else:
            URM_train, URM_test = split_train_test_validation(loaded_dataset, test_dictionary,   validation=validation)
            return URM_train, URM_test
        
    def convert_dictionary_to_dataframe_DGCF(self, train_dictionary, test_dictionary):
        for key, _ in test_dictionary.items():
            train_dictionary[key]+=test_dictionary[key] 
        expanded_data = [(key, value) for key, values in train_dictionary.items() for value in values]
        # Create DataFrame
        URM_dataframe = pd.DataFrame(expanded_data, columns=['UserID', 'ItemID'])
        URM_dataframe["Data"] = 1
        URM_dataframe['UserID']= URM_dataframe['UserID'].astype(str)
        URM_dataframe['ItemID']= URM_dataframe['ItemID'].astype(str)
        return URM_dataframe # return dataframe, which contains all interactions from the training dictionary and testing dictionary.
    
    def checkLeakage(self, train_dictionary, test_dictionary):
        checkLeakage = len([key for key, item in test_dictionary.items() if (len(set(item).intersection(train_dictionary[key])) > 0)])
        if (checkLeakage == 0):
            print("We do not observe data leakage issue")
        else:
            print("Total users: %d, Users with data leakage: %d", (len(train_dictionary), checkLeakage))

    def count_interactions_per_user_item(self, df):
        user_interaction = df.groupby("UserID")["Data"].count()  # count mi
        item_interaction = df.groupby("ItemID")["Data"].count()
        if user_interaction.empty:
            print("No interactions found for users.")
        else:
            print("Interactions per user --> Minimum: %d Maximum: %d" % (min(user_interaction), max(user_interaction)))
        if item_interaction.empty:
            print("No interactions found for items.")
        else:
            print("Interactions per item --> Minimum: %d Maximum: %d" % (min(item_interaction), max(item_interaction)))
    



       
        











