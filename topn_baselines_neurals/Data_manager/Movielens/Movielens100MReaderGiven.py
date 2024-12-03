#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import zipfile, shutil
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DataReader_utils import download_from_URL
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
from topn_baselines_neurals.Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years
import pickle
from topn_baselines_neurals.Data_manager.split_functions.ieee_transactions_given_train_test_splits import split_train_test_validation


class Movielens100MReaderGiven(DataReader):

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
    
    
    def _load_data_from_give_files(self, data_path = "MovieLens.pkl", validation = False):
        
        try:
            with open(data_path, 'rb') as file:
                data_dictionary = pickle.load(file)
    
                train_data = data_dictionary["train_user_list"][1:]
                test_data = data_dictionary["test_user_list"][1:]

                user_features_dictionary = data_dictionary["user_all_feat_dict"]   
                del user_features_dictionary[0]   
                URM_dataframe, UCM_dataframe = self.convert_dictionary_to_dataFrame(train_data, test_data, user_features_dictionary)
                
                
        except FileNotFoundError:
            print(f"File not found: {data_path}")
         
        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        dataset_manager.add_UCM(UCM_dataframe, "UCM_all")
        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)
        
        if validation == True:
            URM_train, URM_test, URM_validation_train, URM_validation_test = split_train_test_validation(loaded_dataset, test_data, validation=validation)
            return URM_train, URM_test, URM_validation_train, URM_validation_test, loaded_dataset.AVAILABLE_UCM['UCM_all']
        else:
            URM_train, URM_test = split_train_test_validation(loaded_dataset,test_data,   validation=validation)
            return URM_train, URM_test, loaded_dataset.AVAILABLE_UCM['UCM_all']
        
        
        
    def _load_from_original_file(self):
        # Load data from original
        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        ICM_genre_path = dataFile.extract("ml-1m/movies.dat", path=zipFile_path + "decompressed/")
        UCM_path = dataFile.extract("ml-1m/users.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")

        self._print("Loading Interactions")
        URM_all_dataframe, URM_timestamp_dataframe = _loadURM(URM_path, header=None, separator='::')

        self._print("Loading Item Features genres")
        ICM_genres_dataframe, ICM_years_dataframe = _loadICM_genres_years(ICM_genre_path, header=None, separator='::', genresSeparator="|")

        self._print("Loading User Features")
        UCM_dataframe = pd.read_csv(filepath_or_buffer=UCM_path, sep="::", header=None, dtype={0:str, 1:str, 2:str, 3:str, 4:str}, engine='python')
        UCM_dataframe.columns = ["UserID", "gender", "age_group", "occupation", "zip_code"]

        # For each user a list of features
        UCM_list = [[feature_name + "_" + str(UCM_dataframe[feature_name][index]) for feature_name in ["gender", "age_group", "occupation", "zip_code"]] for index in range(len(UCM_dataframe))]
        UCM_dataframe = pd.DataFrame(UCM_list, index=UCM_dataframe["UserID"]).stack()
        UCM_dataframe = UCM_dataframe.reset_index()[[0, 'UserID']]
        UCM_dataframe.columns = ['FeatureID', 'UserID']
        UCM_dataframe["Data"] = 1


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_timestamp_dataframe, "URM_timestamp")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genres")
        dataset_manager.add_ICM(ICM_years_dataframe, "ICM_year")
        dataset_manager.add_UCM(UCM_dataframe, "UCM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset











