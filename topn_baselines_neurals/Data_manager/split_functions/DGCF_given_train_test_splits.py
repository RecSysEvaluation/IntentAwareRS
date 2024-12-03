#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from topn_baselines_neurals.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

def split_train_test_validation(loaded_dataset, test_data_dictionary, validation = False, validation_portion = 0.1):
    """
    The function splits an URM in two matrices selecting the k_out interactions one user at a time
    :param URM:
    :param k_out:
    :param use_validation_set:
    :param leave_random_out:
    :return:
    """
    use_validation_set = validation 
    URM = loaded_dataset.AVAILABLE_URM['URM_all']
    
    
    updated_test_data = update_item_ids_of_original_data(loaded_dataset.item_original_ID_to_index,loaded_dataset.user_original_ID_to_index, test_data_dictionary)
    

    URM = sps.csr_matrix(URM)
    n_users, n_items = URM.shape

    URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    if use_validation_set:
         URM_validation_builder_train = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                                          auto_create_col_mapper=False, n_cols = n_items)
         URM_validation_builder_test = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                                          auto_create_col_mapper=False, n_cols = n_items)
    
    for user_id in range(URM.shape[0]):

        if (user_id in updated_test_data):
            test_records_items = np.array(list(updated_test_data[int(user_id)]))
            
            start_user_position = URM.indptr[user_id]
            end_user_position = URM.indptr[user_id+1]

            user_profile = URM.indices[start_user_position:end_user_position]
            # remove test items from the traning data
            user_profile = list(user_profile)

            if (set(user_profile) == set(test_records_items)):
    
                URM_test_builder.add_data_lists([user_id]*len(test_records_items), test_records_items, np.ones(len(test_records_items)))
                URM_train_builder.add_data_lists([user_id]*len(np.array(user_profile)), np.array(user_profile), np.ones(len(user_profile)))

            else:
                for item_ in test_records_items:
                    user_profile.remove(item_)
                URM_test_builder.add_data_lists([user_id]*len(test_records_items), test_records_items, np.ones(len(test_records_items)))
                URM_train_builder.add_data_lists([user_id]*len(user_profile), user_profile, np.ones(len(user_profile)))
                
        else:
            start_user_position = URM.indptr[user_id]
            end_user_position = URM.indptr[user_id+1]
            user_profile = URM.indices[start_user_position:end_user_position]
            train_data_ones = np.ones(len(user_profile))
            URM_train_builder.add_data_lists([user_id]*len(user_profile), user_profile, train_data_ones)


    if use_validation_set == True:
        URM_train = URM_train_builder.get_SparseMatrix()
        for user_id in range(URM_train.shape[0]):

            start_user_position = URM_train.indptr[user_id]
            end_user_position = URM_train.indptr[user_id+1]
            user_profile = URM_train.indices[start_user_position:end_user_position]
            user_interaction_items = user_profile
            user_interaction_data = np.ones(len(user_profile))
            
            k_out = int(len(user_profile)) - int(len(user_profile) * validation_portion)

            if len(user_profile) > 0 and k_out > 0:

                user_interaction_items_validation = user_interaction_items[k_out:]
                user_interaction_data_validation = user_interaction_data[k_out:]
                URM_validation_builder_test.add_data_lists([user_id]*len(user_interaction_data_validation), user_interaction_items_validation, user_interaction_data_validation)

                user_interaction_items_train = user_interaction_items[0:k_out]
                user_interaction_data_train = user_interaction_data[0:k_out]
                URM_validation_builder_train.add_data_lists([user_id]*len(user_interaction_items_train), user_interaction_items_train, user_interaction_data_train)
            else:
                user_interaction_items_train = user_interaction_items
                user_interaction_data_train = user_interaction_data
                URM_validation_builder_train.add_data_lists([user_id]*len(user_interaction_items_train), user_interaction_items_train, user_interaction_data_train)
    
    URM_train = URM_train_builder.get_SparseMatrix()
    URM_test = URM_test_builder.get_SparseMatrix()
    
    URM_train = sps.csr_matrix(URM_train)
    user_no_item_train = np.sum(np.ediff1d(URM_train.indptr) == 0)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no Train items".format(user_no_item_train, user_no_item_train/n_users*100, n_users))

    if use_validation_set:
        
        URM_validation_train = URM_validation_builder_train.get_SparseMatrix()
        URM_validation_test = URM_validation_builder_test.get_SparseMatrix()

        URM_validation_train = sps.csr_matrix(URM_validation_train)
        URM_validation_test = sps.csr_matrix(URM_validation_test)


        user_no_item_validation = np.sum(np.ediff1d(URM_validation_test.indptr) == 0)

        if user_no_item_validation != 0:
            print("Warning: {} ({:.2f} %) of {} users have no Validation items".format(user_no_item_validation, user_no_item_validation/n_users*100, n_users))
        return URM_train, URM_test, URM_validation_train, URM_validation_test
    return URM_train, URM_test


def update_item_ids_of_original_data(dictionary_item_original_to_index, user_original_ID_to_index,  test_data_dictionary):
    updated_original_test_data = dict()

    for uid, items_set in test_data_dictionary.items():
        temp = set()
        for item in items_set:
            temp.add(dictionary_item_original_to_index[str(item)])
        updated_original_test_data[user_original_ID_to_index[str(uid)]] = temp

    return updated_original_test_data