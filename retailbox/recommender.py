# -*- coding: utf-8 -*-
import sys
import time
import pandas as pd
import pickle
import numpy as np
import os.path
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import warnings
import implicit

from scipy.sparse.linalg import spsolve
from output import printGreen, printRed, printYellow, display_customer_information, display_recommender_items
from implicit_als import implicit_weighted_als
from sklearn.preprocessing import MinMaxScaler

from data import preprocess_data_rec_engine, split_data_mask, validate_input, get_items_purchased, rec_items, list_rec, lookup_customer_id
from data import validate_customer_id, search_customer

warnings.filterwarnings("ignore")


def recommender(customer_id, 
                status):
    # Start time
    start = time.time()
    if status:
            printGreen('✔ RetailBox started..\t\t{0:.1f}s'.format(time.time() - start))
    start = time.time()

    # Validate User Input
    validate_customer_id(customer_id)

    # Load Dataframe and create item_table, purchase matrix, etc.
    data = preprocess_data_rec_engine(status=True)

    item_table = data[0]
    purchase_sparse_matrix = data[1]
    customers = data[2]
    products = data[3]
    quantity = data[4]

    if status:
        printGreen('✔ Processed Data..\t\t{0:.1f}s'.format(time.time() - start))
    start = time.time()

    # Split Data (Training/Test Split)
    training_test_split_data = split_data_mask(purchase_sparse_matrix, pct_test = 0.2)
    
    product_training_set = training_test_split_data[0]
    product_test_set = training_test_split_data[1]
    product_user_altered = training_test_split_data[2]

    if status:
        printGreen('✔ Split Data into Training and Test Sets..\t\t{0:.1f}s'.format(time.time() - start))
    start = time.time()
    
    # Train Recommendation Engine on given algorithm
    alpha = 15
    recommender_vecs = implicit.alternating_least_squares((product_training_set * alpha).astype('double'),
                                                          factors = 20,
                                                          regularization = 0.1,
                                                          iterations = 50)

    user_vecs = recommender_vecs[0]
    item_vecs = recommender_vecs[1]
    
    customers_arr  = np.array(customers)
    products_arr = np.array(products)

    if status:
        printGreen('✔ Recommender System Training Done..\t\t{0:.1f}s'.format(time.time() - start))
    start = time.time()

    # Lookup customer id
    cid = lookup_customer_id(customer_id)

    # Generate Recommendations for Customer
    rec_output = rec_items(cid, product_training_set, user_vecs,
                           item_vecs, customers_arr, products_arr,
                           item_table)

    # Display Customer
    df = pd.read_pickle('../data/final/df_final.pkl')
    table_pickle_file = open('../data/final/df_customer_table.pkl', "rb")
    customer_table = pickle.load(table_pickle_file)
    table_pickle_file.close() 
    search_customer(customer_id, df, customer_table)

    # Display Item Recommendations
    recommended_items_list = list_rec(rec_output)
    display_recommender_items(recommended_items_list)
    

def main():
    recommender(customer_id=5,
                status=True)
    

if __name__ == '__main__':
    main()