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
import implicit

from output import printGreen, printRed, display_customer_information
from implicit_als import implicit_weighted_als

# Store CustomerIDs
customer_id = {}
customer_id_search = {}

def invert_dict(c):
    inv = {v: k for k, v in c.items()}
    return inv

#
# Build out Customer Tables and clean dataframe
#
# Dataframe that is built will be used for searching for customers via 
# id's, and be used to provide extended application functionality
# that has to do with basic searching functionality of customers and items
#
def process_data(status):
    start = time.time()
    
    # Get file path for source dataset
    file_path = '../data/processed/df_retail.bin'
    df = None

    # Check if processed data file exists, if not, process raw dataset
    if os.path.exists(file_path) == False:
        df = pd.read_excel('../data/raw/Online Retail.xlsx')
        # # Save df into pickle file
        with open('../data/processed/df_retail.bin', 'wb') as f_out:
            pickle.dump(df, f_out)
    
    with open('../data/processed/df_retail.bin', 'rb') as f_in:
        df = pickle.load(f_in)

    # Display import status
    if status:
        printGreen('✔ Imported Data\t\t{0:.1f}s'.format(time.time() - start))
    
    # Clean Data, lowercase, remove "return" transaction, and unknown user transactions
    start = time.time()

    df.columns = df.columns.str.lower()
    df = df[~df.invoiceno.astype('str').str.startswith('C')].reset_index(drop=True)
    df.customerid = df.customerid.fillna(-1).astype('int32')

    # Encode item IDs with integers
    stockcode_values = df.stockcode.astype('str')
    stockcodes = sorted(set(stockcode_values))
    stockcodes = {c: i for (i, c) in enumerate(stockcodes)}
    df.stockcode = stockcode_values.map(stockcodes).astype('int32')

    # Display process status
    if status:
       printGreen('✔ Processed Data\t\t{0:.1f}s'.format(time.time() - start)) 

    # Store Customer IDs in a table
    start = time.time()
    i = 0
    counter = 0
    while counter != 532620:
        if df['customerid'][counter] not in customer_id and df['customerid'][counter] != -1 and df['customerid'][counter] != None:
            customer_id[df['customerid'][counter]] = i
            i += 1
        counter += 1
    
    # Save Customer ID Table
    customer_id_storage = open('../data/final/df_customer_table_long.pkl', "wb")
    pickle.dump(customer_id, customer_id_storage)
    customer_id_storage.close()
    
    # Display process status
    if status:
       printGreen('✔ Stored Customer Data in Table\t\t{0:.1f}s'.format(time.time() - start))

    # Save final DF for quick access
    df.to_pickle('../data/final/df_final.pkl')

    return df

#
# Build out item tables, and product matrices
#
# This function mainly gets the data processed ready to be fed
# into the recommender system 
#
def preprocess_data_rec_engine(status):
    
    # Load Data
    data = pd.read_excel('../data/raw/Online Retail.xlsx')

    # Get rid of rows that don't have Customer ID's
    cleaned_data = data.loc[pd.isnull(data.CustomerID) == False]

    # Lookup table for item ID, will be used to search for item ID and return description
    item_table = cleaned_data[['StockCode', 'Description']].drop_duplicates()
    item_table['StockCode'] = item_table.StockCode.astype(str)

    # Get rid of data that isn't needed
    cleaned_data['CustomerID'] = cleaned_data.CustomerID.astype(int)
    cleaned_data = cleaned_data[['StockCode', 'Quantity', 'CustomerID']]
    
    # Group together customers and purchases and replace 0's in quantity column to 1 
    # and customers that didn't buy items
    group_cleaned = cleaned_data.groupby(['CustomerID', 'StockCode']).sum().reset_index()
    group_cleaned.Quantity.loc[group_cleaned.Quantity == 0] = 1
    group_cleaned_purchased = group_cleaned.query('Quantity > 0')

    print(group_cleaned_purchased.head())

    # Create Sparse Ratings Matrix of Users and Items U*I
    customers = list(np.sort(group_clean_purchased.CustomerID.unique()))
    products = list(group_clean_purchased.StockCode.unique())
    quantity = list(group_clean_purchased.Quantity)

    # Get rows and columns to craete purchase sparse matrix
    rows = grouped_purchased.CustomerID.astype('category', categories = customers).cat.codes 
    cols = grouped_purchased.StockCode.astype('category', categories = products).cat.codes 
    purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
    
    return [item_table, purchases_sparse, customers, products, quantity]

def split_data_mask(ratings, pct_test=0.2):
    # 1. Make copy of original set to be test set
    test_set = ratings.copy()
    # 2. Store test set as binary preference matrix
    test_set[test_set != 0] = 1
    # 3. Make copy of original data to our training_set
    training_set = ratings.copy()
    # 4. Find indices where ratings data interaction exists, and zip these pairs together
    nonzero_inds = training_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    random.seed(0)
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs)))
    samples = random.sample(nonzero_pairs, num_samples)
    
    # Create user, item indices
    user_inds = [index[0] for index in samples]
    item_inds = [index[1] for index in samples]
    training_set.eliminate_zeros()
    
    return [training_set, test_set, list(set(user_inds))]



#
# Splits up the preprocessed code into train/test/validation splits for recommender system
#
def split_data(df, status):
    start = time.time()

    df_train = df[df.invoicedate < '2011-10-09']
    df_train = df_train.reset_index(drop=True)
    df_val = df[(df.invoicedate >= '2011-10-09') & 
                (df.invoicedate <= '2011-11-09') ]
    
    df_val = df_val.reset_index(drop=True)
    df_test = df[df.invoicedate >= '2011-11-09']
    df_test = df_test.reset_index(drop=True)

    if (status):
        printGreen('✔ Split Data\t\t{0:.1f}s'.format(time.time() - start))
    
    # Pack data
    return [df_train, df_test, df_val]


#
# User Input Validation Functions
#
def validate_customer_id(customer_id):
    # Check if input is in range or is valid type
    if not (isinstance(customer_id, int) and customer_id >= 0 and customer_id <= 4338):
        printRed('✘ Invalid value for customer ID: "' + str(customer_id) + '"')
        printRed('✘ Input is not a valid integer between [0, 4338]')
        sys.exit(1)
    return 0

def validate_length(length):
    # Check whether the list length is valid
    if not (isinstance(length, int) and length >= 1 and length <= 4339):
        printRed('Invalid value for list length: "' + str(length) + '"')
        printRed('Input is not a valid integer between [1, 4339]')
        sys.exit(1)
    return 0

def validate_recommendations(recommendations_number):
    if not (isinstance(recommendations_number, int) and recommendations_number >= 1 and recommendations_number <= 5):
        printRed('Invalid value for recommendations number: "' + str(recommendations_number) + '"')
        printRed('Input is not a valid integer between [1, 5]')
        sys.exit(1)
    return 0

def validate_input(customer_id, recommendations_number):
    validate_customer_id(customer_id)
    validate_recommendations(recommendations_number)
    return 0

def search_customer(customer_id, df, table):
    # Validate user customer_id input
    validate_customer_id(customer_id)
    
    # Find associated customer from input through lookup table
    customer_id_r = table[customer_id]

    customer_df = df.loc[df['customerid'] == customer_id_r]
    amount_spent = 0.00
    list_of_items = []
    country = ""
    len(customer_df)

    for index, row in customer_df.iterrows():
        amount_spent += row['unitprice']
        list_of_items.append(row['description'])
        if country == "":
            country = row['country']
    
    # Display Customer Information Search result
    display_customer_information(customer_id=customer_id_r,
                                 country=country,
                                 items=list_of_items,
                                 amount_spent=amount_spent)
    return 0

def list_customers(length, df, table):
    validate_length(length)
    for i in range(length):
        customer_id_r = table[i]
        customer_df = df.loc[df['customerid'] == customer_id_r]
        amount_spent = 0.00
        list_of_items = []
        country = ""
        len(customer_df)
        for index, row in customer_df.iterrows():
            amount_spent += row['unitprice']
            list_of_items.append(row['description'])
            if country == "":
                country = row['country']
        
        # Display Customer
        display_customer_information(customer_id=customer_id_r,
                                country=country,
                                items=list_of_items,
                                amount_spent=amount_spent)
    return 0

# Stores Customer table as pickle file
def customer_table():
    customer_id_search = invert_dict(customer_id)
    table_storage = open('../data/final/df_customer_table.pkl', "wb")
    pickle.dump(customer_id_search, table_storage)
    table_storage.close()

def get_items_purchased(customer_id, mf_train, customers_list, products_list, item_lookup):
    cust_ind = np.where(customers_list == customer_id)[0][0] # Returns the index row of our customer id
    purchased_ind = mf_train[cust_ind,:].nonzero()[1] # Get column indices of purchased items
    prod_codes = products_list[purchased_ind] # Get the stock codes for our purchased items
    return item_lookup.loc[item_lookup.StockCode.isin(prod_codes)]


def main():
    # df = process_data(status=True)
    # table_file = open('../data/final/df_customer_table.pkl', "rb")
    # customer_table = pickle.load(table_file)
    # search_customer(3, df, customer_table)
    # list_customers(10, df, customer_table)
    
    data = preprocess_data_rec_engine(status=False)

    item_table = data[0]
    p_sparse = data[1]
    customers = data[2]
    products = data[3]
    quantity = data[4]


    tti = split_data_mask(p_sparse, pct_test = 0.2)
    
    product_training_set = tti[0]
    product_test_set = tti[1]
    product_user_altered = tti[2]

    vecs = implicit_weighted_als(product_training_set, 
                                 lambda_val = 0.1, 
                                 alpha = 15, 
                                 iterations = 1, 
                                 rank_size = 20)
    
    # 
    user_vecs = vecs[0]
    item_vecs = vecs[1]

    alpha = 15
    customers_arr  = np.array(customers) # Array of customer IDs from the ratings matrix
    products_arr = np.array(products) # Array of product IDs from the ratings matrix

    print(get_items_purchased(12346, product_training_set, customers_arr, products_arr, item_table))

    




if __name__ == '__main__':
    main()

