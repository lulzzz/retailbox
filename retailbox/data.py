# -*- coding: utf-8 -*-
import sys
import time
import pandas as pd
import pickle
import numpy as np
import os.path

from output import printGreen, printRed, display_customer_information

# Store CustomerIDs
customer_id = {}
customer_id_search = {}

def invert_dict(c):
    inv = {v: k for k, v in c.items()}
    return inv

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

# Splits up the preprocessed code into train/test/validation splits for recommender system
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


def main():
    df = process_data(status=True)
    table_file = open('../data/final/df_customer_table.pkl', "rb")
    customer_table = pickle.load(table_file)
    
    # search_customer(3, df, customer_table)


    list_customers(10, df, customer_table)

if __name__ == '__main__':
    main()

