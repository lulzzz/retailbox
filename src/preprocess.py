# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import numpy as np
import os.path

# CustomerID Storage

## DEPRECIATED


customer_id = {}

# Preprocesses code, and returns a dataframe of clean data
def process_data():
    # Read Data, create dataframe
    file_path = '../data/processed/df_retail.bin'
    df = None
    if os.path.exists(file_path) == False:
        df = pd.read_excel('../data/raw/Online Retail.xlsx')
        # # Save df into pickle file
        with open('../data/processed/df_retail.bin', 'wb') as f_out:
            pickle.dump(df, f_out)
    
    with open('../data/processed/df_retail.bin', 'rb') as f_in:
        df = pickle.load(f_in)

    # Clean Data, lowercase, remove "return" transaction, and unknown user transactions
    df.columns = df.columns.str.lower()
    df = df[~df.invoiceno.astype('str').str.startswith('C')].reset_index(drop=True)
    df.customerid = df.customerid.fillna(-1).astype('int32')

    # Encode item IDs with integers
    stockcode_values = df.stockcode.astype('str')
    stockcodes = sorted(set(stockcode_values))
    stockcodes = {c: i for (i, c) in enumerate(stockcodes)}
    df.stockcode = stockcode_values.map(stockcodes).astype('int32')

    # Store Customers id DB
    start = time.time()

    i = 0
    counter = 0

    while counter != 532620:
        #     print(df['CustomerID'][x])
        if df['customerid'][counter] not in user_id and df['customerid'][counter] != -1 and df['customerid'][counter] != None:
            user_id[df['customerid'][counter]] = i
            #print(i)
            i += 1
        #print(counter)
        counter += 1




    return df

# Splits up the preprocessed code into train/test/validation splits
# todo add starttime, and status prompts
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

    return [df_train, df_test, df_val]

def validateCustomerID(customerID):
    # Check if user ID is valid

def main():
    print("Hello")

main()
