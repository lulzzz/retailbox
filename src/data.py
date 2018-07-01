# -*- coding: utf-8 -*-
import sys
import time
import pandas as pd
import pickle
import numpy as np
import os.path

from output import printGreen, printRed

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

    return df

# Splits up the preprocessed code into train/test/validation splits
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

def validateUserID(userID):
    # Check if input userID is valid
    