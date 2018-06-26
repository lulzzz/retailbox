# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import numpy as np
import os.path

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

    return df
