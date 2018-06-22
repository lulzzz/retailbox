# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from tqdm import tqdm

'''
Process

1. Read in data
2. 

'''

def process_data():
    # Read Data, create dataframe
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

def main():
    # 1. Get Data
    df = process_data()

    # 2. Split dataset
    df_train = df[df.invoicedate < '2011-10-09']
    df_val = df[(df.invoicedate >= '2011-10-09') & (df.invoicedate <= '2011-11-09')]
    df_test = df[df.invoicedate >= '2011-11-09']

    # 3. Get Baselines
    top = df_train.stockcode.value_counts().head(5).index.values
    num_groups = len(df_val.invoiceno.drop_duplicates())
    base = np.tile(top, num_groups).reshape(-1, 5)
    print(base)

    return None

main()



