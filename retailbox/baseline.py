# -*- coding: utf-8 -*-
import numpy as np
import data as d
import precision as p
import pandas as pd

# Group Transactions
# See where a transaction finishes, and where the  next one starts
def group_indptr(df):
    # At each row index, we compare the current index
    # with the previous one, and if it is different
    # we record the index. We use this using the shift() method
    indptr, = np.where(df.invoiceno != df.invoiceno.shift())
    indptr = np.append(indptr, len(df)).astype('int32')
    return indptr

def baseline(df, status):
    # Split Data
    data = d.split_data(df, status)
    
    # Get Training, Test, and Validation sets
    df_train = data[0]
    df_test = data[1]
    df_val = data[2]

    top = df_train.stockcode.value_counts().head(5).index.values

    val_indptr = group_indptr(df_val)
    num_groups = len(val_indptr) - 1
    baseline = np.tile(top, num_groups).reshape(-1, 5)
    
    return baseline

def baseline_precision(df, status):
    # Split Data
    data = d.split_data(df, status)
    
    # Get Training, Test, and Validation sets
    df_train = data[0]
    df_test = data[1]
    df_val = data[2]

    top = df_train.stockcode.value_counts().head(5).index.values

    val_indptr = group_indptr(df_val)
    num_groups = len(val_indptr) - 1
    baseline = np.tile(top, num_groups).reshape(-1, 5)
    
    val_items = df_val.stockcode.values

    # Get Precision metric
    prec = p.precision(val_indptr, val_items, baseline)
    return prec
    
    

def main():
    df = pd.read_pickle('../data/final/df_final.pkl') 
    print(baseline_precision(df, True))
    print(baseline(df, True))
    

if __name__ == '__main__':
    main()

