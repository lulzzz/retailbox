# -*- coding: utf-8 -*-
import numpy as np
import preprocess as pre
import precision as p

# See where a transaction finishes, and where the
# next one starts
def group_indptr(df):
    # At each row index, we compare the current index
    # with the previous one, and if it is different
    # we record the index. We use this using the shift() method
    indptr, = np.where(df.invoiceno != df.invoiceno.shift())
    indptr = np.append(indptr, len(df)).astype('int32')
    return indptr

def baseline(df):
    df_train = df[df.invoicedate < '2011-10-09']
    df_train = df_train.reset_index(drop=True)
    df_val = df[(df.invoicedate >= '2011-10-09') & 
                (df.invoicedate <= '2011-11-09') ]
    
    df_val = df_val.reset_index(drop=True)
    df_test = df[df.invoicedate >= '2011-11-09']
    df_test = df_test.reset_index(drop=True)

    top = df_train.stockcode.value_counts().head(5).index.values

    val_indptr = group_indptr(df_val)
    num_groups = len(val_indptr) - 1
    baseline = np.tile(top, num_groups).reshape(-1, 5)
    
    val_items = df_val.stockcode.values
    prec = p.precision(val_indptr, val_items, baseline)
    print(prec)
    return prec

def runner():
    df = pre.process_data()
    baseline(df)
