# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from tqdm import tqdm
import preprocess as pre


def main():
    # 1. Get Data
    df = pre.process_data()

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



