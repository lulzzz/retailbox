# -*- coding: utf-8 -*-
from implicit.als import AlternatingLeastSquares
import baseline as base
import pandas as pd
import precision as p
import data as d
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

def als_precision(df_train, df_val, baseline):
    # Constructing user-item matrix X, translate both
    # users and items into IDs, so we can map each user to a row of x
    # and an iterm to a column of X
    df_train_user = df_train[df_train.customerid != -1].reset_index(drop=True)
    customers = sorted(set(df_train_user.customerid))
    customers = {c: i for (i, c) in enumerate(customers)}
    df_train_user.customerid = df_train_user.customerid.map(customers)

    # apply same procedure to validation set
    # df_val.customerid = df_val.customerid.apply(lambda c: customers.get(c, -1))

    # use integer codes to construct the matrix X
    uid = df_train_user.customerid.values.astype('int32')
    iid = df_train_user.stockcode.values.astype('int32')
    ones = np.ones_like(uid, dtype='uint8')

    # Create a matrix in Compressed Storage Row format
    X_train = sp.csr_matrix((ones, (uid, iid)))


    df_val.customerid = df_val.customerid.apply(lambda c: customers.get(c, -1))
    uid_val = df_val.drop_duplicates(subset='invoiceno').customerid.values
    known_mask = uid_val != -1
    uid_val = uid_val[known_mask]

    # Training
    item_user = X_train.T.tocsr()
    als = AlternatingLeastSquares(factors=128, regularization=0.000001)
    als.fit(item_user)

    als_U = als.user_factors
    als_I = als.item_factors
    
    imp_baseline = baseline.copy()

    pred_all = als_U[uid_val].dot(als_I.T)
    top_val = (-pred_all).argsort(axis=1)[:, :5]
    val_indptr = base.group_indptr(df_val)
    val_items = df_val.stockcode.values

    imp_baseline[known_mask] = top_val
    prec = p.precision(val_indptr, val_items, imp_baseline)
    return prec


def main():
    # Load Dataframe
    df = pd.read_pickle('../data/final/df_final.pkl')

    # Split data
    data = d.split_data(df, True)

    data_train = data[0] # training
    data_test = data[1]  # test
    data_val = data[2]   # validation

    # Get Baseline
    b = base.baseline(df, False)
    
    # Get ALS
    print(als_precision(data_train, data_val, b))

if __name__ == '__main__':
    main()

def implicit_weighted_ALS(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):
    conf = (alpha*training_set)
    num_user = conf.shape[0]
    num_item = conf.shape[1]
    

    rstate = np.random.RandomState(seed)
    
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size)))
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size)))
    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)

    lambda_eye = lambda_val * sparse.eye(rank_size) 
    
    for iter_step in range(iterations): # Iterate back and forth between solving X given fixed Y and vice versa
        # Compute yTy and xTx at beginning of each iteration to save computing time

        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Being iteration to solve for X based on fixed Y
        for u in range(num_user):
            conf_samp = conf[u,:].toarray() # Grab user row from confidence matrix and convert to dense
            pref = conf_samp.copy() 
            pref[pref != 0] = 1 # Create binarized preference vector 
            CuI = sparse.diags(conf_samp, [0]) # Get Cu - I term, which is just CuI since we never added 1
            yTCuIY = Y.T.dot(CuI).dot(Y) # This is the yT(Cu-I)Y term 
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) # This is the yTCuPu term, where we add the eye back in
                                                      # Cu - I + I = Cu
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper  
        
        # Begin iteration to solve for Y based on fixed X 
        for i in range(num_item):
            conf_samp = conf[:,i].T.toarray() # transpose to get it in row format and convert to dense
            pref = conf_samp.copy()
            pref[pref != 0] = 1 # Create binarized preference vector
            CiI = sparse.diags(conf_samp, [0]) # Get Ci - I term, which is just CiI since we never added 1
            xTCiIX = X.T.dot(CiI).dot(X) # This is the xT(Cu-I)X term
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) # This is the xTCiPi term
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)
            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi, equation 5 from the paper

    # End iterations
    return X, Y.T 