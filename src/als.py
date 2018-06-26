from implicit.als import AlternatingLeastSquares
import baseline as base
import precision as p
import preprocess as pre
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

def als(df_train, df_val, baseline):
    # Constructing user-item matrix X, translate both
    # users and items into IDs, so we can map each user to a row of x
    # and an iterm to a column of X
    df_train_user = df_train[df_train.customerid != -1].reset_index(drop=True)
    customers = sorted(set(df_train_user.customerid))
    customers = {c: i for (i, c) in enumerate(customers)}
    df_train_user.customerid = df_train_user.customerid.map(customers)

    # apply same procedure to validation set
    #df_val.customerid = df_val.customerid.apply(lambda c: customers.get(c, -1))

    # use integer codes to construct the matrix X
    uid = df_train_user.customerid.values.astype('int32')
    iid = df_train_user.stockcode.values.astype('int32')
    ones = np.ones_like(uid, dtype='uint8')

    # Create a matrix in Compressed Storage Row format
    X_train = sp.csr_matrix((ones, (uid, iid)))

    #
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
    
    imp_baseline = baseline
    pred_all = als_U[uid_val].dot(als_I.T)
    top_val = (-pred_all).argsort(axis=1)[:, :5]
    imp_baseline[known_mask] = top_val
    prec = p.precision(val_indptr, val_items, imp_baseline)
    print(prec)
    return prec


def runner():
    df = pre.process_data()
    als