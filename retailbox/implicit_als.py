import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve


def implicit_weighted_als(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):
    
    conf = (alpha*training_set)
    num_user = conf.shape[0]
    num_item = conf.shape[1]
    
    rstate = np.random.RandomState(seed)
    
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size)))
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size)))
    
    X_ = sparse.eye(num_user)
    Y_ = sparse.eye(num_item)
    
    lambda_eye = lambda_val * sparse.eye(rank_size)
    
    for iter_step in range(iterations):
        # transpose
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        
        # Solve for X based on fixed Y
        for u in range(num_user):
            # to-do
            conf_samp = conf[u, :].toarray()
            pref = conf_samp.copy()
            pref[pref != 0]
            CuI = sparse.diags(conf_samp, [0])
            
            yTCuIY = Y.T.dot(CuI).dot(Y)
            yTCupu = Y.T.dot(CuI + Y_).dot(pref.T)
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu)
            
        
        for i in range(num_item):
            # to-do
            conf_samp = conf[:, i].T.toarray()
            pref = conf_samp.copy()
            CiI = sparse.diags(conf_samp, [0])
            xTCiIX = X.T.dot(CiI).dot(X)
            xTCiPi = X.T.dot(CiI + X_).dot(pref.T)
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)
    return [X, Y.T]

def main():
    print("Hello")

if __name__ == '__main__':
    main()