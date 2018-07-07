# -*- coding: utf-8 -*-
from numba import njit

'''
The logic of this function is straight forward, 

For each transaction, we check how many items we predicted correctly,
which is the 'tp' variable

At the end, we divide 'tp' by the total number of predictions, which
is the size of the prediction matrix, that the number of transactions
times 5 in our case.

@njit is a decorator that tells numba to optimize the code, where it 
analyzes the code using the JIT compiler (just-in-time) to translate the
function to native code

When the function is compiled, it runs multiple orders of magnitude faster
comparable to native code written in C
'''

@njit 
def precision(group_indptr, true_items, predicted_items):
    tp = 0 # True # of predictions

    n, m = predicted_items.shape # total number of predictions our system made

    for i in range(n):
        group_start = group_indptr[i]
        group_end = group_indptr[i + 1]
        
        # Groups a single transaction that manifested
        # in multiple rows in the CSV
        group_true_items = true_items[group_start:group_end]
        
        # Checking precision
        for item in group_true_items:
            for j in range(m):
                if item == predicted_items[i, j]:
                    tp = tp + 1
                    continue

    # return the # of correct predictions / total predictions
    return tp / (n * m)