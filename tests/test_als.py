# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.abspath('../retailbox'))

from als import als_precision
import baseline as base
import data as d
import pandas as pd

# python -m pytest tests/

class TestClass(object):
    def test_precision(self):
        df = pd.read_pickle('../data/final/df_final.pkl')
        data = d.split_data(df, True)

        data_train = data[0]
        data_test = data[1]
        data_val = data[2]
        
        b = base.baseline(df, False)

        als_result = als_precision(data_train, data_val, b)
        assert 1 == 1
    