# Module to serperate the target variable from the dataframe and delete target with nan or 0

import pandas as pd

def seperate(X,target):
    #we remove rows with missing target value (nan)
    X.dropna(axis=0, subset= [target], inplace=True)

    # remove target column from X
    y = X.SalePrice
    X.drop(target, axis=1, inplace=True)

    return X, y