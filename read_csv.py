# Module for reading in a csv file to a dataframe

import pandas as pd

def readFile(file_train, file_test, replaceID):
    if replaceID == True:
        X = pd.read_csv(file_train, index_col='Id')
        X_test = pd.read_csv(file_test, index_col='Id')
    else:
        X = pd.read_csv(file_train)
        X_test = pd.read_csv(file_test)

    return X, X_test

