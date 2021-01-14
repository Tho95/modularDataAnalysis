# module for showing information about a dataframe


def general(X):
    # see the shape of training data:
    print('Shape of training data(rows, columns): ', X.shape, '\n')
def missing_value_per_column(X):

    # see the missing values per column
    missing_value_per_column = X.isnull().sum()

    missing_cols = [col for col in X.columns if X[col].isnull().sum() > 0]
    missing = []
    i = 0
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            missing.append(X[col].isnull().sum())
    for col in range(0, len(missing)):
        print(missing_cols[col], 'missing values: ',missing[col])
    print('\n', len(missing),' columns have missing values.')