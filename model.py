#different models for data evaluation

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

def numerical_data_xgboost(numerical_X_train, numerical_X_valid,y_train,y_valid):
    my_imputer = SimpleImputer(strategy='median')

    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(numerical_X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(numerical_X_valid))

    imputed_X_train.columns = numerical_X_train.columns
    imputed_X_valid.columns = numerical_X_valid.columns

    my_model = XGBRegressor(random_state=0,n_estimators=500)

    my_model.fit(imputed_X_train, y_train)

    predictions = my_model.predict(imputed_X_valid)

    mae_1 = mean_absolute_error(y_valid, predictions)

    print("Mean Absolute Error: 1st. Model", mae_1)

    return mae_1

def numerical_data_xgboost_n_estimators(numerical_X_train, numerical_X_valid,y_train,y_valid,n_esimators):
    maes = []
    my_imputer = SimpleImputer(strategy='median')

    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(numerical_X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(numerical_X_valid))

    imputed_X_train.columns = numerical_X_train.columns
    imputed_X_valid.columns = numerical_X_valid.columns
    print(n_esimators)
    for n in n_esimators:
        model = XGBRegressor(random_state=0,n_estimators=n)

        model.fit(imputed_X_train, y_train)

        predictions = model.predict(imputed_X_valid)

        mae = mean_absolute_error(y_valid, predictions)

        maes.append(mae)
        print(mae)
    return maes