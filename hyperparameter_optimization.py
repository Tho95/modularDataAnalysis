#With this function we want to do a hyper_parameter optimization for our xg_boost models

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pandas as pd
from sklearn.impute import SimpleImputer
import math

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities=False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)

    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)

    return fitted_model, pred


#after many  time intesive computations the following parameters seem to be good(best found so far)
#n_estimators=1400,colsample_bytree=0.9,max_depth=20, reg_alpha=1.5,reg_lambda=1.1,subsample=0.7
def xgboost_hyper(numerical_X_train, numerical_X_valid, y_train, y_valid):
    my_imputer = SimpleImputer(strategy='median')

    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(numerical_X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(numerical_X_valid))

    imputed_X_train.columns = numerical_X_train.columns
    imputed_X_valid.columns = numerical_X_valid.columns


    model = XGBRegressor()
    param_grid = {
        'n_estimators': [1000,1400],
        'colsample_bytree': [0.7,0.9,1],
        'max_depth': [10,15,20],
        'reg_alpha': [1.1, 1.5],
        'reg_lambda': [1.1, 1.5],
        'subsample': [0.6,0.7,0.8]
    }

    model, pred = algorithm_pipeline(imputed_X_train, imputed_X_valid, y_train, y_valid, model,
                                 param_grid, cv=5)

    # Root Mean Squared Error
    print(model.best_params_)
    print((-model.best_score_))
    print('root Mean square error', math.sqrt(-model.best_score_))
    print(model.best_params_)

