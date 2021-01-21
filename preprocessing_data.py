# a module to do better preprocessing of the data (better impuation, encoding of numerical data)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error


def preprocess(X_train, X_val, y_train, y_val):
    median_imputed = list(['LotFrontage'])  # numerical
    constant_zero_imputed = list(['MasVnrArea'])  # numerical
    most_frequent_imputed = list(['Electrical'])  # categorial
    none_nominal_imputed = list(['MasVnrType', 'GarageType', 'MiscFeature'])  # strategy constant with None as
    # filler (later nominal) (categorial)
    none_ordinal_imputed = list(['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
                                 'Fence'])  # strategy constant with None as filler (later ordinal as zero)

    object_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
    categorial_cols = list([cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
                            X_train[cname].dtype == "object"])

    numerical_cols = list([cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']])

    high_cardinality_cols = list(set(object_cols) - set(categorial_cols))

    nominal_cols = list(['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2',
                         'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating',
                         'Electrical', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'])
    ordinal_cols = list(set(categorial_cols) - set(nominal_cols))

    print(ordinal_cols)
    #need to order the dictionaries for the attributes like in ordinal_cols[] --> Probelm order is different every time (maybe random state or maps)
    #solutio for now: reorder the array ordinal_cols to fit the order of categories:

    ordinal_cols = ['ExterCond', 'KitchenQual', 'Street', 'Alley', 'BsmtExposure', 'Utilities', 'BsmtQual', 'BsmtFinType1', 'HeatingQC',
                    'PoolQC', 'BsmtCond', 'Functional', 'CentralAir', 'ExterQual', 'PavedDrive', 'GarageFinish', 'GarageCond',
                    'BsmtFinType2', 'GarageQual', 'Fence', 'FireplaceQu']

    ordinal_encoder = OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex'],                                        #ExterCond
                                                 ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                                        #KitchenQual
                                                 ['Grvl', 'Pave'],                                                      #Street
                                                 ['missing_value', 'Grvl', 'Pave'],                                     #Alley
                                                 ['missing_value', 'No', 'Mn', 'Av', 'Gd'],                             #BsmtExposure
                                                 ['ELO', 'NoSeWa', 'Nosewr', 'AllPub'],                                 #Utilities
                                                 ['missing_value', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],                       #BsmtQual
                                                 ['missing_value', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],           #BsmtFinType1
                                                 ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                                        #HeatingQC
                                                 ['missing_value', 'Fa', 'TA', 'Gd', 'Ex'],                             #PoolQC
                                                 ['missing_value', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],                       #BsmtCond
                                                 ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],          #Functional
                                                 ['N', 'Y'],                                                            #CentralAir
                                                 ['Po', 'Fa', 'TA', 'Gd', 'Ex'],                                        #ExterQual
                                                 ['N', 'P', 'Y'],                                                       #PavedDrive
                                                 ['missing_value', 'Unf', 'RFn', 'Fin'],                                #GarageFinish
                                                 ['missing_value', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],                       #GarageCond
                                                 ['missing_value', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],           #BsmtFinType2
                                                 ['missing_value', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],                       #GarageQual
                                                 ['missing_value', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],                   #Fence
                                                 ['missing_value', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])                      #FireplaceQu

    '''['GarageCond', 'GarageQual', 'FireplaceQu', 'BsmtQual', 'CentralAir', 'ExterCond', 'BsmtFinType2', 'BsmtFinType1',
     'Alley', 'BsmtCond', 'GarageFinish', 'KitchenQual', 'Street', 'HeatingQC', 'Fence', 'PavedDrive', 'ExterQual',
     'BsmtExposure', 'Functional', 'PoolQC', 'Utilities']'''


    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    imputer_median = SimpleImputer(strategy='constant')
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')
    imputer_constant_zero = SimpleImputer(strategy='constant', fill_value=0)
    imputer_constant_none = SimpleImputer(strategy='constant', fill_value='missing_value')

    numerical_pipe_median = Pipeline([('imputer', imputer_median)])
    numerical_pipe_constant_zero = Pipeline([('imputer', imputer_constant_zero)])
    categorial_pipe_most_frequent_ohe = Pipeline([('imputer', imputer_most_frequent), ('encoder', one_hot_encoder)])
    categorial_pipe_constant_none_ohe = Pipeline([('imputer', imputer_constant_none), ('encoder', one_hot_encoder)])
    categorial_pipe_constant_none_ordinal = Pipeline([('imputer', imputer_constant_none), ('encoder', ordinal_encoder)])

    rest_cols = (
        list(set(X_train.columns) - set(median_imputed) - set(constant_zero_imputed) - set(most_frequent_imputed)
             - set(nominal_cols) - set(ordinal_cols) - set(high_cardinality_cols)))

    preprocessor = ColumnTransformer(   remainder = 'passthrough',
        transformers=[('num1', numerical_pipe_median, median_imputed),
                      ('num2', numerical_pipe_constant_zero, constant_zero_imputed),
                      ('cat1', categorial_pipe_most_frequent_ohe, most_frequent_imputed),
                      ('cat2', categorial_pipe_constant_none_ohe, nominal_cols),
                      ('cat3', categorial_pipe_constant_none_ordinal, ordinal_cols),
                      ('highcard', categorial_pipe_constant_none_ohe, high_cardinality_cols),
                      ('rest', numerical_pipe_median, rest_cols)
                      ])
    modelxgb = XGBRegressor(colsample_bytree=0.9,max_depth=20, n_estimators=1400, reg_alpha=1.5, rag_lambda=1.1, subsample=0.7, random_state=0)
    #hyperparameter from hyperparameter optimization with numerical model. Because a new comutation ould take to long for now
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', modelxgb)])

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_val)

    print('MAE:', mean_absolute_error(y_val, predictions))

