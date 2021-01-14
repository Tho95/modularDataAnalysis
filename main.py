# 14.01.2021 Thomas Hübscher
#A modular approach to evaluate big csv files. While creating the programm the Iowa house data file will be
#the sample data. Later we will test with other csv files and refine compability

#import general modules
import pandas as pd
from sklearn.model_selection import train_test_split

#import own modules
import read_csv
import seperateTarget
import infoDataFrame
import visualizeInfo


############################################################################################################################
#Preperation of the files

#Here are all configurations to be done for module readCSV:
test_file_path = 'test.csv'
train_file_path = 'train.csv'
replaceID = True
#Here are all configurations to be done for module seperateTarget:
target = 'SalePrice'
#Here are all configurations for train_test_split
train_size = 0.8
test_size = 0.2

############################################################################################################################





X, X_test = read_csv.readFile(train_file_path, test_file_path, replaceID=True)
X, y = seperateTarget.seperate(X,target)

#seperate training data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_size, test_size=test_size,random_state=0)

n_rows,n_cols = infoDataFrame.general(X)
namemissing, missing = infoDataFrame.missing_value_per_column(X)   #return array
object_cols, numeric_cols= infoDataFrame.colType(X)

visualizeInfo.visualize(n_rows,n_cols,object_cols, numeric_cols, namemissing, missing)
########################################################################################################################################
