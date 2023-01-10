# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse 
import os
import numpy as np
import pandas as pd 
import json 
from datetime import datetime 
import time 
from xgboost import XGBClassifier 
import json 
from sklearn import metrics

DATA_FOLDER = "data/mnist/imputed/v12"
ORI_DATA_FOLDER = "data/mnist/original"
SAVED_FOLDER = 'v12_xgboost_gain' 

get_imputed_data_path = lambda train_or_test, sub_folder, algo: \
        os.path.join(DATA_FOLDER, sub_folder, '{}_{}.csv'.format(train_or_test, algo))
get_y_data_path = lambda train_or_test, sub_folder: \
        os.path.join(DATA_FOLDER, sub_folder, 'y_{}.csv'.format(train_or_test))

def main(args): 
    '''
    Calculate RMSE of original data set and the imputed one (after rescaled) for all the imputation sample 
    Args:
        - algo_name: algorithm name to calculate the rmse
        - train_or_test: rmse was calcualted on train set or test set 
    Return: 
        - rmse
    '''
    accuracies = {}

    algo_name = args.algo_name

    folders =  os.listdir(DATA_FOLDER) 
    #folders = ['threshold_50_deletedWidthHeightPc_5050_noImagePc_50']

    for folder_name in folders:
        print(folder_name)
        data_path = os.path.join(DATA_FOLDER, folder_name)
        if os.path.isdir(data_path):
            ##--------------------------------

            Xtrain = pd.read_csv(get_imputed_data_path("train", folder_name, algo_name)).to_numpy()
            Xtest  = pd.read_csv(get_imputed_data_path("test" , folder_name, algo_name)).to_numpy()

            ytrain = pd.read_csv(get_y_data_path("train", folder_name)).to_numpy().ravel()
            ytest  = pd.read_csv(get_y_data_path("test" , folder_name)).to_numpy().ravel()

            ##--------------------------------
           # Xtrain = pd.read_csv(get_imputed_data_path("train", folder_name, algo_name)).to_numpy()[:5000, ]
           # Xtest  = pd.read_csv(get_imputed_data_path("test" , folder_name, algo_name)).to_numpy()[:1000, ]

           # ytrain = pd.read_csv(get_y_data_path("train", folder_name)).to_numpy().ravel()[:5000, ]
           # ytest  = pd.read_csv(get_y_data_path("test" , folder_name)).to_numpy().ravel()[:1000, ]
            ##--------------------------------

            start = time.time()
            model = XGBClassifier() 
            model.fit(Xtrain, ytrain) 

            ypred = model.predict(Xtest)
            
            acc = metrics.accuracy_score(ytest, ypred)
            print(round(time.time()-start)/60, 3)
            
            accuracies.update({folder_name: acc})

    #---------
    now = datetime.now()
    time_string  = now.strftime("%Y%m%d_%H-%M-%S")
    #---------
 
    print(accuracies)
    saved_path = os.path.join(DATA_FOLDER, "_acc_{}_{}.json".format(
        algo_name, 
        time_string))

    print("complete save result at {}".format(saved_path))
    with open(saved_path, "w") as f:
        json.dump(accuracies, f)
    
    return accuracies

    
if __name__=='__main__':
    #   Inputs for the main function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algo_name',
        choices=['impDi', 'softImpute', 'Gain'],
        default='Gain',
        type=str)
   
    args = parser.parse_args() 
    
    rmses = main(args)
