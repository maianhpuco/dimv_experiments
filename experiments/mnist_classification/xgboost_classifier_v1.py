import os
import sys  
import pandas as pd 
import numpy as np 
import time
import json 
from xgboost import XGBClassifier 


from datetime import datetime 
from sklearn import svm 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import pipeline 

from sklearn.kernel_approximation import Nystroem
VERSION = 'v12'
FOLDER = 'v12_xgboost'



def svm_prediction(X_train, y_train, X_test, y_test, name, root, sub_folder):
    start_prediction = time.time()
    print("Start prediction")
   # model = svm.SVC(C = 5, gamma = 0.05)
    model = XGBClassifier() 

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    pred_output_path = os.path.join(root, 'prediction_output/',FOLDER, sub_folder)
    acc_path = os.path.join(root,'accuracy', FOLDER)

    if not os.path.isdir(pred_output_path):
        os.mkdir(pred_output_path)
    if not os.path.isdir(acc_path):
        os.mkdir(acc_path)

    pred_file_path = lambda file: os.path.join(pred_output_path, file)

    ypred_df = pd.DataFrame(y_pred, columns=["ypred"]) 
    ypred_df.to_csv(pred_file_path('{}.csv'.format(name)))
    acc = metrics.accuracy_score(y_test, y_pred)
    duration = (time.time() - start_prediction)/60 
    print("{}  with Acc {} in Predition Time {} mins".format(name, acc, duration))
    return acc  

def svm_prediction_pipeline(root, sub_folder):
    print("updated code")
    path = os.path.join(root, "imputed/", VERSION, sub_folder)
    print("start reading data at path", path)

    get_Xpath = lambda train_test, algo: os.path.join(path, '{}_{}.csv'.format(train_test, algo))
    get_ypath = lambda train_test: os.path.join(path, 'y_{}.csv'.format(train_test))

    softImpute_Xtrain_path = get_Xpath('train','softImpute')
    softImpute_Xtest_path = get_Xpath('test','softImpute')
    softImpute_ytrain_path = get_ypath('train')
    softImpute_ytest_path = get_ypath('test')

    impDi_Xtrain_path = get_Xpath('train','impDi') 
    impDi_Xtest_path = get_Xpath('test','impDi') 
    impDi_ytrain_path = get_ypath('train')
    impDi_ytest_path = get_ypath('test')
#----------------

    start_reading = time.time()
    softImpute_Xtrain = pd.read_csv(softImpute_Xtrain_path).to_numpy()
    softImpute_ytrain = pd.read_csv(softImpute_ytrain_path)
    softImpute_Xtest = pd.read_csv(softImpute_Xtest_path).to_numpy()
    softImpute_ytest = pd.read_csv(softImpute_ytest_path)
    
    impDi_Xtrain = pd.read_csv(impDi_Xtrain_path).to_numpy()
    impDi_ytrain = pd.read_csv(impDi_ytrain_path)
    impDi_Xtest = pd.read_csv(impDi_Xtest_path).to_numpy()
    impDi_ytest = pd.read_csv(impDi_ytest_path)

    softImpute_ytrain = softImpute_ytrain.values.ravel()
    softImpute_ytest = softImpute_ytest.values.ravel()
    impDi_ytrain = impDi_ytrain.values.ravel()
    impDi_ytest = impDi_ytest.values.ravel()

#    start_reading = time.time()
#    softImpute_Xtrain = pd.read_csv(softImpute_Xtrain_path).to_numpy()[:1000,]
#    softImpute_ytrain = pd.read_csv(softImpute_ytrain_path)
#    softImpute_Xtest = pd.read_csv(softImpute_Xtest_path).to_numpy()[:1000,]
#    softImpute_ytest = pd.read_csv(softImpute_ytest_path)
#    
#    
#    impDi_Xtrain = pd.read_csv(impDi_Xtrain_path).to_numpy()[:1000,]
#    impDi_ytrain = pd.read_csv(impDi_ytrain_path)
#    impDi_Xtest = pd.read_csv(impDi_Xtest_path).to_numpy()[:1000,]
#    impDi_ytest = pd.read_csv(impDi_ytest_path)
#
#    softImpute_ytrain = softImpute_ytrain.values.ravel()[:1000, ]
#    softImpute_ytest = softImpute_ytest.values.ravel()[:1000,]
#    impDi_ytrain = impDi_ytrain.values.ravel()[:1000,]
#    impDi_ytest = impDi_ytest.values.ravel()[:1000, ]
    print("data shape:", impDi_Xtrain.shape)
  
    print("complete reading data in subfoler {} \n  after: {} second".format(
        sub_folder, 
        time.time()-start_reading)
    )
    
    softImpute_acc = svm_prediction(
             softImpute_Xtrain, 
             softImpute_ytrain, 
             softImpute_Xtest, 
             softImpute_ytest, 
             "SoftImpute", 
             root, 
             sub_folder 
            )
    impDi_acc = svm_prediction(
             impDi_Xtrain, 
             impDi_ytrain, 
             impDi_Xtest, 
             impDi_ytest, 
             "impDi", 
             root, 
             sub_folder
            )

    acc = {
            sub_folder: {
                "softImpute": softImpute_acc, 
                "impDi" : impDi_acc 
            }
        }

    #save acc 
    #acc_path = os.path.join(root,'accuracy', FOLDER)
    #if not os.path.isdir(acc_path):
    #    os.mkdir(acc_path)
    #acc_file_path = os.path.join(acc_path, "".join([sub_folder, '.json']))

    print("acc ", acc)
   # with open(acc_file_path,'w') as f:
        #json.dump(acc, f)
    return acc

if __name__ == '__main__':
    root = '../../data/mnist/'
    accuracies = {}
    now = datetime.now()
    time_string  = now.strftime("%Y-%m-%d_%H-%M-%S")

    acc_path = os.path.join('../../data/mnist/accuracy/',FOLDER, 'acc_{}.csv'.format(
        time_string))
    sub_folders = os.listdir(os.path.join(root, 'imputed', VERSION))
     
    count = 1 
    exps = [sub_folder for sub_folder in sub_folders \
            if  (len(sub_folder.split("_")) >  5) \
            and (sub_folder.split("_")[-1] == '50') \
            and (sub_folder.split("_")[-3]) == '6060']
    exps = pd.Series(exps).sort_values().to_numpy()
    

        
    for sub_folder in exps:
        print("----No: ",count, len(exps))
        count+=1
        print("-------------------------------")
        print("Starting {}".format(sub_folder))
        acc = svm_prediction_pipeline(root, sub_folder)
        accuracies.update({sub_folder: acc})

    df = pd.DataFrame(accuracies)
    df.to_csv(acc_path)

    
        #with open(acc_path,'a') as f:
        #    json.dump(accuracies, f)


    
