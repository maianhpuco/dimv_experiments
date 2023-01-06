import os
import numpy as np
import pandas as pd 
import json 
from datetime import datetime 
def mse(a, b):
    return (np.square(a-b)).mean(axis=None)

def calc_rmse(sub_folder, train_or_test):
    
    get_Xpath = lambda train_test, algo: \
            os.path.join(root, sub_folder, '{}_{}_Xrecon.csv'.format(train_or_test, algo))
    
    softImpute_Xpath = get_Xpath(train_or_test, 'softImpute')
    impDi_Xpath = get_Xpath(train_or_test, 'impDi')
    original_Xpath = os.path.join(
            root, 
            "../../processed",
            ''.join(["X", train_or_test, ".csv"])) 
    
    softImpute_imputed = pd.read_csv(softImpute_Xpath).to_numpy()
    impDi_imputed = pd.read_csv(impDi_Xpath).to_numpy()
    original_data = pd.read_csv(original_Xpath).to_numpy() 


    softImputed_mse = mse(softImpute_imputed, original_data)
    impDi_mse = mse(impDi_imputed, original_data) 

    result = {train_or_test: {"softImputed_mse": softImputed_mse, "impDi_mse": impDi_mse}} 
    return result 

def calc_rmse_pipeline(sub_folder):
    result = calc_rmse(sub_folder, "train")
    result.update(calc_rmse(sub_folder, "test"))
    print(result)
    return result  

def format_df(df):
    df = df.copy()
    df['sub_folder'] = df.index
    df['deletedWidthHeightPc'] = df['sub_folder'].apply(lambda x: x.split('_')[3])
    df['threshold'] = df['sub_folder'].apply(lambda x: x.split('_')[1])  
    _dict = {'4040': '40% - 40%', '5050':'50% - 50%' , '6060': '60% - 60%', '7070':"70% - 70%"}
    df['deletedWidthHeightPc'] = df['deletedWidthHeightPc'].replace(_dict) 
    df['softImpute_train_mse'] = df['train'].apply(lambda x: round(x.get('softImputed_mse'), 2))  
    
    df['impDi_train_mse'] = df['train'].apply(lambda x: round(x.get('impDi_mse'), 2))  
    df['softImpute_test_mse'] = df['test'].apply(lambda x: round(x.get('softImputed_mse'),2)) 
    df['impDi_test_mse'] = df['test'].apply(lambda x: round(x.get('impDi_mse'),2))   
    df['threshold'] = df['threshold'].apply(lambda x: x.rjust(2, '0'))    
    
    df.drop(columns=['train','test', 'sub_folder'], inplace = True)
    
    df.sort_values(['deletedWidthHeightPc', 'threshold'], inplace=True)
    
    df.rename(columns={
        "deletedWidthHeightPc": "Height - Weight Deleted Percentage", 
        "threshold" :"DIMV threshold parameter (percent)",
        "softImpute_train_mse": "softImpute's MSE on MNIST Train Set", 
        "impDi_train_mse": "DIMV's MSE on MNIST Train Set",
        "softImpute_test_mse": "softImpute's MSE on MNIST Test Set", 
        "impDi_test_mse": "DIMV's MSE on MNIST Test Set"
    }, inplace=True)
    df.reset_index(inplace=True, drop=True) 
    return  df 
    
if __name__=='__main__':
    root = '../../../data/mnist/imputed/v12/'
    mse_results = {}
    for sub_folder in os.listdir(root):
        if len(sub_folder.split("_")) >3 \
                and sub_folder.split("_")[-1]=='50':
            mse_results.update({sub_folder :calc_rmse_pipeline(sub_folder)}) 
    df = pd.DataFrame(mse_results).T 

    result_df = format_df(df)
    
    now = datetime.now()
    string_time = now.strftime("%Y-%m-%d-%H:%M")

    result_path = os.path.join(root, "../../mse", "".join([string_time, '.csv']))
    
    result_df.to_csv(result_path) 
    
