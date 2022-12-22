import os
import numpy as np
import pandas as pd 

def mse(a, b):
    return (np.square(a-b)).mean(axis=None)

if __name__=='__main__':
    root = '../../data/mnist/imputed/'
    print(os.listdir(root))
    for sub_folder in os.listdir(root):
        if len(sub_folder.split("_"))!=6:
            pass 
        
