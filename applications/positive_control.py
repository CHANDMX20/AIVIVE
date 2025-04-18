import sys

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import math

import os
from os import listdir
from os.path import join, isfile

# get the gene feature columns
features = pd.read_csv('/path/to/final_rat_genes.csv').PROBEID.unique()
print(len(features))

def biology_duplicate(data, filename, features=features):
    resultDf = pd.DataFrame(columns = ['BARCODE', 'targetId', 'cosine', 'rmse', 'mape'])
 
    data.insert(1, 'label', data['EXP_ID'].astype(str) + '_' + data['GROUP_ID'].astype(str))
    del data['Unnamed: 0']
 
    for group in data['label'].unique():
        sub = data[data['label'] == group]
        sub = sub.reset_index(drop=True)
        for i in range(len(sub)-1):
            sample1 = sub.loc[i, 'BARCODE']
            true_values1 = sub.loc[i, features].values
            for j in range(i+1, len(sub)):
                sample2 = sub.loc[j, 'BARCODE']          
                true_values2 = sub.loc[j, features].values
 
                cos = cosine_similarity(true_values1.reshape(1,-1), true_values2.reshape(1,-1))[0][0] 
                rmse = math.sqrt(mean_squared_error(true_values1, true_values2))
                mape = abs((true_values1 - true_values2)/true_values1).mean()
 
                resultDf.loc[len(resultDf)] = [sample1, sample2, cos, rmse, mape]
                print(resultDf)
 
    resultDf.to_csv(filename)

resultPath = '/path/to/results/performance/positiveControl'

vivoTest = pd.read_csv('/path/to/vivo_test.csv')  #path to vivo test data
vivoTest_dose = vivoTest[vivoTest['DOSE_LEVEL'] != 'Control']

biology_duplicate(vivoTest_dose, resultPath+'/vivo_test_cosine_rmse_mape_biology_duplicates.csv')
 
#vivoTrain = pd.read_csv('/account/mansi.chandra/vitro_vivo/vivo_train.csv')
#biology_duplicate(vivoTrain, resultPath+'/vivo_train_cosine_rmse_mape_biology_duplicates.csv')
 
#vitroTest = pd.read_csv('/account/mansi.chandra/vitro_vivo/vitro_test.csv')
#biology_duplicate(vitroTest, resultPath+'/vitro_test_cosine_rmse_mape_biology_duplicates.csv')
 
#vitroTrain = pd.read_csv('/account/mansi.chandra/vitro_vivo/vitro_train.csv')
#biology_duplicate(vitroTrain, resultPath+'/vitro_train_cosine_rmse_mape_biology_duplicates.csv')

