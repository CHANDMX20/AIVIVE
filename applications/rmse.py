import sys

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import math

import os
from os import listdir
from os.path import join, isfile

# get the gene feature columns
features = pd.read_csv('/path/to/final_rat_genes.csv').PROBEID.unique()
print(len(features))
# true values
vitro_vivo= pd.read_csv('/path/to/vitro_vivo_train_test.csv', low_memory=False)


# calculate rmse value
def generated_rmse(predicted, filename, org=vitro_vivo, features=features):
    predicted = predicted.reset_index(drop=True)
    result = predicted.iloc[:, :-len(features)]
    rmsevalues = []
    for i in range(len(result)):
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        mse = mean_squared_error(true_value, predicted_value)
        rmse = math.sqrt(mse)
        rmsevalues.append(rmse)
    result['rmse'] = rmsevalues
    result.to_csv(filename)
       

# resultPath
path = '/path/to/results'
# final model number
number='9962160'


### generated test predictions 
dataPath = path + '/predictions_decoded/test_set'
rmsePath = path + '/performance/rmse/test'
# vivo
testVivo = pd.read_csv(dataPath + '/combined_opt_gan_test' + number + '_Vivo.csv')
generated_rmse(testVivo, rmsePath+'/' + number + '_OptVivo.csv')
# vitro
#testVitro = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VitroGenerator.csv')
#generated_rmse(testVitro, rmsePath+'/' + number + '_VitroGenerator.csv')


### train
#dataPath = path + '/predictions_decoded/train'
#rmsePath = path + '/performance/rmse/train'
# vivo
#trainVivo = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VivoGenerator.csv')
#generated_rmse(trainVivo, rmsePath+'/' + number + '_VivoGenerator.csv')
# vitro
#trainVitro = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VitroGenerator.csv')
#generated_rmse(trainVitro, rmsePath+'/' + number + '_VitroGenerator.csv')
