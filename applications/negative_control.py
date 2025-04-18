import sys
#number=sys.argv[1]
#path=sys.argv[2]


import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# get the gene feature columns
features = pd.read_csv('/path/to/final_rat_genes.csv').PROBEID.unique()
print(len(features))
# true values
vitro_vivo= pd.read_csv('/path/to/vitro_vivo_train_test.csv', low_memory=False)  #path to real train and test data

#remove control treatments
vitro_vivo = vitro_vivo[vitro_vivo['DOSE_LEVEL'] != 'Control']

#calculate the cosine in vitro or vivo
def control_oneExp_cosine(data, filename, features=features):
    data.insert(1, 'label', data['EXP_ID'].astype(str) + '_' + data['GROUP_ID'].astype(str))
    data = data.reset_index(drop=True)
    
    df = pd.DataFrame()
    
    for i in range(len(data)-3):
        print(i)
        A = data.loc[i, features].values.reshape(1, -1)
        subdata = data.iloc[i+1:, :]
        B = subdata.loc[subdata.label != data.loc[i, 'label'], features].values
        result = cosine_similarity(A, B)
        df = pd.concat([df, pd.DataFrame(result.reshape(-1, 1))])
    
    df.to_csv(filename)
    #return df

#calculate the rmse in vitro or vivo
def control_oneExp_rmse(data, filename, features=features):
    #data.insert(1, 'label', data['EXP_ID'].astype(str) + '_' + data['GROUP_ID'].astype(str))
    data = data.reset_index(drop=True)
    
    df = pd.DataFrame()
    
    for i in range(len(data)-3):
        #print(i)
        A = data.loc[i, features].values.reshape(1, -1)
        subdata = data.iloc[i+1:, :]
        B = subdata.loc[subdata.label != data.loc[i, 'label'], features].values
        result = (np.square(np.tile(A, (len(B), 1)) - B)).mean(axis=1)
        df = pd.concat([df, pd.DataFrame(result.reshape(-1, 1))])
    
    df[0] = df[0].pow(1./2)
    df.to_csv(filename)
    return df

#calculate the mape in vitro or vivo
def control_oneExp_mape(data, filename, features=features):
    #data.insert(1, 'label', data['EXP_ID'].astype(str) + '_' + data['GROUP_ID'].astype(str))
    data = data.reset_index(drop=True)
    
    df = pd.DataFrame()
    
    for i in range(len(data)-3):
        print(i)
        A = data.loc[i, features].values.reshape(1, -1)
        subdata = data.iloc[i+1:, :]
        B = subdata.loc[subdata.label != data.loc[i, 'label'], features].values
        result = abs((np.tile(A, (len(B), 1)) - B)/B).mean(axis=1)
        df = pd.concat([df, pd.DataFrame(result.reshape(-1, 1))])
    
    df.to_csv(filename)
    return df

vitroTrain = vitro_vivo[(vitro_vivo.usage == 'train') & (vitro_vivo.EXP_TEST_TYPE == 'in vitro')]
vitroTest = vitro_vivo[(vitro_vivo.usage == 'test') & (vitro_vivo.EXP_TEST_TYPE == 'in vitro')]
vivoTrain = vitro_vivo[(vitro_vivo.usage == 'train') & (vitro_vivo.EXP_TEST_TYPE == 'in vivo')]
vivoTest = vitro_vivo[(vitro_vivo.usage == 'test') & (vitro_vivo.EXP_TEST_TYPE == 'in vivo')]
   
#cosine
path = '/path/to/performance/cosine/control/'
control_oneExp_cosine(vitroTrain, path+'vitroTrainCosine.csv')
control_oneExp_cosine(vitroTest, path+'vitroTestCosine.csv')
control_oneExp_cosine(vivoTrain, path+'vivoTrainCosine.csv')
control_oneExp_cosine(vivoTest, path+'vivoTestCosine.csv')

#rmse
path = '/path/to/performance/rmse/control/'
control_oneExp_rmse(vitroTrain, path+'vitroTrainrmse.csv')
control_oneExp_rmse(vitroTest, path+'vitroTestrmse.csv')
control_oneExp_rmse(vivoTrain, path+'vivoTrainrmse.csv')
control_oneExp_rmse(vivoTest, path+'vivoTestrmse.csv')

#mape
path = '/path/to/performance/mape/control/'
control_oneExp_mape(vitroTrain , path+'vitroTrainmape.csv')
control_oneExp_mape(vitroTest, path+'vitroTestmape.csv')
control_oneExp_mape(vivoTrain, path+'vivoTrainmape.csv')
control_oneExp_mape(vivoTest, path+'vivoTestmape.csv')

