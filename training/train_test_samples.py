# import the necessary packages
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import GaussianNoise
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from random import randint
import random

# path
dataPath = '/account/mansi.chandra/vitro_vivo'
resultPath = '/account/mansi.chandra/vitro_vivo/model_2/results/predictions_decoded/test'
# load the vivo generator 
g_vivo = keras.models.load_model('path/to/results/model2/g_model2_9962160.h5')
# load the vitro generator 
#g_vitro = keras.models.load_model('/account/mansi.chandra/vitro_vivo/model_2/results/model1/g_model1_9962160.h5')


def read_data(path):
    data = pd.read_csv(path)
    data = data.iloc[:, 1:]
    return data

def binarizer(data):
    expBinarizer = LabelBinarizer().fit(data["EXP_TEST_TYPE"])
    doseBinarizer = LabelBinarizer().fit(data["DOSE_LEVEL"])
    stageBinarizer = LabelBinarizer().fit(data["SACRIFICE_PERIOD"])
    bioCopyBinarizer = LabelBinarizer().fit(data["INDIVIDUAL_ID"])
    return expBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer

# read data set
vitroTrain = read_data(dataPath + '/vitro_train.csv')
vitroTest = read_data(dataPath + '/vitro_test.csv')
vivoTrain = read_data(dataPath + '/vivo_train.csv')
vivoTest = read_data(dataPath + '/vivo_test.csv')


# concatenate data set
dataset = pd.concat([vitroTrain , vitroTest , vivoTrain , vivoTest], axis = 0)
dataset = dataset.sort_values('EXP_ID', ascending = True)

# initialize binarizer to transfer category data (EXP_TEST_TYPE, SACRIFICE_PERIOD, DOSE_LEVEL) to binary data
expBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)

# get the gene feature columns
cols = pd.read_csv('path/to/final_rat_genes.csv').PROBEID.unique()
print(len(cols))


# scale the dataset 
def scale(df1, df2, cols):
    X1 = df1[cols]
    X2 = df2[cols]
    # scale data
    scaler = MinMaxScaler()
    scaler.fit(X1)
    X2 = scaler.transform(X2)
    scaledf = pd.DataFrame(X2, columns = cols)
    resultdf = pd.concat([df2.iloc[:, :-31099], scaledf], axis = 1)
    return resultdf, scaler

vitroTest, _  = scale(vitroTrain, vitroTest, cols=cols)
vitroTrain, vitroScaler = scale(vitroTrain, vitroTrain, cols=cols)
vivoTest, _  = scale(vivoTrain, vivoTest, cols=cols)
vivoTrain, vivoScaler = scale(vivoTrain, vivoTrain, cols=cols)



# select a batch of random samples, returns Xs and target
def generate_test_real_samples(vitroTest, vivoTest, cols=cols, expBinarizer=expBinarizer, doseBinarizer=doseBinarizer, stageBinarizer=stageBinarizer, bioCopyBinarizer=bioCopyBinarizer):
    dataset = pd.concat([vitroTest, vivoTest])
    dataset = dataset.sort_values(['COMPOUND_NAME', 'DOSE_LEVEL'])
    dataset = dataset.reset_index(drop=True)
    df = pd.DataFrame()
    
    def pairs(subFrom):
        tmpDf = pd.DataFrame()
        tmpDf = tmpDf.append([subFrom]*len(subFrom), ignore_index=True)
        tmpDf = tmpDf.reset_index(drop=True)
        subTo = tmpDf[['BARCODE', 'EXP_TEST_TYPE', 'SACRIFICE_PERIOD', 'DOSE_LEVEL', 'INDIVIDUAL_ID']]
        subTo = subTo.sort_values('BARCODE')
        subTo.columns = ['targetId', 'targetExp', 'targetTime', 'targetDose', 'targetBioCopy']
        subTo = subTo.reset_index(drop = True)
        tmpDf = pd.concat([tmpDf, subTo], axis = 1)
        return tmpDf     
    
    for compound in dataset.COMPOUND_NAME.unique():
        subFrom = dataset[(dataset.COMPOUND_NAME == compound) & (dataset.DOSE_LEVEL == 'Control')]
        tmpDf = pairs(subFrom)
        df = df.append(tmpDf, ignore_index=True)
        subFrom = dataset[(dataset.COMPOUND_NAME == compound) & (dataset.DOSE_LEVEL != 'Control')]
        tmpDf = pairs(subFrom)
        df = df.append(tmpDf, ignore_index=True)
        
    def binaryRepresentation(data):
        data = data.reset_index(drop=True)
        # reform sample label in source
        exp = expBinarizer.transform(data['EXP_TEST_TYPE'])
        dose = doseBinarizer.transform(data['DOSE_LEVEL'])
        time = stageBinarizer.transform(data['SACRIFICE_PERIOD'])
        bioCopy = bioCopyBinarizer.transform(data['INDIVIDUAL_ID'])

        # reform sample label in target
        targetExp = expBinarizer.transform(data['targetExp'])
        targetDose = doseBinarizer.transform(data['targetDose'])
        targetTime = stageBinarizer.transform(data['targetTime'])
        targetBioCopy = bioCopyBinarizer.transform(data['targetBioCopy'])
        
        input_noise = np.random.normal(0, 0.05, [len(np.array(data[cols])),len(np.array(data[cols])[0])])
        
        return data.drop([*cols], axis=1), np.array(data[cols]), np.hstack([exp, dose, time, bioCopy]), np.hstack([targetExp, targetDose, targetTime, targetBioCopy]), input_noise
        
        
    # drop the targetExp and Exp is the same pairs
    toVivo = df[(df.EXP_TEST_TYPE != df.targetExp) & (df.EXP_TEST_TYPE == 'in vitro')]
    toVitro = df[(df.EXP_TEST_TYPE != df.targetExp) & (df.EXP_TEST_TYPE == 'in vivo')]
    
    # get the input for generator in vitro
    masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise = binaryRepresentation(toVitro)
    # get the input for generator in vivo
    masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise = binaryRepresentation(toVivo)

    return masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise


def summarize_performance(step, g_model, input_gene, input_label, input_target, input_noise, expDf, name, expScaler, features=cols, resultPath=resultPath):
    # make prediction
    X_out = g_model.predict([input_gene, input_label, input_target, input_noise])
    X_out = expScaler.inverse_transform(X_out)

    # save prediction
    X_out_df = pd.DataFrame(data=X_out, columns=features)
    X_out_df = pd.concat([expDf, X_out_df], axis = 1)
    filename = resultPath + '/generator1_encoded_prediction_%06d_%s.csv' %(step, name)
    X_out_df.to_csv(filename)
    

    
### for train set
masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise = generate_test_real_samples(vitroTrain, vivoTrain)
# make vivo predictions
summarize_performance(9962160, g_vivo, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise, masterVitro, 'VivoGenerator', vivoScaler)
# make vitro predictions
#summarize_performance(9962160, g_vitro, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVivo, 'VitroGenerator', vitroScaler)


### for test set
masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise = generate_test_real_samples(vitroTest, vivoTest)
# make vivo predictions
summarize_performance(9962160, g_vivo, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise, masterVitro, 'vivoGenerator_test', vivoScaler)
# make vitro predictions
#summarize_performance(9962160, g_vitro, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVivo, 'VitroGenerator_test', vitroScaler)
