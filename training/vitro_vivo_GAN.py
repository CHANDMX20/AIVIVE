#import the packages
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from random import randint
import random

# Function to read csv file and return dataframe
def read_data(path):
    data = pd.read_csv(path)
    data = data.iloc[:, 1:]
    return data

# Binarize categorical columns 
def binarizer(data):
    expBinarizer = LabelBinarizer().fit(data["EXP_TEST_TYPE"])
    doseBinarizer = LabelBinarizer().fit(data["DOSE_LEVEL"])
    stageBinarizer = LabelBinarizer().fit(data["SACRIFICE_PERIOD"])
    bioCopyBinarizer = LabelBinarizer().fit(data["INDIVIDUAL_ID"])
    return expBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer

# Load the data
dataPath = '/path/to/vitro_vivo_data'
vitroTrain = read_data(dataPath + '/vitro_train.csv')
vitroTest = read_data(dataPath + '/vitro_test.csv')
vivoTrain = read_data(dataPath + '/vivo_train.csv')
vivoTest = read_data(dataPath + '/vivo_test.csv')

# concatenate the data
dataset = pd.concat([vitroTrain , vitroTest , vivoTrain , vivoTest], axis = 0)
dataset = dataset.sort_values('EXP_ID', ascending = True)

# initialize binarizer to transfer category data (EXP_TEST_TYPE, SACRIFICE_PERIOD, DOSE_LEVEL) to binary data
expBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)

# get the gene feature columns
cols = pd.read_csv('path/to/final_rat_genes.csv').PROBEID.unique()
print(len(cols))

# scale the dataset columns (MinMax Scaler used)
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


# define the discriminator model
def define_discriminator(input_dim):
    # weight initialization
    init = RandomNormal(stddev=0.01)       

    # model structure
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.0001, momentum=0.9)    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the generator model
def define_generator(input_shape, label_shape, output_shape):
    # gene input
    input_gene = Input(shape=input_shape)
    input_label = Input(shape=label_shape)
    input_target = Input(shape=label_shape)
    input_noise = Input(shape=input_shape)

    # weight initialization
    init = RandomNormal(stddev=0.01)
    
    # l1
    g = Dense(8192, kernel_initializer=init)(Concatenate(axis=1)([input_gene, input_label, input_target, input_noise]))
    g = Dropout(0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)

    # l2
    c = Concatenate()([g, input_target])
    g = Dense(7168, kernel_initializer=init)(c)
    g = Dropout(0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)

    
    # l3
    c = Concatenate()([g, input_target])
    g = Dense(7168, kernel_initializer=init)(c)#activation="relu",
    g = Dropout(0.8)(g)
    g = LeakyReLU(alpha=0.2)(g)

    # l4
    g = Dense(4096, kernel_initializer=init)(g)
    g = Dropout(0.4)(g)
    g = LeakyReLU(alpha=0.2)(g)

    # l5
    g = Dense(4096, kernel_initializer=init)(g)#, activation="relu"
    g = LeakyReLU(alpha=0.2)(g)


    # output
    g = Dense(output_shape, activation='sigmoid', kernel_initializer=init)(g)

    # define model
    model = Model([input_gene, input_label, input_target, input_noise], g)
    return model



# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, input_shape, label_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gene = Input(shape=input_shape)
    input_label = Input(shape=label_shape)
    input_target = Input(shape=label_shape)
    input_noise = Input(shape=input_shape)
    gen1_out = g_model_1([input_gene, input_label, input_target, input_noise])
    output_d = d_model(gen1_out)
    # forward cycle
    output_f = g_model_2([gen1_out, input_target, input_label, input_noise])
    # define model graph
    model = Model([input_gene, input_label, input_target, input_noise], [gen1_out, output_d, output_f])
    # define optimization algorithm configuration
    opt = keras.optimizers.Adam(learning_rate=0.00001)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mae', 'binary_crossentropy', 'mae'], loss_weights=[10, 2, 5], optimizer=opt)
    return model

# select a batch of random samples, returns Xs and target; Generate real samples from the dataset
def generate_real_samples(dataset, n_samples, cols=cols, expBinarizer=expBinarizer, doseBinarizer=doseBinarizer, stageBinarizer=stageBinarizer, bioCopyBinarizer=bioCopyBinarizer):
    # Reset the index
    dataset = dataset.reset_index(drop = True)
    
    # choose random instances
    ix = [randint(0, dataset.shape[0] - 1) for i in range(n_samples)]
    # retrieve selected instances
    samples = dataset.iloc[ix, :]
  
    # reform sample label in source
    exp = expBinarizer.transform(samples['EXP_TEST_TYPE'])
    dose = doseBinarizer.transform(samples['DOSE_LEVEL'])
    time = stageBinarizer.transform(samples['SACRIFICE_PERIOD'])
    bioCopy = bioCopyBinarizer.transform(samples['INDIVIDUAL_ID'])
      
    # generate 'real' class labels (1)
    y = np.ones(len(samples))
    return np.array(samples[cols]), y, np.hstack([exp, dose, time, bioCopy]), samples['BARCODE'].values[0]


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
    
    # get the input for in vitro generator 
    masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise = binaryRepresentation(toVitro)
    # get the input for in vivo generator 
    masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise = binaryRepresentation(toVivo)

    return masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise


# generate a batch of fake samples, returns fake samples and targets
def generate_fake_samples(g_model, input_gene, input_label, input_target, input_noise):
    # generate fake instance
    X = g_model.predict([input_gene, input_label, input_target, input_noise])
    # create 'fake' class labels (0)
    y = np.zeros(len(X))
    return X, y


### result Path
resultPath = '/path/to/results/'

# save the generator models to file
def save_models(step, g_model1, g_model2, resultPath=resultPath):
    
    # save the first generator model
    filename1 = resultPath + '/model1/g_model1_%06d.h5' % (step+1)
    g_model1.save(filename1)
    #print('>Saved: %s' %filename1)

    # save the second generator model
    filename2 = resultPath + '/model2/g_model2_%06d.h5' % (step+1)
    g_model2.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


def summarize_performance(step, g_model, input_gene, input_label, input_target, input_noise, expDf, name, expScaler, features=cols, resultPath=resultPath):
    # make prediction
    X_out = g_model.predict([input_gene, input_label, input_target, input_noise])
    X_out = expScaler.inverse_transform(X_out)
    
    # save prediction
    #features = []
    #for i in range(1, len(X_out[0])+1):
    #    features.append('f'+str(i))
    X_out_df = pd.DataFrame(data=X_out, columns=features)
    X_out_df = pd.concat([expDf, X_out_df], axis = 1)
    filename = resultPath + '/predictions_encoded/generator1_encoded_prediction_%06d_%s.csv' %(step+1, name)
    X_out_df.to_csv(filename)  
    
    
def matchingDataset(dataset1, dataset2):
    # select random compound and dose level and get the sub dataset
    compound = random.choice(dataset1.COMPOUND_NAME.unique())
    treat = random.choice(['Control', 'Low', 'Middle', 'High']) 
    if treat == 'Control':
        sub1 = dataset1[(dataset1.COMPOUND_NAME == compound) & (dataset1.DOSE_LEVEL == 'Control')]
        sub2 = dataset2[(dataset2.COMPOUND_NAME == compound) & (dataset2.DOSE_LEVEL == 'Control')]
    else:
        sub1 = dataset1[(dataset1.COMPOUND_NAME == compound) & (dataset1.DOSE_LEVEL != 'Control')]
        sub2 = dataset2[(dataset2.COMPOUND_NAME == compound) & (dataset2.DOSE_LEVEL != 'Control')]   
    return sub1, sub2

# train cyclegan models
def train(d_model1, d_model2, g_model1, g_model2, composite_model1, composite_model2, vitroTrain, vivoTrain, vitroTest, vivoTest, vitroScaler, vivoScaler, resultPath=resultPath):
    # define properties of the training run
    n_epochs, n_batch, = 10000, 1
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(vitroTrain) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # add the target label in the testSet for prediction
    masterVivo, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVitro, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise = generate_test_real_samples(vitroTest, vivoTest)
    # manually enumerate epochs
    loss = []
    for i in range(n_steps):
        # select subset dataset with a random compound and dose level  
        subVitro, subVivo = matchingDataset(vitroTrain, vivoTrain)       
        # select a batch of real samples
        X_realA, y_realA, X_realA_label, X_barcodeA = generate_real_samples(subVitro, n_batch)
        X_realB, y_realB, X_realB_label, X_barcodeB = generate_real_samples(subVivo, n_batch)
        # generate a batch of fake samples
        ### generate noise
        input_noise = np.random.normal(0, 0.05, [len(X_realA),len(X_realA[0])]) 
        X_fakeA, y_fakeA = generate_fake_samples(g_model1, X_realB, X_realB_label, X_realA_label, input_noise)
        X_fakeB, y_fakeB = generate_fake_samples(g_model2, X_realA, X_realA_label, X_realB_label, input_noise)
        # update generator1 via adversarial and cycle loss
        g_loss1, _, _, _ = composite_model1.train_on_batch([X_realB, X_realB_label, X_realA_label, input_noise], [X_realA, y_realA, X_realB])   
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model1.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model1.train_on_batch(X_fakeA, y_fakeA)
        # update generator1 via adversarial and cycle loss
        g_loss2, _, _, _ = composite_model2.train_on_batch([X_realA, X_realA_label, X_realB_label, input_noise], [X_realB, y_realB, X_realA])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model2.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model2.train_on_batch(X_fakeB, y_fakeB)
        loss.append([i, X_barcodeA, X_barcodeB, g_loss1, g_loss2, dA_loss1[0], dA_loss2[0], dB_loss1[0], dB_loss2[0]])
        # summarize performance
        print('>%d, %d, %d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, int(X_barcodeA), int(X_barcodeB), dA_loss1[0], dA_loss2[0], dB_loss1[0], dB_loss2[0], g_loss1, g_loss2))
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 5) == 0:
            # plot A->B translation
            summarize_performance(i, g_model1, X_vivo, X_vivo_Label, X_vivo_target, vivo_input_noise, masterVivo, 'VitroGenerator', vitroScaler)
            summarize_performance(i, g_model2, X_vitro, X_vitro_Label, X_vitro_target, vitro_input_noise, masterVitro, 'VivoGenerator', vivoScaler)             
            # save the models
            save_models(i, g_model1, g_model2)
        if (i+1) % (bat_per_epo * 50) == 0: 
            loss_filename = resultPath + '/loss/loss_%06d.csv' % (i+1)
            pd.DataFrame(loss).to_csv(loss_filename)     
    pd.DataFrame(loss).to_csv(resultPath + '/loss/loss.csv')        


# define input shape 
input_shape = (len(cols), )
# define output shape 
output_shape = len(cols)
# define label shape for source and target
label_shape = (16, )

# generator in vitro
g_model1 = define_generator(input_shape, label_shape, output_shape)
print('g_model1', g_model1.summary())
# generator in vivo
g_model2 = define_generator(input_shape, label_shape, output_shape)
print('g_model2', g_model2.summary())

# discriminator in vitro
d_model1 = define_discriminator(len(cols))
print('d_model1', d_model1.summary())
# discriminator in vivo
d_model2 = define_discriminator(len(cols))
print('d_model2', d_model2.summary())

# composite 1
composite_model1 = define_composite_model(g_model1, d_model1, g_model2, input_shape, label_shape)
# composite 2
composite_model2 = define_composite_model(g_model2, d_model2, g_model1, input_shape, label_shape)

# train models
train(d_model1, d_model2, g_model1, g_model2, composite_model1, composite_model2, vitroTrain, vivoTrain, vitroTest, vivoTest, vitroScaler, vivoScaler)



