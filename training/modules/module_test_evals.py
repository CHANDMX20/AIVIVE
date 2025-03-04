import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
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

# path
dataPath = '/account/mansi.chandra/vitro_vivo'
resultPath = 'path/to/test_set/module115'
# load the vivo generator 
g_vivo = keras.models.load_model('path/to/predictions_decoded/train/module115/model115_final_epoch_000100.h5') #path to the local optimizer model saved 

#read the data 

train_input = pd.read_csv('path/to/generator1_encoded_prediction_9962160_VivoGenerator.csv', low_memory = False).iloc[:, 1:]
train_output = pd.read_csv('path/to/vivo_train.csv', low_memory = False).iloc[:, 1:]
test_input = pd.read_csv('path/to/generator1_encoded_prediction_9962160_vivoGenerator_test.csv', low_memory=False).iloc[:, 1:]
#test_output = pd.read_csv('/account/mansi.chandra/vitro_vivo/vivo_test.csv', low_memory=False).iloc[:, 1:]

# get the gene feature columns
cols = pd.read_csv('path/to/module115/module115_genes.csv').PROBEID.unique()
print(len(cols))                   


def mapping_barcodes(input_data, output_data):

    # Step 2: Create a Mapping of targetIDs from input_data
    mapping_counts = input_data['targetId'].value_counts()
    
    # Step 3: Prepare a list to hold the results
    result_rows = []

    # Iterate over output_data to generate the results
    for index, row in output_data.iterrows():
        barcode = row['BARCODE']
        
        # Append the original row
        result_rows.append(row)  # Keep the original row
        
        # Get the count of how many times this barcode matches a targetID
        count = mapping_counts.get(barcode, 0)  # Default to 0 if no match
        
        # Add duplicates based on the count minus one (since we already have one original)
        if count > 1:
            result_rows.extend([row] * (count - 1))

    # Convert the list of rows to a DataFrame
    output_df = pd.DataFrame(result_rows)
    
    return output_df

train_df = mapping_barcodes(train_input, train_output)
#test_df = mapping_barcodes(test_input, test_output)

def binarizer(data):
    expBinarizer = LabelBinarizer().fit(data["targetExp"])
    doseBinarizer = LabelBinarizer().fit(data["targetDose"])
    stageBinarizer = LabelBinarizer().fit(data["targetTime"])
    bioCopyBinarizer = LabelBinarizer().fit(data["targetBioCopy"])
    return expBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer

# initialize binarizer to transfer category data (EXP_TEST_TYPE, SACRIFICE_PERIOD, DOSE_LEVEL) to binary data
expBinarizer, doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(train_input)

targetExp = expBinarizer.transform(train_input['targetExp'])
targetDose = doseBinarizer.transform(train_input['targetDose'])
targetTime = stageBinarizer.transform(train_input['targetTime'])
targetBioCopy = bioCopyBinarizer.transform(train_input['targetBioCopy'])

# Concatenate the binary target arrays
target_labels = np.hstack([targetExp, targetDose, targetTime, targetBioCopy])


X_train = train_input[cols].copy()
y_train = train_df[cols].copy()
X_test = test_input[cols].copy()
#y_test = test_df[cols].copy()

predictions = g_vivo.predict([X_test, target_labels])

def save_preds(preds, X_test, test_input, output_filename):
 
    # Step 2: Convert preds to a DataFrame
    preds_df = pd.DataFrame(preds)  # Adjust column name as needed
    preds_df.columns = X_test.columns.tolist()  # Set column names to match additional_data

    # Step 3: Extract the first 23 columns from X_test
    merge_cols = test_input.iloc[:, :23]  # Get the first 23 columns

    # Step 4: Combine the first 23 columns with preds_df
    combined_df = pd.concat([merge_cols.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)

    # Step 5: Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_filename, index=False)

    return combined_df
 
#generated predicitons csv: vivo
number = '9962160'
vivo_opt_gen_test = save_preds(predictions, X_test, test_input, resultPath+ '/opt_gen_test' + number + '_Vivo.csv')




