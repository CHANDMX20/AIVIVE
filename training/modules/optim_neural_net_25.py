import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam, SGD, RMSprop
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
from keras.layers import Dense, Attention, Reshape, Flatten, LayerNormalization
from tensorflow.keras.regularizers import L2, L1
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from sklearn.preprocessing import LabelBinarizer
from random import randint
import random
from sklearn.preprocessing import MinMaxScaler

#read the data 

train_input = pd.read_csv('path/to/generator1_encoded_prediction_9962160_VivoGenerator.csv', low_memory = False).iloc[:, 1:]
train_output = pd.read_csv('path/to/vivo_train.csv', low_memory = False).iloc[:, 1:]
test_input = pd.read_csv('path/to/generator1_encoded_prediction_9962160_vivoGenerator_test.csv', low_memory=False).iloc[:, 1:]
#test_output = pd.read_csv('path/to/vivo_test.csv', low_memory=False).iloc[:, 1:]

# get the gene feature columns
cols = pd.read_csv('path/to/module25_genes.csv').PROBEID.unique()
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


input_shape = X_train.shape[1]
output_shape = y_train.shape[1]
label_shape = (14)


def define_optim_net(input_shape, label_shape, output_shape):
    # Define the input layers
    input_exp = Input(shape=(input_shape,))
    input_target = Input(shape=(label_shape,))

    # Concatenate the inputs
    concatenated = Concatenate(axis=1)([input_exp, input_target])

# Apply Attention Layer
    reshaped = Reshape((1, -1))(concatenated) 
    attention_output = Attention(use_scale=False)([reshaped, reshaped])  # Self-attention

    # Flatten back the attention output to feed into Dense layers
    attention_output = Reshape((-1,))(attention_output)

 # Second Dense Layer with 512 nodes
    x = Dense(512, activation='relu', kernel_regularizer=L2(0.01))(attention_output)
    x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)

# Second Dense Layer with 512 nodes
    x = Dense(256, activation='relu', kernel_regularizer=L2(0.01))(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=L2(0.01))(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)

    # Third Dense Layer with 256 nodes
    x = Dense(64, activation='relu', kernel_regularizer=L2(0.01))(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)   
    
    # Output layer
    output = Dense(output_shape)(x)  # Ensure activation matches the task

    # Create the model
    model = Model(inputs=[input_exp, input_target], outputs=output)

    # Compile the model with SGD optimizer
    opt = SGD(learning_rate = 0.0001, momentum = 0.9)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    model.summary()
    return model
    
model = define_optim_net(input_shape, label_shape, output_shape)

# fit the keras model on the dataset
history = model.fit([X_train, target_labels], y_train, epochs=100, batch_size=256)

#make preds on the test set
predictions = model.predict([X_train, target_labels])

def save_final_model(model, predsPath, step):
    # Create a filename based on the current step/epoch
    filename = predsPath + f'/model25_final_epoch_{step:06d}.h5'
    
    # Save the model
    model.save(filename)
    print(f'> Model saved to: {filename}')


def save_preds(preds, X_train, train_input, output_filename):
 
    # Step 2: Convert preds to a DataFrame
    preds_df = pd.DataFrame(preds)  # Adjust column name as needed
    preds_df.columns = X_train.columns.tolist()  # Set column names to match additional_data

    # Step 3: Extract the first 23 columns from X_test
    merge_cols = train_input.iloc[:, :23]  # Get the first 23 columns

    # Step 4: Combine the first 23 columns with preds_df
    combined_df = pd.concat([merge_cols.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)

    # Step 5: Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_filename, index=False)

    return combined_df
 
#resultPath 
path = 'path/to/results'
predsPath = path+'/predictions_decoded/train/module25'

#generated predicitons csv: vivo
number = '9962160'
vivo_opt_gen_train = save_preds(predictions, X_train, train_input, predsPath+ '/opt_gen_train' + number + '_Vivo.csv')

save_final_model(model, predsPath, step=100)  # Use the final epoch as the step






