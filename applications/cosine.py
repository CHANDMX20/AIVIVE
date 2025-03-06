import sys
#number=sys.argv[1]
#path=sys.argv[2]


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# get the gene feature columns
features = pd.read_csv('/path/to/final_rat_genes.csv').PROBEID.unique()
print(len(features))
# true values
vitro_vivo= pd.read_csv('/path/to/vitro_vivo_train_test.csv', low_memory=False)


def generated_cosine(predicted, filename, features=features, org=vitro_vivo):
    result = predicted.iloc[:, :-len(features)]
    values = []
    for i in range(len(result)):        
        # true values
        targetSample = result.loc[i, 'targetId']
        true_value = org[org['BARCODE'] == targetSample].loc[:, features].values[0]
        # predicted values
        predicted_value = predicted.loc[i, features].values
        cos=cosine_similarity(true_value.reshape(1,-1), predicted_value.reshape(1,-1))[0][0]
        values.append(cos)
    result['cosine'] = values
    result.to_csv(filename)

    
# resultPath
path = '/path/to/results'
# final model number
number='9962160'

### generated test predictions 
dataPath = path + '/predictions_decoded/test_set'
cosinePath = path + '/performance/cosine/test'      #path to store the results
# vivo
testVivo = pd.read_csv(dataPath + '/combined_opt_gan_test' + number + '_Vivo.csv')
generated_cosine(testVivo, cosinePath+'/' + number + '_OptVivo.csv')
# vitro
#testVitro = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VitroGenerator.csv')
#generated_cosine(testVitro, cosinePath+'/' + number + '_VitroGenerator.csv')


### train
#dataPath = path + '/predictions_decoded/train'
#cosinePath = path + '/performance/cosine/train'
# vivo
#trainVivo = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VivoGenerator.csv')
#generated_cosine(trainVivo, cosinePath+'/' + number + '_VivoGenerator.csv')
# vitro
#trainVitro = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VitroGenerator.csv')
#generated_cosine(trainVitro, cosinePath+'/' + number + '_VitroGenerator.csv')


