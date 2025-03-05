import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats


#vitro_org = pd.read_csv('path/to/vitro_train_test.csv', low_memory=False)
vivo_org = pd.read_csv('path/to/vivo_train_test.csv', low_memory=False)
prob2gene = pd.read_csv('path/to/final_rat_genes.csv')
cols = prob2gene.PROBEID.unique()

def foldchangeMap(generated, filename, org, cols=cols):
    resultDf = pd.DataFrame(columns = ['name', *cols])
    for cmpd in generated.COMPOUND_NAME.unique():
        for exp in generated.targetExp.unique():
            for time in generated.targetTime.unique():
                control = org[(org['COMPOUND_NAME'] == cmpd) & (org['EXP_TEST_TYPE'] == exp) & (org['SACRIFICE_PERIOD'] == time) & (org['DOSE_LEVEL'] == 'Control')].loc[:, cols].mean().values
                for dose in ['Low', 'Middle', 'High']:
                    dose_data = generated[(generated['COMPOUND_NAME'] == cmpd) & (generated['targetExp'] == exp) & (generated['targetTime'] == time) & (generated['targetDose'] == dose)]
                        # Compute mean values for each barcode
                    subgroups = dose_data.groupby('BARCODE').agg({col: 'mean' for col in cols}).reset_index()
                        # Process each barcode group
                    for _, row in subgroups.iterrows():
                        barcode = row['BARCODE']
                        predictedValues = row[cols].values

                    # Compute generated fold (difference from control)
                        generatedFold = predictedValues - control

                    # Format result and append to resultDf
                        result = [f"{cmpd}_{exp}_{time}_{dose}*{barcode}"]
                        result.extend(generatedFold)
                        resultDf.loc[len(resultDf)] = result    
                    
    resultDf.to_csv(filename)
    return resultDf

#resultPath 
path = 'path/to/results'

### test
dataPath = path+'/predictions_decoded/test' #path to the generated test predictions 
degPath = path+'/performance/DEG/foldchangeValues' #path to the foldchange results
predsPath = path+'/predictions_decoded/test_set/' #path to the combined generated profile post local optimization 


#generatedData: vivo (before optimization)
number = '9962160'
testVivo = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_vivoGenerator_test.csv', low_memory=False)
testVivofoldchange = foldchangeMap(testVivo, degPath+'/foldchangeValues' + number + '_VivoGenerator_test.csv', vivo_org)


#generatedData: combined optimized vivo
number = '9962160'
testOptVivo = pd.read_csv(predsPath+ '/combined_opt_gan_test' + number + '_Vivo.csv', low_memory=False)
testOptVivofold = foldchangeMap(testOptVivo, degPath+ '/foldchangeValues' + number + '_OptVivo.csv', vivo_org)



#generatedData: vivo
#number = '9962160'
#trainVivo = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VivoGenerator.csv', low_memory=False)
#trainVivofoldchange = foldchangeMap(trainVivo, degPath+'/foldchangeValues' + number + '_VivoGenerator_train.csv', vivo_org)

#generatedData: vitro
number = '9962160'
#testVitro = pd.read_csv(dataPath + '/generator1_encoded_prediction_' + number + '_VitroGenerator.csv', low_memory=False)
#testVitrofoldchange = foldchangeMap(testVitro, degPath+'/foldchangeValues' + number + '_VitroGenerator.csv', vitro_org)


