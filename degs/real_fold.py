import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats


# get the gene feature columns
prob2gene = pd.read_csv('path/to/final_rat_genes.csv')
cols = prob2gene.PROBEID.unique()
# true values
#vitro_data= pd.read_csv('path/to/vitro_train_test.csv', low_memory=False)
vivo_data = pd.read_csv(''path/to/vivo_train_test.csv', low_memory=False) #path to real vivo train+test data

def foldchangeMap(filename, org, cols=cols):
    resultDf = pd.DataFrame(columns = ['name', *cols])
    for cmpd in org.COMPOUND_NAME.unique():
        for exp in org.EXP_TEST_TYPE.unique():
            for time in org.SACRIFICE_PERIOD.unique():
                control = org[(org['COMPOUND_NAME'] == cmpd) & (org['EXP_TEST_TYPE'] == exp) & (org['SACRIFICE_PERIOD'] == time) & (org['DOSE_LEVEL'] == 'Control')].loc[:, cols].mean().values
                for dose in ['Low', 'Middle', 'High']:
                    treatment = org[(org['COMPOUND_NAME'] == cmpd) & (org['EXP_TEST_TYPE'] == exp) & (org['SACRIFICE_PERIOD'] == time) & (org['DOSE_LEVEL'] == dose)].loc[:, cols].mean().values
                    if not np.isnan(treatment).any():
                        result = [cmpd+'_'+exp+'_'+time+'_'+dose]
                        realFold = treatment - control
                        result.extend(realFold)
                        resultDf.loc[len(resultDf.index)] = result
    resultDf.to_csv(filename)
    return resultDf

#resultPath 
path = '/path/to/results'

### real
foldchangePath = path+'/performance/DEG/foldchangeValues'

#vitro_realFoldChange = foldchangeMap(foldchangePath+'/vitro_foldchanged_real.csv', vitro_data)
vivo_realFoldChange = foldchangeMap(foldchangePath+'/vivo_foldchanged_real.csv', vivo_data)



