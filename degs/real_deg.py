
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

# get the gene feature columns
prob2gene = pd.read_csv('path/to/final_rat_genes.csv')
cols = prob2gene.PROBEID.unique()
#vitro_real_foldchange = pd.read_csv('/account/mansi.chandra/vitro_vivo/model_2/results/performance/DEG/foldchangeValues/vitro_foldchanged_real.csv').iloc[:, 1:]
vivo_real_foldchange = pd.read_csv('/account/mansi.chandra/vitro_vivo/model_2/results/performance/DEG/foldchangeValues/vivo_foldchanged_real.csv').iloc[:, 1:]

def realdeg(filename, realfoldchange, prob2gene=prob2gene):
    resultDic = {}
    for name in realfoldchange.name.unique():
        result = []
        realFold = realfoldchange[realfoldchange.name == name].values[0][1:]
        realDiffCols = set(realfoldchange.columns[1:][np.where(abs(realFold) >= 0.58, True, False)])
        realDEG = set(prob2gene[prob2gene.PROBEID.isin(realDiffCols)].SYMBOL.unique())
        result.append(realDEG)
 
        resultDic[name] = result

    resultDicDf = pd.DataFrame(resultDic.items())
    resultDicDf = resultDicDf.rename(columns={0:'name', 1:'DEGs_0.58'})
    resultDicDf['DEG_Count'] = resultDicDf['DEGs_0.58'].apply(lambda x: len(x[0]) if isinstance(x, list) and x else 0)
    resultDicDf.to_csv(filename) 
    return resultDicDf

#resultPath 
path = '/path/to/results'

### test
degPath = path+'/performance/DEG/real'

#vitro_real_deg = realdeg(degPath+ '/vitro_real_deg.csv', vitro_real_foldchange)

vivo_real_deg = realdeg(degPath+ '/vivo_real_deg.csv', vivo_real_foldchange)




