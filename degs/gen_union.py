
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats


prob2gene = pd.read_csv('path/to/final_rat_genes.csv')
cols = prob2gene.PROBEID.unique()
vivo_gen_foldchange = pd.read_csv('path/to/foldchangeValues/foldchangeValues9962160_OptVivo.csv').iloc[:, 1:] #path to generated profile's foldchange values 

def union_deg(filename, generatedfoldchange, prob2gene=prob2gene):
    generatedfoldchange['name'] = generatedfoldchange['name'].str.replace('.0', '', regex=False)
    resultDic = {}
    for name in generatedfoldchange.name.unique():
        result = []
        genFold = generatedfoldchange[generatedfoldchange.name == name].values[0][1:]
        genDiffCols = set(generatedfoldchange.columns[1:][np.where(abs(genFold) >= 0.58, True, False)])
        genDEG = set(prob2gene[prob2gene.PROBEID.isin(genDiffCols)].SYMBOL.unique())
        result.append(genDEG)    
        resultDic[name] = result

    resultDicDf = pd.DataFrame(resultDic.items())
    resultDicDf.iloc[:, 0] = resultDicDf.iloc[:, 0].str.split('*').str[0]
    resultDicDf.iloc[:, 1] = resultDicDf.iloc[:, 1].apply(lambda x: x[0] if x else set())
    
    def union_of_gene_sets(sets_list):
        union_set = set()
        for gene_set in sets_list:
            union_set.update(gene_set)
        return union_set

# Group by treatment and apply the union function
    union_df = resultDicDf.groupby(0)[1].apply(lambda x: union_of_gene_sets(x))

# Calculate the length of unique genes in each treatment
    def calculate_length_of_genes(gene_set):
        return len(gene_set)

# Apply the length calculation
    length_result = union_df.apply(calculate_length_of_genes)

# Convert the results to a DataFrame
    union_df = pd.DataFrame({
        'name': union_df.index,
        'DEGs_0.58': union_df.values,
        'Gene_Count': length_result.values
    })
    union_df.to_csv(filename)
    return union_df

#resultPath 
path = '/path/to/results'

### test
degPath = path+'/overlap_degs'

vivo_union_deg = union_deg(degPath+ '/vivo_union_deg_0.58.csv', vivo_gen_foldchange)

#vitro_union_deg = union_deg(degPath+ '/vitro_union_deg_0.58_later2.csv', vitro_gen_foldchange, vitro_gen_pval)

