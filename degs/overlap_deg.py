
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import ast


prob2gene = pd.read_csv('/path/to/final_rat_genes.csv')
cols = prob2gene.PROBEID.unique()
vivo_real_deg = pd.read_csv('/path/to/real/vivo_real_deg.csv').iloc[:, 1:]   #path to real DEGs
vivo_union_deg = pd.read_csv('path/to/vivo_union_deg_0.58.csv').iloc[:, 1:]  #path to generated DEGs from gen_union.py


def overlap_degs(filename, gendeg_union, realdeg):
    overlap_union = {}
    #overlap_intersect = {}


    # Function to convert the string to a set
    def extract_set(s):
        try:
        # Safely evaluate the string to a Python object
            result = ast.literal_eval(s)
            if isinstance(result, list) and len(result) == 1:
                return result[0]  # Extract the set from the list
            return set()  # Return an empty set if the format is not as expected
        except (ValueError, SyntaxError):
        # Handle cases where the string is not valid
            return set()

# Apply the function to the DataFrame column
    realdeg.iloc[:, 1] = realdeg.iloc[:, 1].apply(extract_set)
    
    for name in gendeg_union.name.unique():
        result_union = []
        
        # Get union DEGs and real DEGs
        gen_union = gendeg_union[gendeg_union.name == name].values[0][1:][0]
        if isinstance(gen_union, str):
            gen_union = eval(gen_union)
        real = realdeg[realdeg.name == name].values[0][1:][0]
        
        
        # Compute overlaps for union DEGs
        overlaps_union = gen_union.intersection(real)
        result_union.append(len(real))  # Length of real DEGs
        result_union.append(len(gen_union))  # Length of union DEGs
        result_union.append(len(overlaps_union))  # Length of overlap
        result_union.append(len(overlaps_union) / len(real) if len(real) > 0 else 0)  # Proportion of overlap
        overlap_union[name] = result_union

    # Create DataFrame for union results
    overlap_union_df = pd.DataFrame(overlap_union.items(), columns=['name', 'Union_Metrics'])
    overlap_union_df[['Real_DEG_Count', 'Union_DEG_Count', 'Overlap_Count', 'Overlap_Proportion']] = pd.DataFrame(overlap_union_df['Union_Metrics'].tolist(), index=overlap_union_df.index)
    overlap_union_df = overlap_union_df.drop(columns=['Union_Metrics'])
    
    # Save the combined DataFrame to CSV
    overlap_union_df.to_csv(filename, index=False)
    
    return overlap_union_df

# Define the result path
path = '/path/to/results'         #path to store results
degPath = path + '/performance/DEG/overlap_degs'

# Calculate overlaps and save results
vivo_overlap = overlap_degs(degPath + '/vivo_overlap_0.58.csv', vivo_union_deg, vivo_real_deg)

#vitro_overlap = overlap_degs(degPath + '/vitro_overlap_0.58_later2.csv', vitro_union_deg, vitro_real_deg)


