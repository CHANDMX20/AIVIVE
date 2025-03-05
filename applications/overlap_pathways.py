
# import the packages
import pandas as pd
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import ast
import itertools
import re

# load the dataset
real = pd.read_csv('/path/to/real_pathways_r_0.05.csv')  #path to real profiles KEGG pathways
gen = pd.read_csv('/path/to/post_pathways_r_0.05.csv')    #path to generated profiles KEGG pathways

# function to find the overlap


def overlap_pathways(filename, gen_union, real_pathways):
    overlap = {}

    # Define a function to properly split pathways
    def split_pathways(pathways_str):
        # This regex splits by ', ' if:
        # 1. The comma is followed by a space and an uppercase letter.
        # 2. The comma is followed by a space and a lowercase letter immediately followed by a number (e.g., 'p53').
        return re.split(r', (?=[A-Z]|[a-z]\d)', pathways_str)

    # Iterate through the unique treatment names
    for name in gen_union['name'].unique():
        result = []
        
        # Get union pathways for each treatment
        gen_pathways = gen_union[gen_union['name'] == name]['kegg_descriptions'].values[0]
        
        # Ensure gen_pathways is iterable (i.e., a list or set), even for NaN or invalid data
        if isinstance(gen_pathways, str):
            gen_pathways = split_pathways(gen_pathways)  # Use the custom split function
        else:
            gen_pathways = []  # If the value is not a string (e.g., NaN), set an empty list
        
        # Get the real pathways, and ensure it is a valid string
        real_pathways_str = real_pathways[real_pathways['name'] == name]['kegg_descriptions'].values[0]
        
        if isinstance(real_pathways_str, str):  # Only split if the value is a string
            real_path = set(split_pathways(real_pathways_str))  # Use the custom split function
        else:
            real_path = set()  # If the value is not a string (e.g., NaN), set an empty set
        
        # Compute overlap for the union DEGs
        overlap_set = set(gen_pathways).intersection(real_path)
        
        # Append the metrics for the treatment
        result.append(len(real_path))  # Length of real pathways
        result.append(len(gen_pathways))  # Length of union pathways
        result.append(len(overlap_set))  # Length of overlap
        result.append(len(overlap_set) / len(real_path) if len(real_path) > 0 else 0)  # Proportion of overlap
        
        overlap[name] = result

    # Create DataFrame for the overlap results
    overlap_df = pd.DataFrame(overlap.items(), columns=['name', 'Union_Metrics'])
    overlap_df[['Real_Pathways_Count', 'Union_Pathways_Count', 'Overlap_Count', 'Overlap_Proportion']] = pd.DataFrame(overlap_df['Union_Metrics'].tolist(), index=overlap_df.index)
    overlap_df = overlap_df.drop(columns=['Union_Metrics'])
    
    # Save the combined DataFrame to CSV if needed
    overlap_df.to_csv(filename, index=False)
    
    return overlap_df

# Define the result path
path = '/path/to/results/'
degPath = path + '/performance/pathways/r_pathways'

# Calculate overlaps and save results
vivo_overlap = overlap_pathways(degPath + '/post_overlap_r_0.05.csv', gen, real)
