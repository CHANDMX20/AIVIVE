#load the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
real_profile = pd.read_csv('/path/to/vivo_test.csv', low_memory=False)   #path to real vivo test data
syn_profile = pd.read_csv('/path/to/combined_opt_gan_test9962160_Vivo.csv', low_memory=False)   #path to the optimized synthetic vivo test profile

# Filter rows based on the conditions
real_profile_df = real_profile[(real_profile['DOSE_LEVEL'] == 'High') &
                                (real_profile['SACRIFICE_PERIOD'] == '24 hr')]

# Filter rows based on the conditions
syn_profile_df = syn_profile[(syn_profile['targetDose'] == 'High') &
                 (syn_profile['targetTime'] == '24 hr')]

# laod the aop genes overlapped with rat S1500+ gene set 
aop_genes = pd.read_csv('path/to/aop_overlap_genes.csv')
aop_probe = aop_genes.PROBEID.unique()

# concatenate the features and aop probe columns
real_profile_df = pd.concat([real_profile_df[['COMPOUND_NAME', 'DOSE_LEVEL', 'SACRIFICE_PERIOD', 'COMPOUND_ABBREVIATION']], real_profile_df[aop_probe]], axis = 1)
syn_profile_df = pd.concat([syn_profile_df[['COMPOUND_NAME', 'targetDose', 'targetTime', 'COMPOUND_ABBREVIATION']], syn_profile_df[aop_probe]], axis = 1)

# rename the probeids as gene symbols 

probe_to_gene_mapping = dict(zip(aop_genes['PROBEID'], aop_genes['SYMBOL']))
real_profile_df = real_profile_df.rename(columns=probe_to_gene_mapping)
syn_profile_df = syn_profile_df.rename(columns=probe_to_gene_mapping)


# average the biological replicates expression values 
# Step 2: Perform groupby and then manually aggregate the means for numeric columns
real_profile_avg = real_profile_df.groupby(['COMPOUND_NAME', 'DOSE_LEVEL', 'SACRIFICE_PERIOD', 'COMPOUND_ABBREVIATION']).mean()

# Step 3: Reset the index to get a proper DataFrame
real_profile_avg = real_profile_avg.reset_index()

# Step 2: Perform groupby and then manually aggregate the means for numeric columns
syn_profile_avg = syn_profile_df.groupby(['COMPOUND_NAME', 'targetDose', 'targetTime', 'COMPOUND_ABBREVIATION']).mean()

# Step 3: Reset the index to get a proper DataFrame
syn_profile_avg = syn_profile_avg.reset_index()


#identify the numerical columns 
real_cols = real_profile_avg.select_dtypes(include='number')
syn_cols = syn_profile_avg.select_dtypes(include='number')


# Group by column names and take the mean of numeric columns (duplicate gennes) 
real_mean = real_cols.groupby(real_cols.columns, axis=1).mean()
# Combine with non-numeric columns
real_non_numeric = real_profile_avg.select_dtypes(exclude='number')
# Concatenate the numeric mean columns with the non-numeric columns
real_avg_profile = pd.concat([real_non_numeric, real_mean], axis=1)


# Group by column names and take the mean of numeric columns
syn_mean = syn_cols.groupby(syn_cols.columns, axis=1).mean()
# Combine with non-numeric columns
syn_non_numeric = syn_profile_avg.select_dtypes(exclude='number')
# Concatenate the numeric mean columns with the non-numeric columns
syn_avg_profile = pd.concat([syn_non_numeric, syn_mean], axis=1)


#rename the columns
syn_avg_profile.rename(columns = {'targetDose': 'DOSE_LEVEL',
                                  'targetTime': 'SACRIFICE_PERIOD'}, inplace=True)



# Get the feature columns (compound name, dose, time period)
feature_columns = ['COMPOUND_NAME', 'DOSE_LEVEL', 'SACRIFICE_PERIOD', 'COMPOUND_ABBREVIATION']

# Identify the gene columns
gene_columns = [col for col in real_avg_profile.columns if col not in feature_columns]

# Create a copy of the real_avg_profile DataFrame
heatmap_df = real_avg_profile.copy()

# Perform the calculation (real - syn) / real for each gene
for gene in gene_columns:
    heatmap_df[gene] = (real_avg_profile[gene] - syn_avg_profile[gene]) / real_avg_profile[gene]

# View the resulting DataFrame
heatmap_df[feature_columns + gene_columns]



# Set the index to compound name (or abbreviate if needed)
# Assuming 'COMPOUND_NAME' column contains the compound names
heatmap_data = heatmap_df.set_index('COMPOUND_ABBREVIATION')[gene_columns]

# Convert the gene expression values to percentage (i.e., the expression values as a percentage of the real expression)
heatmap_percentage = heatmap_data.applymap(lambda x: x * 100)  # Convert to percentage




# Calculate the min and max values for better control of the color scale
vmin = heatmap_percentage.min().min()  # Get the minimum value in the dataframe
vmax = heatmap_percentage.max().max()  # Get the maximum value in the dataframe

# Plot the heatmap
plt.figure(figsize=(20, 10), dpi=600)
ax = sns.heatmap(
    heatmap_percentage.T,  # Transpose if necessary to have genes as y-axis
    annot=True,  # Annotate cells with the actual data value
    fmt='.1f',  # Format the annotation to 1 decimal place
    cmap='coolwarm',  # Choose a diverging color map
    cbar_kws={'label': 'Percentage'},  # Color bar label
    xticklabels=True,  # Show x-axis tick labels
    annot_kws={'size': 14},  # Set annotation font size
    vmin=vmin,  # Set the minimum value for the color scale
    vmax=vmax,  # Set the maximum value for the color scale
)

# Customize the axes
plt.xlabel('Compounds', fontsize=20)
plt.ylabel('AOP Genes', fontsize=20)

# Increase the font size for x and y axis tick labels
plt.xticks(fontsize=16)  # Increase the font size for x-axis tick labels
plt.yticks(fontsize=16)  # Increase the font size for y-axis tick labels

# Adjust colorbar font size
cb = ax.collections[0].colorbar
cb.set_label('Percentage Error', fontsize=16)  # Set the colorbar label font size
cb.ax.tick_params(labelsize=16)  # Set the colorbar ticks font size

# Display the plot
plt.tight_layout()

# Save the plot to a file (high resolution)
plt.savefig('AOP_heatmap_python.png', dpi=600)  # Save at 600 dpi for even better resolution

# Show the plot
plt.show()





































