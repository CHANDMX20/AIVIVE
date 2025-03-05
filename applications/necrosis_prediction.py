#load the packages 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

#load the data
necrosis_df = pd.read_csv('/path/to/necrosis_df.csv')     #path to necrosis finding data for in vivo profiles from TG-GATEs
vivo_train = pd.read_csv('/path/to/vivo_train.csv', low_memory=False)     #path to vivo train data (Real)
rat_genes = pd.read_csv('/path/to/final_rat_genes.csv').PROBEID.unique()  

#subset the vivo train for rat S1500+ gene set
vivo_train = pd.concat([vivo_train.iloc[:, :19], vivo_train[rat_genes]], axis=1)

# Add treatment column to the data frames
vivo_train['treatment'] = vivo_train['COMPOUND_NAME'] + '_' + vivo_train['DOSE_LEVEL'].astype(str) + '_' + vivo_train['SACRIFICE_PERIOD'].astype(str)
necrosis_df['treatment'] = necrosis_df['COMPOUND_NAME'] + '_' + necrosis_df['DOSE_LEVEL'].astype(str) + '_' + necrosis_df['SACRIFICE_PERIOD'].astype(str)


# Identify all columns that are numeric (gene expression columns)
gene_columns = vivo_train.iloc[:, 19:-1].columns
#Group by 'treatment' and calculate the mean of gene expression columns (average of replicates for evrey treatment)
vivo_df = vivo_train.groupby('treatment')[gene_columns].mean().reset_index()


# Identify feature columns (non-numeric, excluding 'treatment')
feature_columns = vivo_train.iloc[:, list(range(19)) + [-1]].columns
# Extract the feature columns and drop duplicates based on 'treatment'
vivo_features = vivo_train[feature_columns].drop_duplicates(subset='treatment').reset_index(drop = True)
# Now, merge the feature columns with the averaged gene expression data
vivo_df_final = pd.merge(vivo_features, vivo_df, on='treatment', how='left')


# Perform the left join on 'treatment', keeping all columns of vivo_df_final and only the 'FINDING_TYPE' from necrosis_df_filtered
vivo_necrosis = pd.merge(vivo_df_final, necrosis_df[['treatment', 'FINDING_TYPE']], on='treatment', how='left')
# Drop duplicate rows based on 'treatment' (if any)
vivo_necrosis = vivo_necrosis.drop_duplicates(subset='treatment').reset_index(drop=True)


# Replace 'Necrosis' with 'Necrosis Positive' and NaN with 'Necrosis Negative'
vivo_necrosis['FINDING_TYPE'] = vivo_necrosis['FINDING_TYPE'].replace({'Necrosis': 'Necrosis Positive'})
vivo_necrosis['FINDING_TYPE'] = vivo_necrosis['FINDING_TYPE'].fillna('Necrosis Negative')

# Filter the DataFrame to keep only rows where FINDING_TYPE is 'Necrosis Negative' or 'Necrosis Positive'
vivo_necrosis = vivo_necrosis[vivo_necrosis['FINDING_TYPE'].isin(['Necrosis Negative', 'Necrosis Positive'])].reset_index(drop=True)

# Filter rows for Necrosis Positive and Necrosis Negative
necrosis_positive = vivo_necrosis[vivo_necrosis['FINDING_TYPE'] == 'Necrosis Positive']
necrosis_negative = vivo_necrosis[vivo_necrosis['FINDING_TYPE'] == 'Necrosis Negative']

# Randomly select 111 rows from Necrosis Negative to match Necrosis Positive count
necrosis_negative_sample = necrosis_negative.sample(n=111, random_state=42)
# Combine the Necrosis Positive and sampled Necrosis Negative rows into a new DataFrame
new_df = pd.concat([necrosis_positive, necrosis_negative_sample])

# Select the gene expression columns (assuming all columns except 'finding_type' are gene columns)
# Drop the first 19 columns and the 'finding_type' column
X = new_df.drop(columns=['FINDING_TYPE']).iloc[:, 20:]
# The target column
y = new_df['FINDING_TYPE']
# Ensure the target is in a binary format, for example, encoding "Necrosis positive" as 1 and "Necrosis negative" as 0
y = y.map({'Necrosis Positive': 1, 'Necrosis Negative': 0})


# Scale the features to standardize them
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#model1 = LogisticRegression(penalty = 'l2', C = 0.1)
model1 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
model2 = SVC(kernel='rbf', C=1, gamma='scale', random_state=42, class_weight='balanced')
model3 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

ensemble_model = VotingClassifier(estimators=[('xg', model1), ('svm', model2), ('rf', model3)], voting='hard')
ensemble_model.fit(X_scaled, y)


## Model Evaluation on Real Test Set

#load the real in vivo test profiles
vivo_real = pd.read_csv('/path/to/vivo_test.csv', low_memory=False)

# subset for specific cols
vivo_real = pd.concat([vivo_real.iloc[:, :19], vivo_real[rat_genes]], axis=1)
#create treatment column
vivo_real['treatment'] = vivo_real['COMPOUND_NAME'] + '_' + vivo_real['DOSE_LEVEL'].astype(str) + '_' + vivo_real['SACRIFICE_PERIOD'].astype(str)

#Group by 'treatment' and calculate the mean of gene expression columns
real_df = vivo_real.groupby('treatment')[gene_columns].mean().reset_index()
# Extract the feature columns and drop duplicates based on 'treatment'
real_features = vivo_real[feature_columns].drop_duplicates(subset='treatment').reset_index(drop = True)
# Now, merge the feature columns with the averaged gene expression data
vivo_real_df = pd.merge(real_features, real_df, on='treatment', how='left')

# Perform the left join on 'treatment', keeping all columns of vivo_df_final and only the 'FINDING_TYPE' from necrosis_df_filtered
real_necrosis = pd.merge(vivo_real_df, necrosis_df[['treatment', 'FINDING_TYPE']], on='treatment', how='left')
# Drop duplicate rows based on 'treatment' (if any)
real_necrosis = real_necrosis.drop_duplicates(subset='treatment').reset_index(drop=True)

# Replace 'Necrosis' with 'Necrosis Positive' and NaN with 'Necrosis Negative'
real_necrosis['FINDING_TYPE'] = real_necrosis['FINDING_TYPE'].replace({'Necrosis': 'Necrosis Positive'})
real_necrosis['FINDING_TYPE'] = real_necrosis['FINDING_TYPE'].fillna('Necrosis Negative')

# Filter the DataFrame to keep only rows where FINDING_TYPE is 'Necrosis Negative' or 'Necrosis Positive'
real_necrosis = real_necrosis[real_necrosis['FINDING_TYPE'].isin(['Necrosis Negative', 'Necrosis Positive'])].reset_index(drop=True)

#Feature columns 
real_X = real_necrosis.drop(columns=['FINDING_TYPE']).iloc[:, 20:]
# The target column
real_y = real_necrosis['FINDING_TYPE']
# Ensure the target is in a binary format, for example, encoding "Necrosis positive" as 1 and "Necrosis negative" as 0
real_y = real_y.map({'Necrosis Positive': 1, 'Necrosis Negative': 0})

# Scale the features to standardize them
real_X_scaled = scaler.fit_transform(real_X)

#predict on the test set 
voting_real_preds = ensemble_model.predict(real_X_scaled)

# Calculate accuracy
voting_real_accuracy = accuracy_score(real_y, voting_real_preds)
print(f'Accuracy: {voting_real_accuracy * 100:.2f}%')

# Confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(real_y, voting_real_preds))

# Classification report
print('Classification Report:')
print(classification_report(real_y, voting_real_preds))


## Model Evaluation on Synthetic Test Profile

#load the data
vivo_syn = pd.read_csv('/path/to/combined_opt_gan_test9962160_Vivo.csv', low_memory = False)  #path to optimized synthetic in vivo profile

#subset for specific cols
vivo_syn = pd.concat([vivo_syn.iloc[:, :23], vivo_syn[rat_genes]], axis=1)
# create treatment column
vivo_syn['treatment'] = vivo_syn['COMPOUND_NAME'] + '_' + vivo_syn['targetDose'].astype(str) + '_' + vivo_syn['targetTime'].astype(str)

# Identify feature columns (non-numeric, excluding 'treatment')
syn_feature_cols = vivo_syn.iloc[:, list(range(23)) + [-1]].columns
#Group by 'treatment' and calculate the mean of gene expression columns
syn_df = vivo_syn.groupby('treatment')[gene_columns].mean().reset_index()
# Extract the feature columns and drop duplicates based on 'treatment'
syn_features = vivo_syn[syn_feature_cols].drop_duplicates(subset='treatment').reset_index(drop = True)
# Now, merge the feature columns with the averaged gene expression data
vivo_syn_df = pd.merge(syn_features, syn_df, on='treatment', how='left')

# Perform the left join on 'treatment', keeping all columns of vivo_df_final and only the 'FINDING_TYPE' from necrosis_df_filtered
syn_necrosis = pd.merge(vivo_syn_df, necrosis_df[['treatment', 'FINDING_TYPE']], on='treatment', how='left')
# Drop duplicate rows based on 'treatment' (if any)
syn_necrosis = syn_necrosis.drop_duplicates(subset='treatment').reset_index(drop=True)

# Replace 'Necrosis' with 'Necrosis Positive' and NaN with 'Necrosis Negative'
syn_necrosis['FINDING_TYPE'] = syn_necrosis['FINDING_TYPE'].replace({'Necrosis': 'Necrosis Positive'})
syn_necrosis['FINDING_TYPE'] = syn_necrosis['FINDING_TYPE'].fillna('Necrosis Negative')

# Filter the DataFrame to keep only rows where FINDING_TYPE is 'Necrosis Negative' or 'Necrosis Positive'
syn_necrosis = syn_necrosis[syn_necrosis['FINDING_TYPE'].isin(['Necrosis Negative', 'Necrosis Positive'])].reset_index(drop=True)

# feature columns
syn_X = syn_necrosis.drop(columns=['FINDING_TYPE']).iloc[:, 24:]
# The target column
syn_y = syn_necrosis['FINDING_TYPE']
# Ensure the target is in a binary format, for example, encoding "Necrosis positive" as 1 and "Necrosis negative" as 0
syn_y = syn_y.map({'Necrosis Positive': 1, 'Necrosis Negative': 0})

# Scale the features to standardize them
syn_X_scaled = scaler.fit_transform(syn_X)

# make predictions on the test set 
voting_syn_preds = ensemble_model.predict(syn_X_scaled)

# Calculate accuracy
voting_syn_accuracy = accuracy_score(syn_y, voting_syn_preds)
print(f'Accuracy: {voting_syn_accuracy * 100:.2f}%')

# Confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(syn_y, voting_syn_preds))

# Classification report
print('Classification Report:')
print(classification_report(syn_y, voting_syn_preds))






