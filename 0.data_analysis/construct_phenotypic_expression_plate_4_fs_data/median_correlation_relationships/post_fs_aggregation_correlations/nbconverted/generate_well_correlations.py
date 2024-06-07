#!/usr/bin/env python
# coding: utf-8

# # Generate pairwise correlations from Plate 4 between wells of the same dose per construct or no construct
# 
# Note: We are currently not including NF1 Target 2 at this time due to not have titration curve data for that construct.

# ## Import libraries

# In[1]:


import pandas as pd
import pathlib
import numpy as np


# ## Get root directory to access data

# In[2]:


# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")


# ## Load in Plate 4 data

# In[3]:


# Load in plate 4 dataframe (includes controls and siRNAs)
plate4_path = pathlib.Path(f"{root_dir}/../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_4_bulk_camerons_method.parquet")
plate4df = pd.read_parquet(plate4_path)

# Fill missing values in Metadata_siRNA column with 'No Construct'
plate4df['Metadata_siRNA'] = plate4df['Metadata_siRNA'].fillna('No Construct')

# Remove rows where Metadata_siRNA contains NF1 Target 2
plate4df = plate4df[~plate4df['Metadata_siRNA'].str.contains('NF1 Target 2')]

print(plate4df.shape)
plate4df.head()


# ## Split out metadata and feature columns in the data frame

# In[4]:


meta_cols = [col for col in plate4df.columns if col.startswith('Metadata')]
feat_cols = [col for col in plate4df.columns if col not in meta_cols]


# ## Compute correlations between each well

# In[5]:


# Generate pearson correlations between all wells
plate4_correlations = plate4df.loc[:, feat_cols].transpose().corr(method='pearson')

# Remove the lower triangle
plate4_correlations = plate4_correlations.where(np.triu(np.ones(plate4_correlations.shape), k=1).astype(bool))

# Flip, reset index, and add column names
plate4_correlations = plate4_correlations.stack().reset_index()
plate4_correlations.columns = ['group0_index', 'group1_index', 'correlation']

print(plate4_correlations.shape)
plate4_correlations.head()


# ## Add important metadata for each correlation duo

# In[6]:


# Map index to corresponding Metadata_Well__group
plate4_correlations['Metadata_Well__group0'] = plate4df.loc[plate4_correlations['group0_index'], 'Metadata_Well'].values
plate4_correlations['Metadata_Well__group1'] = plate4df.loc[plate4_correlations['group1_index'], 'Metadata_Well'].values

# Add Metadata_siRNA and Metadata_Concentration for group0
plate4_correlations['Metadata_siRNA__group0'] = plate4df.loc[plate4_correlations['group0_index'], 'Metadata_siRNA'].values
plate4_correlations['Metadata_Concentration__group0'] = plate4df.loc[plate4_correlations['group0_index'], 'Metadata_Concentration'].values
plate4_correlations['Metadata_genotype__group0'] = plate4df.loc[plate4_correlations['group0_index'], 'Metadata_genotype'].values

# Add Metadata_siRNA and Metadata_Concentration for group1
plate4_correlations['Metadata_siRNA__group1'] = plate4df.loc[plate4_correlations['group1_index'], 'Metadata_siRNA'].values
plate4_correlations['Metadata_Concentration__group1'] = plate4df.loc[plate4_correlations['group1_index'], 'Metadata_Concentration'].values
plate4_correlations['Metadata_genotype__group1'] = plate4df.loc[plate4_correlations['group1_index'], 'Metadata_genotype'].values

print(plate4_correlations.shape)
plate4_correlations.head()


# ## Ensure that the constructs do not flip flop groups

# In[7]:


# Define the correct priority order
priority = {"NF1 Target 1": 1, "No Construct": 2, "Scramble": 3}

# Map the priority values to the siRNA columns
group0_priority = plate4_correlations['Metadata_siRNA__group0'].map(priority)
group1_priority = plate4_correlations['Metadata_siRNA__group1'].map(priority)

# Swap groups based on priority
swap_mask = (
    ((plate4_correlations['Metadata_siRNA__group0'] == 'Scramble') & 
    (plate4_correlations['Metadata_siRNA__group1'] != 'Scramble')) |
    ((plate4_correlations['Metadata_siRNA__group0'] == 'No Construct') & 
    (plate4_correlations['Metadata_siRNA__group1'] == 'NF1 Target 1')) |
    (group0_priority > group1_priority)
)

# Swap the values where needed
plate4_correlations.loc[swap_mask, ['Metadata_siRNA__group0', 'Metadata_siRNA__group1']] = plate4_correlations.loc[swap_mask, ['Metadata_siRNA__group1', 'Metadata_siRNA__group0']].values
plate4_correlations.loc[swap_mask, ['Metadata_Concentration__group0', 'Metadata_Concentration__group1']] = plate4_correlations.loc[swap_mask, ['Metadata_Concentration__group1', 'Metadata_Concentration__group0']].values
plate4_correlations.loc[swap_mask, ['Metadata_genotype__group0', 'Metadata_genotype__group1']] = plate4_correlations.loc[swap_mask, ['Metadata_genotype__group1', 'Metadata_genotype__group0']].values

print(plate4_correlations.shape)
plate4_correlations.head()


# ## Remove any rows where the doses don't match

# In[8]:


# Remove rows where both groups are NF1 Target 1 and their concentrations do not match
filtered_df = plate4_correlations[~(
    ((plate4_correlations['Metadata_siRNA__group0'] == 'NF1 Target 1') & (plate4_correlations['Metadata_siRNA__group1'] == 'NF1 Target 1')) &
    (plate4_correlations['Metadata_Concentration__group0'] != plate4_correlations['Metadata_Concentration__group1'])
)]


# Remove rows where both groups are Scramble and their concentrations do not match
filtered_df = filtered_df[~(
    ((filtered_df['Metadata_siRNA__group0'] == 'Scramble') & (filtered_df['Metadata_siRNA__group1'] == 'Scramble')) &
    (filtered_df['Metadata_Concentration__group0'] != filtered_df['Metadata_Concentration__group1'])
)]

# Remove rows where one group is NF1 Target 1 and the other is Scramble and the concentrations do not match
filtered_df = filtered_df[~(
    (((filtered_df['Metadata_siRNA__group0'] == 'NF1 Target 1') & (filtered_df['Metadata_siRNA__group1'] == 'Scramble')) |
    ((filtered_df['Metadata_siRNA__group1'] == 'NF1 Target 1') & (filtered_df['Metadata_siRNA__group0'] == 'Scramble'))) &
    (filtered_df['Metadata_Concentration__group0'] != filtered_df['Metadata_Concentration__group1'])
)]

print(filtered_df.shape)
filtered_df.head()


# ## Merge concentrations into one column

# In[9]:


# Create a new column for Metadata_Concentration
filtered_df['Metadata_Concentration'] = filtered_df.apply(lambda row:
    row['Metadata_Concentration__group0'] if row['Metadata_Concentration__group1'] == 0 else
    row['Metadata_Concentration__group1'] if row['Metadata_Concentration__group0'] == 0 else
    row['Metadata_Concentration__group0'] if row['Metadata_Concentration__group0'] == row['Metadata_Concentration__group1'] else
    'Different doses found!',
    axis=1
)

# Remove Metadata_Concentration group columns and group index columns
final_df = filtered_df.drop(['Metadata_Concentration__group0', 'Metadata_Concentration__group1', 'group0_index', 'group1_index'], axis=1)

print(final_df.shape)
final_df.head()


# ## Save correlations as a parquet file for downstream analysis

# In[10]:


# Save correlation data as a parquet file
final_df.to_parquet("./construct_correlation_data/plate_4_well_correlations.parquet", index=False)

