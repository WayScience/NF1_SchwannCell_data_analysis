#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pathlib
import numpy as np


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


# ## Function for pairwise correlations

# In[3]:


def generate_correlations(df, feat_cols):
    # Copy df so that data is not lost
    df_corr = df.copy()

    # Generate Pearson correlations between all wells
    correlations = df_corr.loc[:, feat_cols].transpose().corr(method='pearson')

    # Remove the lower triangle
    correlations = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))

    # Flip, reset index, and add column names
    correlations = correlations.stack().reset_index()
    correlations.columns = ['group0_index', 'group1_index', 'correlation']

    # Map index to corresponding Metadata_Well__group
    correlations['Metadata_Well__group0'] = df.loc[correlations['group0_index'], 'Metadata_Well'].values
    correlations['Metadata_Well__group1'] = df.loc[correlations['group1_index'], 'Metadata_Well'].values

    correlations['Metadata_genotype__group0'] = df.loc[correlations['group0_index'], 'Metadata_genotype'].values
    correlations['Metadata_genotype__group1'] = df.loc[correlations['group1_index'], 'Metadata_genotype'].values

    # Map index to Metadata_plate
    correlations['Metadata_plate__group0'] = df.loc[correlations['group0_index'], 'Metadata_Plate'].values
    correlations['Metadata_plate__group1'] = df.loc[correlations['group1_index'], 'Metadata_Plate'].values

    # Conditionally include Metadata_seed_density
    if 'Metadata_seed_density' in df.columns:
        correlations['Metadata_seed_density__group0'] = df.loc[correlations['group0_index'], 'Metadata_seed_density'].values
        correlations['Metadata_seed_density__group1'] = df.loc[correlations['group1_index'], 'Metadata_seed_density'].values
    else: # Default to 0 since the column has to be of the same type (can't be a str)
        correlations['Metadata_seed_density__group0'] = 0
        correlations['Metadata_seed_density__group1'] = 0

    # Drop the index columns
    correlations = correlations.drop(columns=['group0_index', 'group1_index'])

    return correlations


# ## Load in data and compute correlations per well

# ### Plate 4 only controls

# In[4]:


# Load in plate 4 dataframe (only controls)
plate4_path = pathlib.Path(f"{root_dir}/../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_4_bulk_camerons_method.parquet")
plate4df = pd.read_parquet(plate4_path)
# Fill missing values in Metadata_siRNA column with 'No Construct'
plate4df['Metadata_siRNA'] = plate4df['Metadata_siRNA'].fillna('No Construct')
# Only include rows where Metadata_siRNA contains 'No Construct'
plate4df = plate4df[plate4df['Metadata_siRNA'].str.contains('No Construct')]

print(plate4df.shape)
plate4df.head()


# In[5]:


# Identify feature columns
feat_cols = [col for col in plate4df.columns if not col.startswith('Metadata')]

plate4_corr = generate_correlations(df=plate4df, feat_cols=feat_cols)

print(plate4_corr.shape)
plate4_corr.head()


# ### Plate 3

# In[6]:


# Load in plate 3 dataframe
plate3_path = pathlib.Path(f"{root_dir}/../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_3_bulk_camerons_method.parquet")
plate3df = pd.read_parquet(plate3_path)

print(plate3df.shape)
plate3df.head()


# In[7]:


# Identify feature columns
feat_cols = [col for col in plate3df.columns if not col.startswith('Metadata')]

plate3_corr = generate_correlations(df=plate3df, feat_cols=feat_cols)

print(plate3_corr.shape)
plate3_corr.head()


# ### Plate 3 prime

# In[8]:


# Load in plate 3 prime dataframe
plate3p_path = pathlib.Path(f"{root_dir}/../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_3_prime_bulk_camerons_method.parquet")
plate3pdf = pd.read_parquet(plate3p_path)

# Update Metadata_Plate for all rows
plate3pdf['Metadata_Plate'] = 'Plate_3_prime'

print(plate3pdf.shape)
plate3pdf.head()


# In[9]:


# Identify feature columns
feat_cols = [col for col in plate3pdf.columns if not col.startswith('Metadata')]

plate3p_corr = generate_correlations(df=plate3pdf, feat_cols=feat_cols)

print(plate3p_corr.shape)
plate3p_corr.head()


# ### Plate 5

# In[10]:


# Load in plate 3 rpime dataframe
plate5_path = pathlib.Path(f"{root_dir}/../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_5_bulk_camerons_method.parquet")
plate5df = pd.read_parquet(plate5_path)

print(plate5df.shape)
plate5df.head()


# In[11]:


# Identify feature columns
feat_cols = [col for col in plate5df.columns if not col.startswith('Metadata')]

plate5_corr = generate_correlations(df=plate5df, feat_cols=feat_cols)

print(plate5_corr.shape)
plate5_corr.head()


# In[12]:


# Concatenate the DataFrames vertically
concatenated_df = pd.concat([plate4_corr, plate5_corr, plate3_corr, plate3p_corr], ignore_index=True)

# Save the concatenated DataFrame as a Parquet file
concatenated_df.to_parquet('./construct_correlation_data/concatenated_all_plates_correlations.parquet', index=False)

print(concatenated_df.shape)
concatenated_df.head()

