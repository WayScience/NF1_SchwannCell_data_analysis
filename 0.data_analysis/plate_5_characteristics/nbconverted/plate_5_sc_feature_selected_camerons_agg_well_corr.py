#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import pandas as pd

# ## Find the root of the git repo on the host system

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


# ## Define paths

# ### Input paths

# In[3]:


# Path to correlation class
sys.path.append(
    f"{root_dir}/0.data_analysis/plate_5_characteristics/utils"
)

platedf_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_5_sc_feature_selected.parquet").resolve(strict=True)
platedf = pd.read_parquet(platedf_path)

# Class for calculating correlations
from CorrelatePlate import CorrelatePlate

# ### Output paths

# In[4]:


fig_path = pathlib.Path("plate_5_sc_feature_selected_figures")
fig_path.mkdir(parents=True, exist_ok=True)


# ## Drop missing columns

# In[5]:


platedf.dropna(inplace=True)


# ## Aggregate cells with cameron's method

# In[6]:


meta_cols = platedf.filter(like="Metadata").columns
feat_cols = platedf.drop(columns=meta_cols).columns

median_cols = {col_name: "median" for col_name in platedf.columns if col_name not in meta_cols}

# Set metadata columns to lambda functions set to the first row
meta_cols = {
    col_name: lambda x: x.iloc[0]
    for col_name in meta_cols
}

# Combine the dictionaries
median_cols.update(meta_cols)

# Aggregate the plate data
welldf = platedf.groupby("Metadata_Well").agg(median_cols)


# ## Compute Correlations

# In[7]:


cp = CorrelatePlate()
correlationsdf = cp.correlate_agg_wells(welldf, "Metadata_Well", feat_cols, "Metadata_genotype")


# In[8]:


correlationsdf

