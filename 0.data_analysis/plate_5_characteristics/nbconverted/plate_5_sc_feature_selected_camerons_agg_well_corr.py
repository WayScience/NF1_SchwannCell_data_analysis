#!/usr/bin/env python
# coding: utf-8

# # Correlate Cell Profiler Aggregated Wells in Plate 5

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
    f"{root_dir}/0.data_analysis/utils"
)

# Class for calculating correlations
from CorrelateData import CorrelateData

platedf_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_5_bulk_camerons_method.parquet").resolve(strict=True)
platedf = pd.read_parquet(platedf_path)


# ### Output paths

# In[4]:


data_path = pathlib.Path("plate_5_sc_feature_selected_camerons_agg_well_corr_data")
data_path.mkdir(parents=True, exist_ok=True)


# ## Drop missing columns

# In[5]:


platedf.dropna(inplace=True)


# In[6]:


meta_cols = platedf.filter(like="Metadata").columns
feat_cols = platedf.drop(columns=meta_cols).columns


# ## Compute Correlations

# In[7]:


cd = CorrelateData()
correlationsdf = []

correlation_params = {
    "_df": platedf.reset_index(drop=True),
    "_antehoc_group_cols": ["Metadata_genotype"],
    "_feat_cols": feat_cols,
    "_posthoc_group_cols": ["Metadata_Well"]
}

# Correlates aggregated wells across genotype
correlationsdf.append(cd.inter_correlations(**correlation_params))


# In[8]:


# Correlates aggregated wells within genotype
correlationsdf.append(cd.intra_correlations(**correlation_params))


# ## Store Correlation Data

# In[9]:


correlationsdf = pd.concat(correlationsdf, axis=0)
correlationsdf.to_parquet(f"{data_path}/plate_5_sc_feature_selected_camerons_agg_well_corr.parquet")


# In[10]:


correlationsdf.head()

