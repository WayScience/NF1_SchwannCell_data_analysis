#!/usr/bin/env python
# coding: utf-8

# # Well-Aggregated Plate and Genotype Correlation Analysis
# Correlations between groups defined by genotype and plate are determined to understand the similarities between group morphologies.
# These correlations are computed between cell morphologies aggregated to the well level after feature selection.

# In[1]:


import pathlib
import sys

import pandas as pd

# Path to correlation class
sys.path.append(
    "../utils"
)

# Class for calculating correlations
from CorrelateData import CorrelateData

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


# # Inputs

# In[3]:


data_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles").resolve(strict=True)

plate3df_path = pathlib.Path(root_dir / data_path / "Plate_3_bulk_camerons_method.parquet").resolve(strict=True)
plate3pdf_path = pathlib.Path(root_dir / data_path / "Plate_3_prime_bulk_camerons_method.parquet").resolve(strict=True)
plate5df_path = pathlib.Path(root_dir / data_path / "Plate_5_bulk_camerons_method.parquet").resolve(strict=True)

plate3df = pd.read_parquet(plate3df_path)
plate3pdf = pd.read_parquet(plate3pdf_path)
plate5df = pd.read_parquet(plate5df_path)


# # Outputs

# In[4]:


plate_correlation_path = pathlib.Path("construct_correlation_data/well_agg_plate_genotype_correlations.parquet")
plate_correlation_path.mkdir(parents=True, exist_ok=True)


# # Process Bulk Plate Data

# ## Combine data
# Concat plate data and retain common columns.

# In[5]:


plates_cols = plate3df.columns.intersection(plate3pdf.columns).intersection(plate5df.columns)
platesdf = pd.concat([plate3df, plate3pdf, plate5df], axis=0)
platesdf = platesdf[plates_cols]


# In[6]:


# Morphology and metadata columns
morph_cols = [col for col in platesdf.columns if "Metadata" not in col]
meta_cols = platesdf.columns.difference(morph_cols)


# # Correlate wells
# Wells are correlated between plate and genotype.

# In[7]:


cd = CorrelateData()
correlationsdf = []


# In[8]:


cd.intra_correlations(
    _df=plate3df.loc[plate3df["Metadata_genotype"] == "WT"].copy(),
    _antehoc_group_cols=["Metadata_Plate", "Metadata_genotype"],
    _feat_cols=morph_cols,
    _posthoc_group_cols=["Metadata_Well"],
    _drop_cols=["Metadata_Well"]
)


# ## Well Correlations (same genotypes different plates)

# In[9]:


for genotype in platesdf["Metadata_genotype"].unique():

    correlation_params = {
    }

    correlationsdf.append(
        cd.inter_correlations(
            _df=platesdf.loc[platesdf["Metadata_genotype"] == genotype].copy(),
            _antehoc_group_cols=["Metadata_Plate"],
            _feat_cols=morph_cols,
            _posthoc_group_cols=["Metadata_Well", "Metadata_genotype"],
            _drop_cols=["Metadata_Well"]
        )
    )


# ## Well Correlations (different genotypes and all possible plates)

# In[10]:


correlationsdf.append(
    cd.inter_correlations(
        _df=platesdf.copy(),
        _antehoc_group_cols=["Metadata_genotype"],
        _feat_cols=morph_cols,
        _posthoc_group_cols=["Metadata_Plate", "Metadata_Well"],
        _drop_cols=["Metadata_Well"]
    )
)


# ## Well Correlations (same genotype and same plate)

# In[11]:


correlationsdf.append(
    cd.intra_correlations(
        _df=platesdf.copy(),
        _antehoc_group_cols=["Metadata_Plate", "Metadata_genotype"],
        _feat_cols=morph_cols,
        _posthoc_group_cols=["Metadata_Well"],
        _drop_cols=["Metadata_Well"]
    )
)


# # Save Plate Correlations

# In[12]:


correlationsdf = pd.concat(correlationsdf, axis=0)

correlationsdf.to_parquet()


# In[13]:


correlationsdf.head()

