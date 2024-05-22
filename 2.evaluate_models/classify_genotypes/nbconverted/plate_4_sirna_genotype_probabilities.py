#!/usr/bin/env python
# coding: utf-8

# # Evaluate model performance siRNA-treated single cells
# Genotype probabilies of siRNA-treated single cell data are computed to evaluate model perormance.
# This is performed on plate 4.

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from joblib import load

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

# ### Input

# In[3]:


plate4df_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_4_sc_feature_selected.parquet").resolve(strict=True)
plate4df = pd.read_parquet(plate4df_path)

data_path = pathlib.Path(root_dir / "1.train_models/classify_genotypes/data").resolve(strict=True)
le = load(f"{data_path}/trained_nf1_model_label_encoder.joblib")
model = load(f"{data_path}/trained_nf1_model.joblib")

# Set the seed
rng = np.random.default_rng(0)


# ### Outputs

# In[4]:


probability_path = pathlib.Path("genotype_probabilities")
probability_path.mkdir(parents=True, exist_ok=True)


# ## Process plate 4

# In[5]:


# Removed siRNA-treated cells to retain only Null and WT cells
plate4df["Metadata_siRNA"].fillna("No Construct", inplace=True)
plate4df.dropna(inplace=True)
plate4df = plate4df.loc[plate4df["Metadata_siRNA"] != "No Construct"]

meta_cols = [col for col in plate4df.columns if "Metadata" in col]


# ## Save siRNA genotype probabilities

# In[6]:


probabilitydf = pd.DataFrame(
    {
        f"probability {le.inverse_transform([1])[0]}":
        model.predict_proba(plate4df[model.feature_names_in_])[:, 1]
    }
)

pd.concat([probabilitydf, plate4df[meta_cols]], axis=1).to_parquet(
    f"{probability_path}/plate_4_sirna_single_cell_probabilities.parquet"
)

