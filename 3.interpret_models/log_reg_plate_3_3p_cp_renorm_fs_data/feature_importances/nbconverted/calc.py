#!/usr/bin/env python
# coding: utf-8

# # Determine the best features using a Logistic Regression Model

# ## Imports

# In[1]:


import sys
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load


# ## Find the git root Directory

# In[2]:


# Get the current working directory
cwd = Path.cwd()

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


# ## Import Utilities

# In[3]:


sys.path.append(f"{root_dir}/utils")


# # Seed and Generator for Reproducibility

# In[4]:


rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator


# In[5]:


data_path = Path("data")
filename = "feature_importances.tsv"

if not data_path.exists():
    data_path.mkdir()

data_path = data_path / filename


# ## Load Model

# In[6]:


models_path = Path(
    f"{root_dir}/1.train_models/log_reg_plate_3_3p_cp_renorm_fs_data/data"
)
lr = load(models_path / "lr_model.joblib")


# ## Save Data

# In[7]:


testdf = load(models_path / "testdf.joblib")
le = load(models_path / "label_encoder.joblib")


# ## Create Dataframe with coefficients for each Genotype

# In[8]:


featdf = pd.DataFrame(lr.coef_.T, columns=le.classes_.tolist())
featdf["feature"] = testdf.drop(["label"], axis=1).columns


# ## Save the feature importance data

# In[9]:


featdf.to_csv(data_path, sep="\t", index=False)
