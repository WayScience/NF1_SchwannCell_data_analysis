#!/usr/bin/env python
# coding: utf-8

# # Training a Logistic Regression Model

# ## Imports

# In[1]:


import pandas as pd
import sys
from pathlib import Path
from joblib import dump


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


# ## Import Utils

# In[3]:


sys.path.append(f"{root_dir}/1.train_models/log_reg_plates_cp_fs_data/utils")
import log_reg_plates_cp_fs_data_train_util as au


# In[4]:


# Random integer as a seed
rnd_val = 0


# ## Create paths

# In[5]:


filename = "Plate_2_sc_norm_fs.parquet"
plate_path = Path(
    f"{root_dir}/nf1_painting_repo/3.processing_features/data/feature_selected_data"
)

path = plate_path / filename


data_path = Path("data")
output_prefix = "plate_2_cp_fs_data"
# Create the parent directories if they don't exist
data_path.mkdir(parents=True, exist_ok=True)


# ## Generate plate dataframe

# In[6]:


platedf = pd.read_parquet(path)


# ## Get the best model, the test set, and the label encode

# In[7]:


lr, testdf, le = au.get_model_data(platedf)


# ## Save Data

# In[8]:


dump(lr, data_path / f"{output_prefix}_lr_model.joblib")
testdf.to_csv(f"{data_path}/{output_prefix}_testdf.tsv", sep="\t", index=False)
dump(le, data_path / f"{output_prefix}_label_encoder.joblib")
