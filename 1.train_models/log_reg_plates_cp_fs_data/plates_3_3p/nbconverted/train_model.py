#!/usr/bin/env python
# coding: utf-8

# # Training a Logistic Regression Model

# ## Imports

# In[1]:


import pandas as pd
import sys
from pathlib import Path
from joblib import dump
from sklearn.linear_model import LogisticRegression


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


# Random value for all seeds
rnd_val = 0


# # Converting csv to pandas dataframe

# In[5]:


filename3 = "Plate_3_sc_norm_fs.parquet"
filename3p = "Plate_3_prime_sc_norm_fs.parquet"
plate_path = Path(
    f"{root_dir}/nf1_painting_repo/3.processing_features/data/feature_selected_data"
)

data_path = Path("data")

data_path.mkdir(
    parents=True, exist_ok=True
)  # Create the parent directories if they don't exist

output_prefix = "plates_3_3p_cp_fs_data"

path3 = plate_path / filename3

path3p = plate_path / filename3p


# ## Generate plate dataframes

# In[6]:


# Returns the dataframe returned by the plate 3 parquet file
plate3df = pd.read_parquet(path3)

# Returns the dataframe returned by the plate 3 prime parquet file
plate3pdf = pd.read_parquet(path3p)


# # Preprocess Data

# ## Use only common columns

# In[7]:


# Set plate column:
plate3df["Metadata_plate"] = "3"
plate3pdf["Metadata_plate"] = "3p"

common_columns = list(plate3df.columns.intersection(plate3pdf.columns))
plate3df = plate3df.loc[:, common_columns]
plate3pdf = plate3pdf.loc[:, common_columns]

# Combine the plate dataframes:
platedf = pd.concat([plate3df, plate3pdf], axis="rows")


# ## Get the best model, the test set, and the label encode

# In[8]:


lr = LogisticRegression(
    max_iter=1000, solver="sag", multi_class="ovr", random_state=rnd_val, n_jobs=-1
)
lr, testdf, le = au.get_model_data(platedf, lr, will_cross_validate=False)


# ## Save Data

# In[9]:


dump(lr, data_path / f"{output_prefix}_lr_model.joblib")
testdf.to_csv(f"{data_path}/{output_prefix}_testdf.tsv", sep="\t", index=False)
dump(le, data_path / f"{output_prefix}_label_encoder.joblib")
