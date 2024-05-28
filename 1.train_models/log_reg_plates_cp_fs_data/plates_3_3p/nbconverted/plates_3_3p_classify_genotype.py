#!/usr/bin/env python
# coding: utf-8

# # Train logistic regressions to classify genotypes for plates 3 and 3 prime

# ## Imports

# In[1]:


import sys
import warnings
from pathlib import Path

import pandas as pd
from joblib import dump
from scipy.stats import uniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import parallel_backend

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


# ## Import processing utils

# In[3]:


sys.path.append(f"{root_dir}/1.train_models/log_reg_plates_cp_fs_data/utils")
import log_reg_plates_cp_fs_data_process_split_util as process_split

# ## Define and create paths

# In[4]:


# Input paths
filename3 = "Plate_3_sc_norm_fs.parquet"
filename3p = "Plate_3_prime_sc_norm_fs.parquet"
plate_path = Path(
    f"{root_dir}/nf1_painting_repo/3.processing_features/data/feature_selected_data"
)

path3 = plate_path / filename3
path3p = plate_path / filename3p

# Output paths
models_path = Path("trained_models")
data_path = Path("model_data")

# Create output paths if nonexistent
models_path.mkdir(parents=True, exist_ok=True)
data_path.mkdir(parents=True, exist_ok=True)


# ## Generate plate dataframes

# In[5]:


# Returns the dataframe returned by the plate 3 parquet file
plate3df = pd.read_parquet(path3)

# Returns the dataframe returned by the plate 3 prime parquet file
plate3pdf = pd.read_parquet(path3p)


# ## Define column names

# In[6]:


well_column = "Metadata_Well"
gene_column = "Metadata_genotype"


# # Preprocess Data

# ## Use only common columns

# In[7]:


# Set plate column:
plate3df["Metadata_plate"] = "3"
plate3pdf["Metadata_plate"] = "3p"

common_columns = plate3df.columns.intersection(plate3pdf.columns).to_list()
plate3df = plate3df.loc[:, common_columns]
plate3pdf = plate3pdf.loc[:, common_columns]

# Combine the plate dataframes:
platedf = pd.concat([plate3df, plate3pdf], axis="rows")


# # Process data

# In[8]:


# Use only WT and Null Genotypes
platedf = platedf.loc[platedf[gene_column] != "HET"]

# Create splits and the label encoder
X_train, X_test, y_train, y_test, X_train_shuf, le = process_split.process_splits(platedf, _test_frac=0.2, _well_column=well_column, _gene_column=gene_column)


# # Train Models

# ## Define model variables

# In[9]:


# Define the hyperparameter search space
param_dist = {
    'C': uniform(loc=0.5, scale=1.5),
    "l1_ratio": uniform(loc=0, scale=1),
}

# Create a Logistic Regression model
logreg_params = {
    "max_iter": 1000,
    "multi_class": "ovr",
    "penalty": 'elasticnet',
    "solver": 'saga',
    "n_jobs": -1,
    "random_state": 0,
    "l1_ratio": 0.5
}

# Initialize the RandomizedSearchCV
random_search_params = {
    "param_distributions": param_dist,
    "scoring": "precision",
    "n_iter": 100,
    "cv": 10,
    "random_state": 0,
    "n_jobs": -1,
}


# ## Train the unshuffled model

# In[10]:


logreg = LogisticRegression(**logreg_params)

# Initialize the RandomizedSearchCV
random_search = RandomizedSearchCV(logreg, **random_search_params)

# Prevent the convergence warning in sklearn
with parallel_backend("multiprocessing"):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ConvergenceWarning, module="sklearn"
        )
        # Perform the random hyperparameter search
        random_search.fit(X_train, y_train)


# ## Train the shuffled model

# In[11]:


# Create a Logistic Regression model for shuffled data
shuf_logreg = LogisticRegression(**logreg_params)

# Initialize the RandomizedSearchCV for shuffled data
shuf_random_search = RandomizedSearchCV(shuf_logreg, **random_search_params)

# Prevent the convergence warning in sklearn
with parallel_backend("multiprocessing"):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ConvergenceWarning, module="sklearn"
        )
        # Perform the random hyperparameter search
        shuf_random_search.fit(X_train_shuf, y_train)


# # Save models and model data

# In[12]:


data_suffix = "log_reg_cp_fs_data_plate_3_3p"

# Save the models
dump(random_search.best_estimator_, f"{models_path}/{data_suffix}.joblib")
dump(shuf_random_search.best_estimator_, f"{models_path}/log_reg_shuf_cp_fs_data_plate_3_3P.joblib")

# Save label encoder
dump(le, f"{data_path}/label_encoder_{data_suffix}.joblib")

# Save the feature names
dump(random_search.feature_names_in_, f"{data_path}/feature_names_{data_suffix}.joblib")


# ## Save data indices

# In[13]:


# Assign indices and name each of the splits
X_train_shuf_idxdf = pd.DataFrame({"Metadata_split_name": "X_train_shuf"}, index=X_train_shuf.index)
X_train_idxdf = pd.DataFrame({"Metadata_split_name": "X_train"}, index=X_train.index)
y_train_idxdf = pd.DataFrame({"Metadata_split_name": "y_train"}, index=y_train.index)
y_test_idxdf = pd.DataFrame({"Metadata_split_name": "y_test"}, index=y_test.index)
X_test_idxdf = pd.DataFrame({"Metadata_split_name": "X_test"}, index=X_test.index)

# Concatenate the splits
data_split_indices = pd.concat([X_train_shuf_idxdf, X_train_idxdf, y_train_idxdf, y_test_idxdf, X_test_idxdf])

# Save the splits to a tsv file
data_split_indices.to_csv(f"{data_path}/data_split_indices_{data_suffix}.tsv", sep="\t")

