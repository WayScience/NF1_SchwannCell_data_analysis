#!/usr/bin/env python
# coding: utf-8

# # Train logistic regressions to classify genotypes for plate 4

# In[1]:


import pathlib
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import uniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import parallel_backend

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


# ## Define and create paths

# In[3]:


# Input paths
plate_path = "Plate_4_sc_norm_fs.parquet"
plate_path = pathlib.Path(f"{root_dir}/nf1_painting_repo/3.processing_features/data/feature_selected_data/{plate_path}")

# Output paths
models_path = pathlib.Path("trained_models")
data_path = pathlib.Path("model_data")

# Create output paths if nonexistent
models_path.mkdir(parents=True, exist_ok=True)
data_path.mkdir(parents=True, exist_ok=True)


# ## Load the dataframe

# In[4]:


platedf = pd.read_parquet(plate_path)


# ## Define column names

# In[5]:


well_column = "Metadata_Well"
gene_column = "Metadata_genotype"
concentration_column = "Metadata_Concentration"


# # Process data

# ## Down-sample by genotype

# In[6]:


# Filter data where the siRNA construct concentration is zero
platedf = platedf[platedf[concentration_column] == 0]

min_gene = platedf[gene_column].value_counts().min()
platedf = platedf.groupby(gene_column, group_keys=False).apply(lambda x: x.sample(n=min_gene, random_state=0))


# ## Encode genotypes

# In[7]:


# Encode classes
le = LabelEncoder()
featdf = platedf.copy()
featdf[gene_column] = le.fit_transform(featdf[gene_column])


# ## Remove Metadata and encode labels

# In[8]:


# Create dataframe without Metadata
feat_columns = [col for col in platedf.columns if "Metadata" not in col]
featdf = platedf[feat_columns]

# Encode labels
platedf[gene_column] = le.transform(platedf[gene_column])


# ## Split the Data

# In[9]:


test_frac = 0.2
val_frac = 0.15

X_train, X_test, y_train, y_test = train_test_split(featdf, platedf[gene_column], test_size = test_frac, random_state=0, shuffle=True, stratify=platedf[gene_column])


# ## Shuffle the training data for the shuffled model

# In[10]:


X_train_shuf = X_train.copy()

rng = np.random.default_rng(0)

for column in X_train_shuf.columns:
    X_train_shuf[column] = rng.permutation(X_train_shuf[column])


# # Train Models

# ## Define model variables

# In[11]:


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
    "solver": 'saga'
}

# Initialize the RandomizedSearchCV
random_search_params = {
    "param_distributions": param_dist,
    "scoring": "precision",
    "n_iter": 100,
    "cv": 5,
    "random_state": 0,
    "n_jobs": -1,
}


# ## Train the unshuffled model

# In[12]:


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

# In[13]:


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

# In[14]:


data_suf = "log_reg_cp_fs_data_plate_4"

# Save the models
dump(random_search.best_estimator_, f"{models_path}/{data_suf}.joblib")
dump(shuf_random_search.best_estimator_, f"{models_path}/log_reg_shuf_cp_fs_data_plate_4.joblib")

# Save label encoder
dump(le, f"{data_path}/label_encoder_{data_suf}.joblib")


# ## Save data indices

# In[15]:


# Assign indices and name each of the splits
X_train_shuf_idxdf = pd.DataFrame({"Metadata_split_name": "X_train_shuf"}, index=X_train_shuf.index)
X_train_idxdf = pd.DataFrame({"Metadata_split_name": "X_train"}, index=X_train.index)
y_train_idxdf = pd.DataFrame({"Metadata_split_name": "y_train"}, index=y_train.index)
y_test_idxdf = pd.DataFrame({"Metadata_split_name": "y_test"}, index=y_test.index)
X_test_idxdf = pd.DataFrame({"Metadata_split_name": "X_test"}, index=X_test.index)

# Concatenate the splits
data_split_indices = pd.concat([X_train_shuf_idxdf, X_train_idxdf, y_train_idxdf, y_test_idxdf, X_test_idxdf])

# Save the splits to a tsv file
data_split_indices.to_csv(f"{data_path}/data_split_indices_{data_suf}.tsv", sep="\t")

