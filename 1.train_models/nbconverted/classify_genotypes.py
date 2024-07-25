#!/usr/bin/env python
# coding: utf-8

# # Classify WT and Null Genotypes Logistic Regression
# Plates 3, 3p, and 5 are used in all splits to classify genotypes either (WT or Null)
# The feature selected data is used in all data splits.
# Pre-evaluation metrics are stored from all splits and these plates.

# In[1]:


import pathlib
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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


# ## Define paths

# ### Input

# In[3]:


plate3df_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_3_sc_feature_selected.parquet").resolve(strict=True)
plate3pdf_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_3_prime_sc_feature_selected.parquet").resolve(strict=True)
plate5df_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_5_sc_feature_selected.parquet").resolve(strict=True)

plate3df = pd.read_parquet(plate3df_path)
plate3pdf = pd.read_parquet(plate3pdf_path)
plate5df = pd.read_parquet(plate5df_path)

# Set the seed
rng = np.random.default_rng(0)


# ### Outputs

# In[4]:


data_path = pathlib.Path("data")
data_path.mkdir(parents=True, exist_ok=True)


# ## Splitting and Processing
# Functions to split and process data

# In[5]:


gene_column = "Metadata_genotype"

def down_sample_by_genotype(_df):
    """
    Return an equal number of cells from each genotype.
    The number of cells in a genotype is the minimum number of cells from all genotypes.

    Parameters
    ----------
    _df: Pandas Dataframe
        The data to be downsampled by the gene_column column.

    Returns
    -------
    The dataframe down-sampled by genotype.
    """

    min_gene = _df[gene_column].value_counts().min()
    return (_df.groupby(gene_column, group_keys=False)
            .apply(lambda x: x.sample(n=min_gene, random_state=0))
            )

def process_plates(_df):
    """
    Drop rows with nans from the single cell data and remove HET cells.

    Parameters
    ----------
    _df: Pandas Dataframe
        Uncleaned plate data with nans and HET cells to be removed. Contains the column "Metadata_genotype".

    Returns
    -------
    _df: Pandas Dataframe
        Cleaned single cell data by removing nans and HET cells.
    """

    _df.dropna(inplace=True)
    _df = _df.loc[_df[gene_column] != "HET"]
    return _df

def shuffle_data(_X):
    """
    Shuffle the columns of the input dataframe independently.

    Parameters
    ----------
    _X: Pandas Dataframe
        Input feature data for shuffling the columns.
    """

    for column in _X.columns:
        _X[column] = rng.permutation(_X[column])

def store_pre_evaluation_data(_X, _y, _metadata, _datasplit):
    """
    Store model data to evaluate performance.

    Parameters
    ----------
    _X: Pandas Dataframe
        Feature dataframe from a given plate and data split.

    _y: Numpy Array
        A numerically-encoded label vector ordered according to _X.

    _metadata: Pandas Dataframe
        Plate name.

    _datasplit: String
        Data split name.
    """

    eval_data[f"probability_{probability_class}"].extend(logreg.predict_proba(_X)[:, 1].tolist())
    eval_data["datasplit"].extend([_datasplit] * _X.shape[0])
    eval_data["predicted_genotype"].extend(logreg.predict(_X).tolist())
    eval_data["true_genotype"].extend(_y.tolist())
    for meta_col in _metadata.columns:
        eval_data[meta_col].extend(_metadata[meta_col].tolist())


# ## Split and process plates

# In[6]:


def create_splits(_wells, _plate):
    """
    Create data splits for model training. The splits are rest (train and validation) and test.

    Parameters
    ----------
    _wells: List(String)
        The well names from which single cells will be used in the test set.

    _plate: Pandas Dataframe
        Single cell data from one of the plate's containing a "Metadata_Well" column.

    Returns
    -------
    Dataframes of the split single cell data.
    """

    return _plate[~_plate["Metadata_Well"].isin(_wells)], _plate[_plate["Metadata_Well"].isin(_wells)]


# In[7]:


plate3df = process_plates(plate3df)
p3_wells = ["C11", "E11", "C3", "F3"]
rest3df, test3df = create_splits(p3_wells, plate3df)
rest3df, test3df = down_sample_by_genotype(rest3df), down_sample_by_genotype(test3df)

plate3pdf = process_plates(plate3pdf)
p3p_wells = ["F11", "G11", "C3", "F3"]
rest3pdf, test3pdf = create_splits(p3p_wells, plate3pdf)
rest3pdf, test3pdf = down_sample_by_genotype(rest3pdf), down_sample_by_genotype(test3pdf)

plate5df = process_plates(plate5df)
p5_wells = ["C9", "E11", "E3", "G3"]
rest5df, test5df = create_splits(p5_wells, plate5df)
rest5df, test5df = down_sample_by_genotype(rest5df), down_sample_by_genotype(test5df)


# ## Combine plate columns across each data split

# In[8]:


# Columns common to all plates
plate_cols = list(set(plate5df.columns) & set(plate3df.columns) & set(plate3pdf.columns))

restdf = pd.concat([rest3df[plate_cols], rest3pdf[plate_cols], rest5df[plate_cols]], ignore_index=True).reset_index(drop=True)

testdf = pd.concat([test3df[plate_cols], test3pdf[plate_cols], test5df[plate_cols]], ignore_index=True).reset_index(drop=True)


# ## Encode genotypes and extract feature data

# In[9]:


meta_cols = testdf.filter(like="Metadata").columns
feat_cols = testdf.drop(columns=meta_cols).columns


# In[10]:


le = LabelEncoder()

y = le.fit_transform(restdf["Metadata_genotype"])
X = restdf.drop(columns=meta_cols)

y_test = le.fit_transform(testdf["Metadata_genotype"])
X_test = testdf.drop(columns=meta_cols)

# Class for saving probabilities
probability_class = le.inverse_transform([1])[0]


# # Train Models

# ## Specify parameters for training

# In[11]:


logreg_params = {
    "max_iter": 250,
    "random_state": 0,
    "n_jobs": -1,
    "penalty": "l2",
}

# Random sampling range of hyperparameter
param_ranges = {
    "C": (0, 200)
}

# Number of iteration to optimize hyperparameters
rand_iter = 500

# Best accuracy
best_acc = 0

# Initial accuracy
acc = 0

# Number of folds
n_splits = 8

# Generate hyperparameter samples
random_params = {
    i:
    {key: random.uniform(*param_ranges[key]) for key in param_ranges}
    for i in range(rand_iter)
}


# ## Hyperparameter search

# In[12]:


# Store model results for evaluation
eval_data = defaultdict(list)

# Iterate through hyperparameters
for idx, rparams in random_params.items():

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Combine parameters in current search with logistic regression parameters
    comb_params = logreg_params | rparams

    # Loop through the folds
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_val_shuf = X_val.copy()
        shuffle_data(X_val_shuf)

        # Prevent the convergence warning in sklearn
        with parallel_backend("multiprocessing"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=ConvergenceWarning, module="sklearn"
                )
                logreg = LogisticRegression(**comb_params)
                logreg.fit(X_train, y_train)

        # Cumulative accuracy for all folds
        preds = logreg.predict(X_val)
        preds_shuf = logreg.predict(X_val_shuf)
        acc += accuracy_score(y_val, preds)

        store_pre_evaluation_data(X_val, y_val, restdf.iloc[val_index][meta_cols], "val")
        store_pre_evaluation_data(X_val_shuf, y_val, restdf.iloc[val_index][meta_cols], "shuffled_val")

    # Average accuracy for the folds
    acc = acc / n_splits

    # Store the data with the best performance
    if acc > best_acc:
        best_hparam = eval_data.copy()
        best_acc = acc
        best_hp = rparams

print(f"Best average validation accuracy = {best_acc}")


# ## Retrain model

# In[13]:


logreg_params = {
    "max_iter": 3000,
    "random_state": 0,
    "n_jobs": -1,
    "penalty": "l2",
}

comb_params = logreg_params | best_hp

logreg = LogisticRegression(**comb_params)
logreg.fit(X, y)


# ## Shuffle train and validation data

# In[14]:


X_shuf = X.copy()
shuffle_data(X_shuf)

X_test_shuf = X_test.copy()
shuffle_data(X_test_shuf)


# # Save models and model data

# ## Store pre-evaluation split data

# In[15]:


store_pre_evaluation_data(X, y, restdf[meta_cols], "train")
store_pre_evaluation_data(X_shuf, y, restdf[meta_cols], "shuffled_train")

store_pre_evaluation_data(X_test, y_test, testdf[meta_cols], "test")
store_pre_evaluation_data(X_test_shuf, y_test, testdf[meta_cols], "shuffled_test")


# In[16]:


dump(logreg, f"{data_path}/trained_nf1_model.joblib")
dump(le, f"{data_path}/trained_nf1_model_label_encoder.joblib")
pd.DataFrame(eval_data).to_parquet(f"{data_path}/nf1_model_pre_evaluation_results.parquet")

