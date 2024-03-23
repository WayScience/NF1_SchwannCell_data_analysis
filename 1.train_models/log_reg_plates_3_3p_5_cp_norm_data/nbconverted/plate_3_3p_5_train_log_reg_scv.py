#!/usr/bin/env python
# coding: utf-8

# # Random Search with logistic regression (Genotype Classification)
# We perform a random search using logistic regression to improve the classification performance on plates 3, 3 prime, and 5.

# In[1]:


import pathlib
import random
import sys
import warnings
from collections import defaultdict

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


plate5df_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_5_sc_normalized.parquet").resolve(strict=True)
plate3df_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_3_sc_normalized.parquet").resolve(strict=True)
plate3pdf_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_3_prime_sc_normalized.parquet").resolve(strict=True)

plate5df = pd.read_parquet(plate5df_path)
plate3df = pd.read_parquet(plate3df_path)
plate3pdf = pd.read_parquet(plate3pdf_path)

sys.path.append(f"{root_dir}/1.train_models/utils")
from WellSubsetSum import WellSubsetSum

# ### Outputs

# In[4]:


models_path = pathlib.Path("models")
models_path.mkdir(parents=True, exist_ok=True)

data_path = pathlib.Path("data")
data_path.mkdir(parents=True, exist_ok=True)


# ## Splitting and Processing
# Functions to split and process data

# In[5]:


gene_column = "Metadata_genotype"
meta_cols = plate5df.filter(like="Metadata").columns

def down_sample_by_genotype(_df):
    """
    Parameters
    ----------
    _df: Pandas Dataframe
        The data to be downsampled by the gene_column column.

    Returns
    -------
        The data down-sampled by genotype.
    """

    min_gene = _df[gene_column].value_counts().min()
    return (_df.groupby(gene_column, group_keys=False)
            .apply(lambda x: x.sample(n=min_gene, random_state=0))
            )

def split_plates(_df, _num_test_wells):
    """
    Parameters
    ----------
    _df: Pandas Dataframe
       Cleaned single-cell plate data after removing nans and other data not included in the data splits.

    _num_test_wells: Integer
        The number of test wells to be used by the class determined to be the minority class according to the train and validation datasets.

    Returns
    -------
    _restdf: Pandas Dataframe
        The train and validation datasets.

    _testdf: Pandas Dataframe
        The test dataset which contains cells from different wells other than cells in _restdf.
    """

    _welldf = (
        _df.groupby(["Metadata_genotype", "Metadata_Well"])
        .size().reset_index(name="Metadata_cell_count")
    )

    _pkwargs = {
        "_welldf": _welldf,
        "_category_col": "Metadata_genotype",
        "_well_col": "Metadata_Well",
        "_cell_count_col": "Metadata_cell_count",
        "_test_well_count": _num_test_wells
    }

    _gss = WellSubsetSum()
    _wells = _gss.update_test_wells(**_pkwargs)

    _restdf = _df.loc[~_df["Metadata_Well"].isin(_wells)]
    _testdf = _df.loc[_df["Metadata_Well"].isin(_wells)]

    return _restdf, _testdf

def process_plates(_df):
    """
    Parameters
    ----------
    _df: Pandas Dataframe
        Uncleaned plate data with nans and HET cells to be removed.

    Returns
    -------
    _df: Pandas Dataframe
        Cleaned plated data with nans and HET cells removed.
    """

    _df.dropna(inplace=True)
    _df = _df.loc[_df[gene_column] != "HET"]
    return _df


# ## Split and process plates
# We aim to maximize the the number of cells in the train-validation set per plate.
# We achieve this by selecting specific holdout wells that maximize the minority class in the train-validation set.
# In other words, we choose the combination of wells for train-validation that, together, include the highest number of cells in the genotype category which has the fewest number of cells.
# By side-effect, this process also minimizes the number of cells dropped from training in our downsampling procedure to balance datasets for class size prior to model training.

# In[6]:


plate5df = process_plates(plate5df)
rest5df, test5df = split_plates(plate5df, 4)
rest5df, test5df = down_sample_by_genotype(rest5df), down_sample_by_genotype(test5df)
num_test = test5df.shape[0]
print(f"Fraction of test cells plate 5 = {num_test / (num_test + rest5df.shape[0])}\n")

plate3df = process_plates(plate3df)
rest3df, test3df = split_plates(plate3df, 7)
rest3df, test3df = down_sample_by_genotype(rest3df), down_sample_by_genotype(test3df)
num_test = test3df.shape[0]
print(f"Fraction of test cells plate 3 = {num_test / (num_test + rest3df.shape[0])}\n")

plate3pdf["Metadata_Plate"] = "Plate_3p"
plate3pdf = process_plates(plate3pdf)
rest3pdf, test3pdf = split_plates(plate3pdf, 5)
rest3pdf, test3pdf = down_sample_by_genotype(rest3pdf), down_sample_by_genotype(test3pdf)
num_test = test3pdf.shape[0]
print(f"Fraction of test cells plate 3 prime = {num_test / (num_test + rest3pdf.shape[0])}\n")


# ## Harmonize data across plates to each data split

# In[7]:


# Columns common to all plates
plate_cols = list(set(plate5df.columns) & set(plate3df.columns) & set(plate3pdf.columns))

restdf = pd.concat([rest5df[plate_cols], rest3df[plate_cols], rest3pdf[plate_cols]], ignore_index=True)

testdf = pd.concat([test5df[plate_cols], test3df[plate_cols], test3pdf[plate_cols]], ignore_index=True)


# ## Encode genotypes and extract feature data

# In[8]:


le = LabelEncoder()

y = le.fit_transform(restdf["Metadata_genotype"])
X = restdf.drop(columns=meta_cols)

y_test = le.fit_transform(restdf["Metadata_genotype"])
X_test = restdf.drop(columns=meta_cols)


# # Train Models

# ## Specify parameters for training

# In[9]:


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

# In[10]:


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
        acc += accuracy_score(y_val, preds)

        # Store model data for folds
        eval_data["plate"].extend(restdf.iloc[val_index]["Metadata_Plate"].tolist())
        eval_data["predicted_probability"].extend(logreg.predict_proba(X_val).tolist())
        eval_data["datasplit"].extend(["val"] * val_index.shape[0])
        eval_data["predicted_genotype"].extend(preds.tolist())
        eval_data["true_genotype"].extend(y_val.tolist())

    # Average accuracy for the folds
    acc = acc / n_splits

    # Store the data with the best performance
    if acc > best_acc:
        best_hparam = eval_data.copy()
        best_acc = acc
        best_hp = rparams

print(f"Best average validation accuracy = {best_acc}")


# ## Retrain model

# In[11]:


logreg_params = {
    "max_iter": 3000,
    "random_state": 0,
    "n_jobs": -1,
    "penalty": "l2",
}

comb_params = logreg_params | best_hp

logreg = LogisticRegression(**comb_params)
logreg.fit(X, y)


# ## Store training and testing data

# In[12]:


eval_data["plate"].extend(restdf["Metadata_Plate"].tolist())
eval_data["predicted_probability"].extend(logreg.predict_proba(X).tolist())
eval_data["datasplit"].extend(["train"] * X.shape[0])
eval_data["predicted_genotype"].extend(logreg.predict(X).tolist())
eval_data["true_genotype"].extend(y.tolist())

eval_data["plate"].extend(restdf["Metadata_Plate"].tolist())
eval_data["predicted_probability"].extend(logreg.predict_proba(X_test).tolist())
eval_data["datasplit"].extend(["test"] * X_test.shape[0])
eval_data["predicted_genotype"].extend(logreg.predict(X_test).tolist())
eval_data["true_genotype"].extend(y_test.tolist())


# # Save models and model data

# ## Save model

# In[13]:


data_suf = "log_reg_cp_fs_data_plate_5"

# Save the models
dump(logreg, f"{models_path}/{data_suf}.joblib")

# Save label encoder
dump(le, f"{data_path}/label_encoder_{data_suf}.joblib")


# ## Save data folds

# In[14]:


pd.DataFrame(eval_data).to_parquet(f"{data_path}/model_data_{data_suf}.parquet")

