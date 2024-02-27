#!/usr/bin/env python
# coding: utf-8

# # Random Search with xgboost (Genotype Classification)
# We perform a random search using xgboost to improve the classification performance on plate 5.

# In[1]:


import pathlib
import random
import sys
import time

import pandas as pd
import xgboost as xgb
from joblib import load
from sklearn.preprocessing import LabelEncoder

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


# Load normalized qc data
platedf_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_5_sc_normalized_qc.parquet").resolve(strict=True)

platedf = pd.read_parquet(platedf_path)

# Access spliting utilities
sys.path.append(f"{root_dir}/1.train_models/utils")
feat_cols = load("selected_features_1_qc_prefs.joblib")
from GreedySubsetSum import GreedySubsetSum

# ### Output paths

# In[4]:


models_path = pathlib.Path("models")
models_path.mkdir(parents=True, exist_ok=True)

data_path = pathlib.Path("data")
data_path.mkdir(parents=True, exist_ok=True)


# # Define Functions

# In[5]:


def filter_features(X_data):
    """
    Parameters
    ----------
    X_data: pandas Dataframe
        Contains the feature data to be filtered.

    Returns
    -------
    X_data: pandas Dataframe
        The features filtered from the data.
    """

    X_data.drop(columns=X_data.filter(like="ZernikePhase").columns, inplace=True)
    X_data.drop(columns=X_data.filter(like="ZernikeMagnitude").columns, inplace=True)
    X_data.drop(columns=X_data.filter(like="FracAtD").columns, inplace=True)

    return X_data


# In[6]:


def down_sample_by_genotype(df):
    """
    Parameters
    ----------
    df: pandas Dataframe
        Cell data to be down-sampled by genotype.

    Returns
    -------
    pandas Dataframe
        Cell data down-sampled by genotype.
    """

    min_gene = df[gene_column].value_counts().min()
    return (df.groupby(gene_column, group_keys=False)
            .apply(lambda x: x.sample(n=min_gene, random_state=0))
            )


# # Sample Cells

# ## Remove nan Cells

# In[7]:


platedf.dropna(inplace=True)


# ## Genotype and Well groups

# In[8]:


gene_column = "Metadata_genotype"
meta_cols = platedf.filter(like="Metadata").columns

welldf = (
    platedf.groupby(["Metadata_genotype", "Metadata_Well"])
    .size().reset_index(name="Metadata_cell_count")
)


# ## Split Data

# ### Seperate the test (holdout) cell data
# The number of cells in the train-validation set is maximized by minimizing the number of cells sampled from the adjusted minority class for a desired number of holdout wells.
# This also adjusts the number of cells in the holdout wells for the classes other than the minority class to minimize the number of cells that would otherwise not be included in any of the datasets.

# In[9]:


kwargs = {
    "_welldf": welldf,
    "_category_col": "Metadata_genotype",
    "_well_col": "Metadata_Well",
    "_cell_count_col": "Metadata_cell_count",
    "_test_well_count": 3
}


# In[10]:


gss = GreedySubsetSum()
wells = gss.update_test_wells(**kwargs)


# In[11]:


restdf = platedf.loc[~platedf["Metadata_Well"].isin(wells)]
testdf = platedf.loc[platedf["Metadata_Well"].isin(wells)]


# ### Down-sample by genotype

# In[12]:


restdf = down_sample_by_genotype(restdf)
testdf = down_sample_by_genotype(testdf)


# # Process data

# ## Encode labels

# In[13]:


le = LabelEncoder()
y = le.fit_transform(restdf["Metadata_genotype"])


# ## Filter columns

# In[14]:


X = restdf.drop(columns=meta_cols)
X = X[feat_cols]
X = filter_features(X)


# ## XGboost-compatible data structure

# In[15]:


dX = xgb.DMatrix(X, label=y, nthread=-1)


# # Train Models

# ## Define parameters

# In[16]:


param_ranges = {
    'learning_rate': (0.01, 0.51),
    'subsample': (0.8, 1),
    'colsample_bytree': (0.8, 1),
    'gamma': (0, 0.4),
    'reg_lambda': (0, 20),
}


# In[17]:


xgb_params = {
    "n_estimators": 5000,
    "n_jobs": -1,
    "random_state": 0,
    "device": "cuda",
    "verbosity": 0,
    "importance_type": "total_gain",
    "validate_parameters": True,
    "objective": "multi:softmax",
    "num_class": 3
}


# ## Randomly sample training hyperparameters

# In[18]:


rand_iter = 1400
random_params = {
    i:
    {key: random.uniform(*param_ranges[key]) for key in param_ranges}
    for i in range(rand_iter)
}


# ## Random Search

# In[19]:


for idx, rparams in random_params.items():

    start_time = time.time()
    comb_params = xgb_params | rparams
    xgb_cv = xgb.cv(
        params=comb_params,
        dtrain=dX,
        num_boost_round=100,
        early_stopping_rounds=10,
        nfold=5,
        stratified=True,
        seed=0,
        metrics="merror",
        verbose_eval=True
    )
    end_time = time.time()

    min_merror_idx = xgb_cv["test-merror-mean"].argmin()

    random_params[idx] = {
        "min_test_merror_mean": xgb_cv["test-merror-mean"].iloc[min_merror_idx],
        "min_test_merror_std": xgb_cv["test-merror-std"].iloc[min_merror_idx],
        "min_test_merror_mean_idx": xgb_cv["test-merror-mean"].argmin()
    }

    print(f"Time to run iteration [[{idx}]] = ({(end_time - start_time) / 60:.2f} minutes)")


# # Save model data

# ## Format random search results

# In[20]:


rsrdf = pd.DataFrame.from_dict(random_params, orient='index')

rsrdf.rename_axis("parameter_search_idx", inplace=True)

rsrdf["max_test_accuracy_mean"] = 1 - rsrdf["min_test_merror_mean"]


# ## Save random search results

# In[21]:


rsrdf.to_csv(f"{data_path}/random_search_xgb_opt_results.tsv", sep="\t")


# ## Save split indices

# In[22]:


# Assign indices and name each of the splits
rest_idxdf = pd.DataFrame({"Metadata_split_name": "rest"}, index=restdf.index)
test_idxdf = pd.DataFrame({"Metadata_split_name": "test"}, index=testdf.index)

# Concatenate the splits
data_split_indices = pd.concat([rest_idxdf, test_idxdf])

# Save the splits to a tsv file
data_split_indices.to_csv(f"{data_path}/data_split_indices_xgb_opt.tsv", sep="\t")

