#!/usr/bin/env python
# coding: utf-8

# # Training a Logistic Regression Model

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump, load


# ## Find the git root Directory

# In[ ]:


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


# # Seed and Generator for Reproducibility

# In[ ]:


rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator


# # Converting csv to pandas dataframe

# In[ ]:


filename3 = "Plate_3_sc_norm_fs.parquet"
filename3p = "Plate_3_prime_sc_norm_fs.parquet"
plate_path = Path(
    f"{root_dir}/nf1_painting_repo/3.processing_features/data/feature_selected_data"
)

data_path = Path("data")

data_path.mkdir(
    parents=True, exist_ok=True
)  # Create the parent directories if they don't exist

path3 = plate_path / filename3

path3p = plate_path / filename3p


# ## Generate plate dataframes

# In[ ]:


# Returns the dataframe returned by the plate 3 parquet file
plate3df = pd.read_parquet(path3)

# Returns the dataframe returned by the plate 3 prime parquet file
plate3pdf = pd.read_parquet(path3p)


# # Preprocess Data

# ## Use only common columns

# In[ ]:


# Set plate column:
plate3df["Metadata_plate"] = "3"
plate3pdf["Metadata_plate"] = "3p"

common_columns = list(plate3df.columns.intersection(plate3pdf.columns))
plate3df = plate3df.loc[:, common_columns]
plate3pdf = plate3pdf.loc[:, common_columns]

# Combine the plate dataframes:
platedf = pd.concat([plate3df, plate3pdf], axis="rows")


# ## Create Classes

# In[ ]:


target_column = "Metadata_genotype"
stratify_column = "Metadata_Well"

# These represent the fractions of the entire dataset
train_val_frac = 0.85
test_frac = 1 - train_val_frac
val_frac = 0.15


# ## Down-sample and stratify by well

# In[ ]:


smallest_gene = platedf[target_column].value_counts().min()
platedata = pd.DataFrame()

for gene in platedf[target_column].unique():
    df = platedf.loc[platedf["Metadata_genotype"] == gene]
    df_frac = smallest_gene / len(df)
    stratwell = df.groupby(stratify_column, group_keys=False).apply(
        lambda x: x.sample(frac=df_frac, random_state=rnd_val)
    )
    platedata = pd.concat([platedata, stratwell], axis="rows")


# ## Stratified Train-test split

# In[ ]:


traindf, testdf = train_test_split(
    platedata,
    train_size=train_val_frac,
    stratify=platedata[[target_column, stratify_column]],
    shuffle=True,
    random_state=rnd_val,
)


# ## Encode Labels

# In[ ]:


le = LabelEncoder()
testdf["label"] = le.fit_transform(testdf[target_column].values)
traindf["label"] = le.transform(traindf[target_column].values)


# ## Remove unecessary columns

# In[ ]:


# Remove Metadata
feat_col = [col for col in platedata.columns if "Metadata" not in col]

# Keep the label column
feat_col.append("label")

traindf = traindf[feat_col]
testdf = testdf[feat_col]


# # Model Training

# In[ ]:


lr = LogisticRegression(
    max_iter=1000, solver="sag", multi_class="ovr", random_state=rnd_val, n_jobs=-1
)
lr.fit(X=traindf.drop("label", axis="columns"), y=traindf["label"])


# ## Save Data

# In[ ]:


dump(lr, data_path / "lr_model.joblib")
dump(testdf, data_path / "testdf.joblib")
dump(le, data_path / "label_encoder.joblib")
