#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pandas as pd
import numpy as np

from joblib import load

# Set the seed
rng = np.random.default_rng(0)


# In[2]:


def shuffle_data(df):
    """
    Shuffle the feature columns of the input dataframe independently while keeping metadata columns unchanged.
    Columns with 'Metadata' prefix are considered metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing both features and metadata.
    """
    feature_columns = [col for col in df.columns if not col.startswith('Metadata')]

    shuffled_df = df.copy()
    
    for column in feature_columns:
        shuffled_df[column] = rng.permutation(shuffled_df[column])
    
    return shuffled_df


# In[3]:


# Path to encoder
le_path = pathlib.Path("./data/trained_nf1_model_label_encoder.joblib")

# Path to model
model_path = pathlib.Path("./data/trained_nf1_model.joblib")

# Load in encoder
le = load(le_path)

# Load in NF1 model
model = load(model_path)


# In[4]:


# Load in the model data
model_df = pd.read_parquet(pathlib.Path("./model_data.parquet"))

meta_cols = model_df.filter(like="Metadata").columns

print(model_df.shape)
model_df.head()


# In[5]:


# Decode the true genotypes using the encoder
true_genotypes = le.fit_transform(model_df["Metadata_genotype"])

probabilitydf = pd.DataFrame(
    {
        f"probability_{le.inverse_transform([1])[0]}": model.predict_proba(model_df[model.feature_names_in_])[:, 1],
        "predicted_genotype": model.predict(model_df[model.feature_names_in_]),
        "true_genotype": true_genotypes
    }
)

probabilitydf = pd.concat([probabilitydf, model_df[meta_cols].reset_index(drop=True)], axis=1)

# Rename 'Metadata_datasplit' to 'datasplit'
probabilitydf.rename(columns={'Metadata_datasplit': 'datasplit'}, inplace=True)

# Move 'datasplit' to the start of the DataFrame
cols = ['datasplit'] + [col for col in probabilitydf if col != 'datasplit']
probabilitydf = probabilitydf[cols]

print(probabilitydf.shape)
probabilitydf.head()


# In[6]:


probabilitydf["Metadata_Plate"].unique()


# ## Shuffle the data and apply model

# In[7]:


model_shuffled_df = shuffle_data(model_df)

meta_cols = model_shuffled_df.filter(like="Metadata").columns

model_shuffled_df['Metadata_datasplit'] = 'shuffled_' + model_shuffled_df['Metadata_datasplit'].astype(str)

print(model_shuffled_df.shape)
model_shuffled_df.head()


# In[8]:


# Decode the true genotypes using the encoder
true_genotypes = le.fit_transform(model_shuffled_df["Metadata_genotype"])

shuffled_probabilitydf = pd.DataFrame(
    {
        f"probability_{le.inverse_transform([1])[0]}": model.predict_proba(model_shuffled_df[model.feature_names_in_])[:, 1],
        "predicted_genotype": model.predict(model_shuffled_df[model.feature_names_in_]),
        "true_genotype": true_genotypes
    }
)

shuffled_probabilitydf = pd.concat([shuffled_probabilitydf, model_shuffled_df[meta_cols].reset_index(drop=True)], axis=1)

# Rename 'Metadata_datasplit' to 'datasplit'
shuffled_probabilitydf.rename(columns={'Metadata_datasplit': 'datasplit'}, inplace=True)

# Move 'datasplit' to the start of the DataFrame
cols = ['datasplit'] + [col for col in shuffled_probabilitydf if col != 'datasplit']
shuffled_probabilitydf = shuffled_probabilitydf[cols]

print(shuffled_probabilitydf.shape)
shuffled_probabilitydf.head()


# In[9]:


# Concatenate the DataFrames vertically and save to a Parquet file
combined_df = pd.concat([shuffled_probabilitydf, probabilitydf], axis=0).reset_index(drop=True)
combined_df.to_parquet(f"./data/nf1_eval_data.parquet")

combined_df

