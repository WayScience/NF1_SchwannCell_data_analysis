#!/usr/bin/env python
# coding: utf-8

# # Determine the best features using a Logistic Regression Model

# ## Imports

# In[ ]:


import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
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
import itertools

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


# ## Import Utilities

# In[ ]:


sys.path.append(f"{root_dir}/utils")
import preprocess_utils as ppu


# # Seed and Generator for Reproducibility

# In[ ]:


rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator


# In[ ]:


fig_out_path = Path("feature_importance_figures")
data_path = Path("data")

if not fig_out_path.exists():
    fig_out_path.mkdir()

if not data_path.exists():
    data_path.mkdir()


# ## Load Model

# In[ ]:


models_path = Path(f"{root_dir}/1.train_models/lr_plate3_cp_fs_data/data")
lr = load(models_path / "lr_model.joblib")


# ## Save Data

# In[ ]:


testdf = load(models_path / "testdf.joblib")
le = load(models_path / "label_encoder.joblib")


# ## Create Dataframe with coefficients for each Genotype

# In[ ]:


featdf = testdf.reset_index(drop=True)
featdf.drop(["label", "preds"], inplace=True, axis=1)
featdf = featdf.T
featdf = featdf.reset_index()
featdf = featdf.rename(columns={"index": "features"})
featdf = pd.concat(
    [pd.DataFrame(lr.coef_.T, columns=le.classes_.tolist()), featdf], axis="columns"
)


# ## Get the most important features for HET from the entire Test Set

# In[ ]:


featdf["abs_HET"] = featdf["HET"].abs()


# ## Create dictionaries that map labels

# In[ ]:


pos_genes = ["HET", "Null", "WT"]
label2gene = dict(zip(le.transform(pos_genes), pos_genes))
gene2label = {gene: label for label, gene in label2gene.items()}


# ## Create a DataFrame with all correctly predicted data

# In[ ]:


correct_filt = (
    ((testdf["preds"] == 0) & (testdf["label"] == 0))
    | ((testdf["preds"] == 1) & (testdf["label"] == 1))
    | ((testdf["preds"] == 2) & (testdf["label"] == 2))
)
correctdf = testdf.loc[correct_filt]
correctdf = correctdf.reset_index(drop=True)


# ## Find the most important features by average

# In[ ]:


featimp = {}

# Columns to be used as features
kept_cols = correctdf.drop(["label", "preds"], axis="columns").columns

for genotype in featdf[pos_genes]:
    featimp[genotype] = {}  # Create a DataFrame for each genotype
    label = gene2label[genotype]  # Get the label for the genotype
    gene_filt = correctdf["label"] == label  # Filter to get the data for a genotype

    # Product of model weights by the feature values
    mat_imp = (
        featdf[genotype].values
        * correctdf.loc[gene_filt].drop(["label", "preds"], axis="columns").values
    )

    # Create the dataframe of product of model weights by the feature values
    featimp[genotype]["featdf"] = pd.DataFrame(mat_imp, columns=kept_cols.to_list())

    # Calculate quartiles and IQR:
    q1 = featimp[genotype]["featdf"].quantile(0.25)
    q3 = featimp[genotype]["featdf"].quantile(0.75)
    iqr = q3 - q1

    # Calculate bounds:
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers from bounds:
    featimp[genotype]["featdf"] = featimp[genotype]["featdf"][
        (featimp[genotype]["featdf"] >= lower_bound)
        & (featimp[genotype]["featdf"] <= upper_bound)
    ]

    # Select inliers
    featimp[genotype]["featdf"] = featimp[genotype]["featdf"][
        featimp[genotype]["featdf"] > 0
    ]

    # Calculate the mean for each (cell feature value / feature coefficient) product
    featimp[genotype]["featnorm_avg"] = featimp[genotype]["featdf"].mean()

    # Drop the means that are not a number
    featimp[genotype]["featnorm_avg"].dropna(inplace=True)

    # Total sum normalize w.r.t. all feature means
    featimp[genotype]["featnorm_avg_norm"] = (
        featimp[genotype]["featnorm_avg"] / featimp[genotype]["featnorm_avg"].sum()
    )

    # Sort the normalized averaged features
    featimp[genotype]["featnorm_avg_norm"] = featimp[genotype][
        "featnorm_avg_norm"
    ].sort_values(ascending=False)

    # Sort the averaged features
    featimp[genotype]["featnorm_avg"] = featimp[genotype]["featnorm_avg"].sort_values(
        ascending=False
    )


# ## Create a dataframe of averaged importantces across Genotype

# In[ ]:


totfeatimp = {}

# Extract the common columns
common_indices = (
    featimp["HET"]["featnorm_avg_norm"]
    .index.intersection(featimp["Null"]["featnorm_avg_norm"].index)
    .intersection(featimp["WT"]["featnorm_avg_norm"].index)
)

# Create a dataframe from the common columns accross genotype
avgfeatimpdf = pd.DataFrame(common_indices.tolist(), columns=["Features"])

featimpdf = pd.DataFrame([])

for genotype in pos_genes:
    # Create a dataframe for each genotype's feature importances
    avgfeatimpdf = pd.merge(
        avgfeatimpdf,
        pd.DataFrame(
            list(featimp[genotype]["featnorm_avg_norm"].items()),
            columns=["Features", genotype],
        ),
        on="Features",
        how="inner",
    )

    # Ensure we can distinguish between genotype
    featimp[genotype]["featdf"]["genotype"] = [genotype] * len(
        featimp[genotype]["featdf"]
    )

    # Combine all genotype dataframes
    featimpdf = pd.concat([featimpdf, featimp[genotype]["featdf"]], axis=0)

# Calculate the Overal feature importance by averaging each feature's importance across genotypes
avgfeatimpdf["Overall"] = (
    avgfeatimpdf[pos_genes[0]] + avgfeatimpdf[pos_genes[1]] + avgfeatimpdf[pos_genes[2]]
)
avgfeatimpdf["Overall"] = avgfeatimpdf["Overall"] / 3


# ## Convert Overall Importances to sorted normalized series

# In[ ]:


totfeatimp = pd.Series(avgfeatimpdf["Overall"].values, index=avgfeatimpdf["Features"])

min_val = totfeatimp.min()
max_val = totfeatimp.max()
totfeatimp = totfeatimp / totfeatimp.sum()

totfeatimp = totfeatimp.sort_values(ascending=False)


# ## Save the most important feature values

# In[ ]:


totfeatimp.to_csv(
    data_path / "overall_feature_importances.tsv",
    sep="\t",
    header=["feature_importance"],
    index=True,
)


# ## Save the Feature Weights

# In[ ]:


featdf.to_csv(data_path / "feature_weights.tsv", sep="\t", index=False)


# ## Save the Averaged Feature Importances

# In[ ]:


avgfeatimpdf.to_csv(
    data_path / "avg_norm_feature_importances.tsv", sep="\t", index=False
)


# ## Save the correct cell data

# In[ ]:


correctdf.to_csv(data_path / "correctly_predicted_cells.tsv", sep="\t", index=False)
