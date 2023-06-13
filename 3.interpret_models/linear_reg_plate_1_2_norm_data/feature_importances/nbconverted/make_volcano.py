#!/usr/bin/env python
# coding: utf-8

# # Here we understand the relationship between WT contribution and model significance
#
# Each linear model is regressed to each morphology feature according to covariate data

# In[1]:


#%%--%%| <qGnYViiwRD|SyZ3qa8iz3>
r"""°°°
## Imports
°°°"""


# In[ ]:


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


# In[ ]:


output_path = Path("figures")

output_path.mkdir(
    parents=True, exist_ok=True
)  # Create the parent directories if they don't exist


# ## Import the model data as a dataframe

# In[ ]:


concatenated_df = pd.read_csv(
    root_dir
    / "3.interpret_models/linear_reg_plate_1_2_norm_data/feature_importances/data"
    / "feature_compartments.tsv",
    sep="\t",
)


# ## Plot the WT contribution for our features vs the Morphology Feature

# In[2]:


# Set significance threshold and fold change threshold
significance_threshold = -np.log10(0.05)
fold_change_threshold = 1.0
colors = ["red", "green", "blue", "black"]

# Set the figure size
plt.figure(figsize=(15, 10))

# Create the scatter plot with size and color differentiation
sns.scatterplot(
    data=concatenated_df,
    x="WT_coef",
    y="-log_p",
    size="r2_score",
    hue="compartment",
    palette=colors,
)

# Add threshold lines
plt.axhline(
    significance_threshold, color="cyan", linestyle="--", label="Significance Threshold"
)

# For adding contours:
# sns.kdeplot(data=concatenated_df, x='WT_coef', y='-log_p', levels=3, color='yellow', alpha=0.5)

# Set plot labels and title
plt.xlabel("WT_coef")
plt.ylabel("-log10(p-value)")
plt.title("Significance of WT Contribution per Morphology Feature")

# Show the plot
plt.savefig(
    output_path / "feature_wt_contribution_and_significance_with_correlation.png"
)
