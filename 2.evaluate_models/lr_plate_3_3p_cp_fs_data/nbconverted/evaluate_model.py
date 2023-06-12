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

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
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


# ## Import Utilities

# In[ ]:


sys.path.append(f"{root_dir}/utils")
import preprocess_utils as ppu


# # Seed and Generator for Reproducibility

# In[ ]:


rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator
# plt.switch_backend("Agg") # Switch to non-interactive backend


# ## Create the directory path if non-existent

# In[ ]:


fig_out_path = Path("model_performance_figures")

if not fig_out_path.exists():
    fig_out_path.mkdir()


# ## Load Model

# In[ ]:


models_path = Path(f"{root_dir}/1.train_models/lr_plate3_cp_fs_data/data")

lr = load(models_path / "lr_model.joblib")


# ## Load Data

# In[ ]:


testdf = load(models_path / "testdf.joblib")
le = load(models_path / "label_encoder.joblib")


# ## Make Predictions

# In[ ]:


testdf["preds"] = lr.predict(testdf.drop("label", axis="columns"))


# In[ ]:


print(f"Accuracy = {accuracy_score(testdf['label'], testdf['preds'])}")


# ## Resave the testdf with Predictions

# In[ ]:


dump(testdf, models_path / "testdf.joblib")


# ## Create Dataframe with coefficients for each Genotype

# In[ ]:


featdf = testdf.drop(testdf.index)
featdf.drop(["label", "preds"], inplace=True, axis=1)
featdf = featdf.T
featdf = featdf.reset_index()
featdf = featdf.rename(columns={"index": "features"})


# In[ ]:


featdf = pd.concat(
    [pd.DataFrame(lr.coef_.T, columns=le.classes_.tolist()), featdf], axis="columns"
)


# ## Find Confusion Matrix

# In[ ]:


cm3 = pd.crosstab(
    testdf["label"], testdf["preds"], rownames=["True"], colnames=["Predicted"]
)
ax = sns.heatmap(
    cm3,
    annot=True,
    cmap="Blues",
    xticklabels=le.classes_.tolist(),
    yticklabels=le.classes_.tolist(),
)
cbar = ax.collections[0].colorbar
cbar.set_label("Number of Single Cells")
plt.title("Performance predicting Genotype")
plt.savefig(f"{fig_out_path}/lr_conf_mat.png")  # Save the Confusion Matrix


# In[ ]:


print(
    f"The number of incorrectly classified cells is {(cm3.sum() - np.diag(cm3)).sort_values(ascending=False)}"
)


# ## Calculate the metrics for each Genotype

# In[ ]:


precision = precision_score(testdf["label"], testdf["preds"], average=None)
recall = recall_score(testdf["label"], testdf["preds"], average=None)
f1 = f1_score(testdf["label"], testdf["preds"], average=None)

df = pd.DataFrame({"recall": recall, "precision": precision, "f1_score": f1})


# Makes the columns individual values in the 'Group' column and
df = df.melt(var_name="Group", value_name="Value")

# Assigns the genotype to each row:
pos_genes = le.classes_.tolist()
df["genotype"] = pos_genes * (len(df) // len(pos_genes))


# ## Plot the performance of the Logistic Regression

# In[ ]:


plt.figure(figsize=(12, 12))
sns.set(font_scale=2)
sns.barplot(x="Group", y="Value", hue="genotype", data=df)

plt.xlabel("Metric")
plt.ylabel("Value")
plt.title("Logistic Regression Model Performance")
plt.legend(loc="upper right", bbox_to_anchor=(1, 1.01), frameon=False)
plt.tight_layout()
plt.savefig(fig_out_path / "lr_performance_per_genotype_bar_plot.png")
