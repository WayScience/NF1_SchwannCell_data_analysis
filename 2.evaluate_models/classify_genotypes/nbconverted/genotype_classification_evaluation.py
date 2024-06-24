#!/usr/bin/env python
# coding: utf-8

# # Calculate evaluation Plate and Datasplit evaluation results
# Evaluation results include confusion matrices, pr curves, precision, recall, f1-score, and accuracy.
# This occurs for each plate across all splits.

# In[1]:


import pathlib
from collections import defaultdict

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

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


data_path = pathlib.Path(f"{root_dir}/1.train_models/classify_genotypes/data")

evaldf = pd.read_parquet(f"{data_path}/nf1_final_model_eval_data.parquet")
model = load(f"{data_path}/trained_nf1_model.joblib")
le = load(f"{data_path}/trained_nf1_model_label_encoder.joblib")


# In[4]:


evaldf


# ### Outputs

# In[5]:


eval_path = pathlib.Path("model_evaluation_data")
eval_path.mkdir(parents=True, exist_ok=True)


# In[6]:


gene_column = "true_genotype"

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


# ## Calculate evaluation metrics

# In[7]:


# Define evaluation metric data
# The "metrics" include precision, recall, accuracy, and f1 scores
eval_mets = {
    met: defaultdict(list) for met in
    ("metrics", "precision_recall", "confusion_matrix")
}

# Labels of confusion matrices in dataframe
cm_true_labels = [
    le.classes_[0],
    le.classes_[0],
    le.classes_[1],
    le.classes_[1]
]

cm_pred_labels = [
    le.classes_[0],
    le.classes_[1],
    le.classes_[0],
    le.classes_[1]
]

def compute_metrics(_df, _plate, _split):
    """
    Parameters
    ----------
    _df: Pandas Dataframe
        Model data to be evaluated.

    _plate: String
        Name of the plate for storing the metrics

    _split: String
        Name of the data split for storing the metric
    """

    y_true = _df[gene_column]
    y_pred = _df["predicted_genotype"]
    y_proba = _df["probability_WT"]

    # Store metrics
    eval_mets["metrics"]["f1_score"].append(f1_score(y_true, y_pred))
    eval_mets["metrics"]["precision"].append(precision_score(y_true, y_pred))
    eval_mets["metrics"]["recall"].append(recall_score(y_true, y_pred))
    eval_mets["metrics"]["accuracy"].append(accuracy_score(y_true, y_pred))
    eval_mets["metrics"]["plate"].append(_plate)
    eval_mets["metrics"]["datasplit"].append(_split)

    # Store precision and recall data
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_size = precision.shape[0]
    eval_mets["precision_recall"]["precision"].extend(precision.tolist())
    eval_mets["precision_recall"]["recall"].extend(recall.tolist())
    eval_mets["precision_recall"]["plate"].extend([_plate] * pr_size)
    eval_mets["precision_recall"]["datasplit"].extend([_split] * pr_size)

    # Store confusion matrices
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.flatten()
    cm_size = cm.shape[0]
    eval_mets["confusion_matrix"]["confusion_values"].extend(cm.tolist())
    eval_mets["confusion_matrix"]["true_genotype"].extend(cm_true_labels)
    eval_mets["confusion_matrix"]["predicted_genotype"].extend(cm_pred_labels)
    eval_mets["confusion_matrix"]["plate"].extend([_plate] * cm_size)
    eval_mets["confusion_matrix"]["datasplit"].extend([_split] * cm_size)


# In[8]:


# Iterate through each data split
for split in evaldf["datasplit"].unique():

    # Calculate metrics for all plates
    df_temp = evaldf.loc[(evaldf["datasplit"] == split)].copy()
    compute_metrics(df_temp, "all_plates", split)

    # Calculate metrics for each plate
    for plate in evaldf["Metadata_Plate"].unique():
        df_temp = evaldf.loc[(evaldf["Metadata_Plate"] == plate) & (evaldf["datasplit"] == split)].copy()
        df_temp = down_sample_by_genotype(df_temp)
        compute_metrics(df_temp, plate, split)


# ### Save evaluation metrics and model coefficients for plotting

# In[9]:


for met, met_data in eval_mets.items():
    pd.DataFrame(eval_mets[met]).to_parquet(f"{eval_path}/{met}_final_model.parquet")

pd.DataFrame(
    {
        "feature_names": model.feature_names_in_,
        "feature_importances": model.coef_.reshape(-1)
    }
).to_parquet(f"{eval_path}/feature_importances.parquet")

