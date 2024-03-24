#!/usr/bin/env python
# coding: utf-8

# # Calculate evaluation metrics from training, validation, and testing data

# In[1]:


import pathlib
from collections import defaultdict

import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

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


data_suf = "log_reg_cp_fs_data_plate_5"

data_path = pathlib.Path(f"{root_dir}/1.train_models/log_reg_plates_3_3p_5_cp_norm_data/data")

model_predf = pd.read_parquet(f"{data_path}/model_data_{data_suf}.parquet")
evaldf = pd.read_parquet(f"{data_path}/model_data_log_reg_cp_fs_data_plate_5.parquet")
le = load(f"{data_path}/label_encoder_log_reg_cp_fs_data_plate_5.joblib")


# ### Outputs

# In[4]:


models_path = pathlib.Path("model_eval_data")
models_path.mkdir(parents=True, exist_ok=True)


# In[5]:


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

# In[6]:


# Define evaluation metrics
eval_mets = {
    met: defaultdict(list) for met in
    ("f1_score", "precision_recall", "confusion_matrix")
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

    # Store f1 scores
    eval_mets["f1_score"]["f1_score"].append(f1_score(y_true, y_pred))
    eval_mets["f1_score"]["plate"].append(_plate)
    eval_mets["f1_score"]["datasplit"].append(_split)

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


# In[7]:


# Iterate through each data split
for split in evaldf["datasplit"].unique():

    # Calculate metrics for all plates
    df_temp = evaldf.loc[(evaldf["datasplit"] == split)].copy()
    compute_metrics(df_temp, "all_plates", split)

    # Calculate metrics for each plate
    for plate in evaldf["plate"].unique():
        df_temp = evaldf.loc[(evaldf["plate"] == plate) & (evaldf["datasplit"] == split)].copy()
        df_temp = down_sample_by_genotype(df_temp)
        compute_metrics(df_temp, plate, split)


# ### Save evaluation metrics for plotting

# In[8]:


for met, met_data in eval_mets.items():
    pd.DataFrame(eval_mets[met]).to_parquet(f"{data_path}/plate_{met}.parquet")

