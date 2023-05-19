#!/usr/bin/env python
# coding: utf-8

# # Obnibus and post hoc testing with Anova and scheffe's test

# Obnibius Testing: https://www.statology.org/omnibus-test/
#
# Post hoc Testins: https://www.statology.org/anova-post-hoc-tests/

# ## Imports

# In[1]:


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from sklearn.ensemble import IsolationForest

sys.path.append("../utils")
import analysis_utils as au
import preprocess_utils as ppu


# In[2]:


rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator


# # Preprocess data using preprocess utils

# ## Remove Outliers

# In[3]:


plates = {}

plates["1"] = {
    "path": "../nf1_painting_repo/3.processing_features/data/feature_selected_data/Plate_1_sc_norm_fs.parquet"
}
plates["2"] = {
    "path": "../nf1_painting_repo/3.processing_features/data/feature_selected_data/Plate_2_sc_norm_fs.parquet"
}

for plate, vals in plates.items():
    # Initializes the preprocessing class
    po = ppu.Preprocess_data(path=vals["path"])

    # Gets the dataframe after removing metadata columns,
    # except for the 'Metadata_genotype' column specified above
    platedf = po.remove_meta(po.df, kept_meta_columns=["Metadata_genotype"])

    # Use isolation forest to select inliers:
    isof = IsolationForest(random_state=rnd_val)
    out_preds = isof.fit_predict(platedf.drop(columns=["Metadata_genotype"]))
    ind = np.nonzero(out_preds == 1)[0]  # Select inliers

    # Select inlier samples:
    plates[plate]["df"] = platedf.iloc[ind]

platesdf = [plates[key]["df"] for key in plates]


# In[4]:


### Use only the common columns between both plates:
common_columns = list(platesdf[0].columns.intersection(platesdf[1].columns))
platesdf[0] = platesdf[0].loc[:, common_columns]
platesdf[1] = platesdf[1].loc[:, common_columns]


# # Conduct testing with scheffe's test

# In[5]:


test = sp.posthoc_scheffe
# Pass the plates dataframes in order, so that genotypes are suffixed corretly (eg. WT1WT2)
# Plates are 1 indexed
st = au.Sig_testing(platesdf)
anova_feats, sig_anova_pvals = st.anova_test()
res_test = st.posthoc_test(anova_feats, sig_anova_pvals, test)


# In[6]:


# Returns a dictionary of genotype pairs containing the significant column names
# being analyzed
sig_groups = st.get_columns(res_test["sig_feat_phoc"])


# In[7]:


tot_columns = len(platesdf[0].columns)
sig_groupsdf = pd.DataFrame(
    list({grp: len(tot) for grp, tot in sig_groups.items()}.items()),
    columns=["genotype", "number_of_significant_columns"],
)
sig_groupsdf["proportion_of_significant_columns"] = (
    sig_groupsdf["number_of_significant_columns"] / tot_columns
)
sig_groupsdf = sig_groupsdf.round({"proportion_of_significant_columns": 2})


# In[8]:


out_file = Path("data/plate_1_2_genotype_signifance.tsv")

if not out_file.parent.is_dir():
    out_file.parent.mkdir(parents=True, exist_ok=True)

sig_groupsdf.to_csv(out_file, sep="\t", index=False)
