#!/usr/bin/env python
# coding: utf-8

# # Obnibus and post hoc testing

# ## Imports

# In[1]:


import sys
from sklearn.ensemble import IsolationForest
import scikit_posthocs as sp
import pathlib
import numpy as np
import pandas as pd

rel_root = pathlib.Path("..")
sys.path.append(f'{rel_root}/utils')
import analysis_utils as au
import preprocess_utils as ppu


# # Preprocess data using preprocess utils

# In[2]:


filename = 'nf1_sc_all_cellprofiler.csv.gz'
po1 = ppu.preprocess_data(1, filename, rel_root, ['Metadata_genotype'])
po2 = ppu.preprocess_data(2, filename, rel_root, ['Metadata_genotype'])


# In[3]:


rnd_val = po1.rnd_val # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val) # random number generator


# In[4]:


plate1df = po1.get_ml_df()
plate2df = po2.get_ml_df()


# In[5]:


plate1df.drop(['Cytoplasm_Number_Object_Number'], axis=1, inplace=True)
plate2df.drop(['Cytoplasm_Number_Object_Number'], axis=1, inplace=True)


# ## Remove outliers

# In[6]:


isof1 = IsolationForest(random_state = rnd_val)
out_preds1 = isof1.fit_predict(plate1df.drop(columns=['Metadata_genotype']))
ind1 = np.nonzero(out_preds1 == 1)[0]


# In[7]:


isof2 = IsolationForest(random_state = rnd_val)
out_preds2 = isof2.fit_predict(plate2df.drop(columns=['Metadata_genotype']))
ind2 = np.nonzero(out_preds2 == 1)[0]


# In[8]:


plate1df = plate1df.iloc[ind1]
plate2df = plate2df.iloc[ind2]


# # Conduct testing with scheffe's test

# In[9]:


test = sp.posthoc_scheffe
# Pass the plates dataframes in order, so that genotypes are suffixed corretly (eg. WT1WT2)
# Plates are 1 indexed
res_test = au.sig_test(test, [plate1df, plate2df])


# In[10]:


sig_groups = au.get_columns(res_test['sig_feat_phoc'])


# In[11]:


tot_columns = len(plate1df.columns)
for group, cols in sig_groups.items():
    print(f'In group {group}: {(len(cols) / tot_columns)*100:.2f}% of columns are significant')

