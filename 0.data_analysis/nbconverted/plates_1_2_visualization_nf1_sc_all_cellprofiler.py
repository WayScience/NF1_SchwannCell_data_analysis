#!/usr/bin/env python
# coding: utf-8

# # PCA and UMAP Visualization

# In[1]:


import sys
import pathlib
import pandas as pd
from sklearn.preprocessing import LabelBinarizer as labi

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


plate1df = po1.get_ml_df()
plate2df = po2.get_ml_df()


# In[4]:


plate1df['labels'] = plate1df['Metadata_genotype'] + '1'
plate2df['labels'] = plate2df['Metadata_genotype'] + '2'


# In[5]:


plate1df.drop(['Cytoplasm_Number_Object_Number','Metadata_genotype'], axis=1, inplace=True)
plate2df.drop(['Cytoplasm_Number_Object_Number','Metadata_genotype'], axis=1, inplace=True)


# # Combining Data

# In[6]:


plates = pd.concat([plate1df, plate2df], axis=0)


# In[8]:


plateswt = plates[~plates.iloc[:,-1].str.contains('Null')]
platesnull = plates[~plates.iloc[:,-1].str.contains('WT')]


# # Visualization

# In[9]:


au.plot_pca(plate1df, title='PCA of Plate 1')


# In[10]:


au.plot_umap(plate1df, title='UMAP of Plate 1')


# In[11]:


au.plot_pca(plate2df, title='PCA of Plate 2')


# In[19]:


au.plot_umap(plate2df, loc='upper left', title='UMAP of Plate 2')


# In[13]:


au.plot_pca(plates, title='PCA of Plates 1 and 2')


# In[14]:


au.plot_umap(plates, loc='upper right', title='UMAP of Plates 1 and 2')


# In[15]:


au.plot_umap(plateswt, loc='lower left', title='UMAP of Plates 1 and 2')


# In[16]:


au.plot_pca(plateswt, title='PCA of Plates 1 and 2')


# In[17]:


au.plot_umap(platesnull, title='UMAP of Plates 1 and 2')


# In[18]:


au.plot_pca(platesnull, title='PCA of Plates 1 and 2')

