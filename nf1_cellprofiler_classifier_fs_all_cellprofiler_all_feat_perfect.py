#!/usr/bin/env python
# coding: utf-8

# # Feature Analysis and Genotype Classification

# ## Imports

# In[1]:


import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import pathlib
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap


# # Seed and Generator for Reproducibility

# In[2]:


rnd_val = 0 # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val) # random number generator


# # Converting csv to pandas dataframe

# In[3]:


morph_features = 'nf1_sc_norm_fs_all_cellprofiler.csv.gz'


# In[4]:


data_dir = pathlib.Path("nf1_data_repo/4_processing_features/data/Plate1/CellProfiler")
data_comp = data_dir / morph_features


# In[5]:


# If the file isn't found in the path above then raise an error.
if not data_comp.is_file():
    raise FileNotFoundError(f"File '{data_comp}' does not exist")
    
morph_feat = pd.read_csv(data_comp)


# # Creating binary labels for the genotype

# In[6]:


lb = LabelBinarizer()
morph_feat['genotype_label'] = lb.fit_transform(morph_feat['Metadata_genotype'])
featdf = morph_feat # Create a copy for modification


# # Selecting Features

# In[7]:


feat_col = [col for col in featdf.columns if 'Metadata' not in col] # Select all columns that don't contain the Metadata in their name

featdf = featdf[feat_col]
featdf = featdf.loc[:,featdf.columns != 'Unnamed: 0'] # Remove the unnamed column


# # Sampling

# In[8]:


def down_sample(df, samp_size):
    """
    Parameters
    ----------
    df: Pandas Dataframe
        The dataframe to be sampled.
    samp_size
        The sample size to be sampled from each class.

    Returns
    -------
    Pandas Dataframe
        The sampled dataframe.

    """
    return df.sample(n=samp_size, random_state=rnd_val)

min_class_samps_size = min(featdf['genotype_label'].value_counts().values) # Sample size for the minority class
dsamp_featdf = featdf.groupby('genotype_label', group_keys=False).apply(down_sample, min_class_samps_size) # Down-sample features according to minority class
traindf, testdf = train_test_split(dsamp_featdf, random_state=rnd_val, shuffle=True, train_size=0.9) # Train and Test Dataframes


# In[11]:


train_feat = traindf.to_numpy()
test_feat = testdf.to_numpy()


# # Principle Component Analysis (PCA) Visualization

# In[12]:


null_idx = np.nonzero(train_feat[:,-1] == 0)[0]
wt_idx = np.nonzero(train_feat[:,-1] != 0)[0]


# In[13]:


pca = PCA(n_components=2)
pca_features = pca.fit_transform(train_feat[:,0:-1])
print(f'Explained variance in PC1 and PC2 = {np.sum(pca.explained_variance_ratio_)}')
null = plt.scatter(pca_features[null_idx,0],pca_features[null_idx,1], marker='x', color='r')
wt = plt.scatter(pca_features[wt_idx,0],pca_features[wt_idx,1], marker='.', color='b')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend((null, wt), ('null', 'wt'))
plt.show()


# In[14]:


reducer = umap.UMAP(random_state=rnd_val)
reducer.fit(train_feat[:,0:-1])


# In[15]:


null = plt.scatter(reducer.embedding_[null_idx,0],reducer.embedding_[null_idx,1], marker='x', color='r')
wt = plt.scatter(reducer.embedding_[wt_idx,0],reducer.embedding_[wt_idx,1], marker='.', color='b')
plt.title('Embedding of the training set by UMAP', fontsize=24)
plt.legend((null, wt), ('null', 'wt'), loc='lower right')
plt.show()


# # K Cross Validation

# In[16]:


num_splits = 5 # Default number of splits

def kcross_val(model, feat, splits=num_splits):
    """
    Parameters
    ----------
    model: sklearn model
        The model to be trained and evaluated.
    feat : Pandas Dataframe
        The preprocessed dataframe with features and labels.

    Returns
    -------
    dict
        A dictionary containing: {Best validation accuracy, Best model, Prediction of best model, Validation set labels}.
        "Best" corresponds to the greatest validation accuracy.
    """

    cv = KFold(n_splits = splits, random_state=rnd_val, shuffle=True)
    max_acc = 0 # Keep track of the maximum accuracy

    for idx, (train_idx, val_idx) in enumerate(cv.split(X=feat)): # iterates through index splits
        yval = feat[val_idx,-1].astype(np.int64) # Validation labels converted to ints
        ytrain = feat[train_idx,-1].astype(np.int64) # Training labels converted to ints
        Xval = feat[val_idx,0:-1] # Validation features
        Xtrain = feat[train_idx,0:-1] # Training features

        model.fit(Xtrain, ytrain) # Fitting the model
        acc = model.score(Xval, yval) # Scoring the model on the validation set

        if max_acc < acc: # Find the model with the best validation accuracy
            
            max_acc, best_model, best_preds, val_labels = acc, model, model.predict(Xval), yval
    
    res = {'acc':max_acc, 'model':best_model, 'preds':best_preds, 'labels':val_labels} # Store as a dictionary

    return res


# In[17]:


# Returns the naive accuracy:
def naive_acc(labels):
    """
    Parameters
    ----------
    labels
        The genotype labels.

    Returns
    -------
    float
        A naive accuracy (baseline) for the given labels.
    """
    naive_pred = rng.integers(low=0, high=2, size=labels.shape[0])
    naive_acc = accuracy_score(naive_pred,labels)
    return naive_acc


# ## Confusion Matrix

# In[18]:


def conf_mat(model_res, mat_title='Confusion Matrix'):
    """
    Parameters
    ----------
    model_res : dict
        A dictionary containing: Best validation accuracy, Best model, Prediction of best model, Validation set labels.
        "Best" corresponds to the greatest validation accuracy.

    Returns
    -------
    dict
        A naive accuracy (baseline) for the given labels.
    """
    cm = confusion_matrix(model_res['labels'],model_res['preds'])
    fig, ax = plt.subplots(figsize=(5,5), dpi=100)
    display = ConfusionMatrixDisplay(cm, display_labels=np.unique(model_res['labels']))
    ax.set(title=mat_title)
    display.plot(ax=ax);


# # LRC Model

# In[19]:


lrc = LogisticRegression(random_state=rnd_val)

lrc_results = kcross_val(lrc, train_feat)


# In[20]:


print(f"Validation model accuracy = {lrc_results['acc']}")
print(f"Naive accuracy = {naive_acc(lrc_results['preds'])}")


# In[21]:


conf_mat(lrc_results, 'Confusion Matrix for Logistic Regression')


# In[22]:


lrc_test_acc = lrc.score(test_feat[:,0:-1], test_feat[:,-1].astype(np.int64))
print(f'Logistic Regression Test Accuracy: {lrc_test_acc:.2f}')


# # Adaboost

# In[23]:


adab = AdaBoostClassifier(n_estimators=100, random_state=rnd_val)

adab_results = kcross_val(adab,train_feat)


# In[24]:


print(f"Model Validation accuracy = {adab_results['acc']}")
print(f"Naive accuracy = {naive_acc(adab_results['preds'])}")


# In[25]:


conf_mat(adab_results, 'Confusion Matrix for Adaboost Regression')


# In[26]:


adab_test_acc = adab.score(test_feat[:,0:-1], test_feat[:,-1].astype(np.int64))

print(f'Adaboost Test Accuracy: {adab_test_acc}')

