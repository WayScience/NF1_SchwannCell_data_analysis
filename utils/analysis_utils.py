from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import itertools
from collections import defaultdict

rnd_val = 0 # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val) # random number generator

# Displays a plot of the first two principal components
def plot_pca(df, loc='lower right', title='Principal component plot of training set'):
    """
    Parameters
    ----------
    data: pandas dataframe of shape (samples, features)
        The data to be plotted, where the last column contains the labels that will be in the legend
        
    loc: string
    	Location of the legend as specified by matplotlib (optional)
    	
    title : str
        The title of the PC plot. (optional)

    """
    feats = df.iloc[:,:-1]
    labels = df.iloc[:,-1]
    
    unique_labels = labels.unique()
    labelsdt = {lab: np.nonzero(labels.isin([lab]))[0] for lab in unique_labels}

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(feats)
    for gene, labs in labelsdt.items():
    	plt.scatter(pca_features[labs,0],pca_features[labs,1], label=gene)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title, fontsize=24)
    plt.legend(loc=loc)
    plt.show()
    print(f'Explained variance in PC1 and PC2 = {np.sum(pca.explained_variance_ratio_)}')

# Displays a plot of the umap components
def plot_umap(df, loc='lower right', title='Embedding of the training set by UMAP'):
    """
    Parameters
    ----------
    data: pandas dataframe of shape (samples, features)
        The data to be plotted, where the last column contains the labels that will be in the legend
        
    loc: string
    	Location of the legend as specified by matplotlib (optional)
    	
    title : str
        The title of the UMAP plot. (optional)

    """
    feats = df.iloc[:,:-1]
    labels = df.iloc[:,-1]
    
    unique_labels = labels.unique()
    labelsdt = {lab: np.nonzero(labels.isin([lab]))[0] for lab in unique_labels}
    
    reducer = umap.UMAP(random_state=rnd_val)
    reducer.fit(feats)
    
    for gene, labs in labelsdt.items():
    	plt.scatter(reducer.embedding_[labs,0],reducer.embedding_[labs,1], label=gene)
    
    plt.title(title, fontsize=24)
    plt.legend(loc=loc)
    plt.show()
    
def sig_test(test, plates):
    """
    Parameters
    ----------
    test: scikit_posthocs test function
    plates : A list, a tuple, or another similar iterable of plate dataframes
        Each plate in the iterable should have the same number of features and be preprocessed to remove columns that arent features. The exception is the 'Metadata_genotype' column, where the values must be either 'Null' or 'WT'. If there is a 'plate' or 'group' column, those will not be considered. Outliers should also be removed.
        
    Returns
    ----------
    Dictionary of anova and post hoc results. What is returned depends on the significance of the tests.

    """
        
    platesdt = defaultdict(None)
    gtypes = ['Null','WT'] # The 2 types of genotypes
    
    # Store genotype data in dictionary:
    for i, plate in enumerate(plates):
        for gtype in gtypes:
            platesdt[gtype + str(i)] = plate.loc[plate['Metadata_genotype'] == gtype].drop(columns=['Metadata_genotype'])

    anova = defaultdict(None)
    posdf = list(platesdt.values())
    
    # Concatenate genotype series to calculate p value
    for col in posdf[0].columns:
        col_series = []
        for df in platesdt.values():
            col_series.append(df[col])
            
        _, pval = f_oneway(*col_series)
        anova[col] = pval
        
    alpha = 0.05 # Critical value    
             
    # Find the significant features based on the critical value:
    sig_anova = {k: v for k, v in anova.items() if v < alpha}
    anova_pvals = np.array(list(anova.values()))
    sig_ind = np.nonzero(anova_pvals < 0.05)[0]
             
    pot_feat = posdf[0].iloc[:,sig_ind] # Dataframe with significant features
    test_cols = pot_feat.columns # Significant columns to use for post hoc tests

    platesdt = defaultdict(None)
    
    # Combine the plates data and create groups from plate number and genotype for post hoc tests:
    for i, df in enumerate(plates):
        df['plate'] = [str(i+1)]*len(df)
        df['group'] = df['Metadata_genotype'] + df['plate']
        df.drop(columns=['Metadata_genotype','plate'], inplace=True) # Remove unnecessary columns for testing
        platesdt[i] = df

    combdf = pd.concat([df for _, df in platesdt.items()], axis=0) # Dataframe with combined plates
             
    # Post hoc test:
    ## Combine pairs of groups:
    groups = combdf['group'].unique()

    # Find paired combinations of genotypes:
    group_comb = [' '.join(p) for p in itertools.combinations(groups, 2)]
    group_comb = [[p[0:p.index(' ')], p[p.index(' ')+1:]] for p in group_comb]
             
    sig_col_ptests =  defaultdict(None) # Holds significant p values for each feature
    nsig_col_ptests =  defaultdict(None) # Holds insignificant p values for each feature

    for ccol in test_cols:
        sig_group_tests = defaultdict(None) # Stores significant p test values for a feature's groups
        nsig_group_tests = defaultdict(None)  # Stores insignificant p test values for a feature's groups
        col_tests = test(combdf, val_col=ccol, group_col='group') # Get test results
        for pair in group_comb: # Iterate through each test pair
            pval = col_tests.loc[pair[0], pair[1]] # Obtain the p value for a given test pair
            # Checks if the p value is critical:
            if pval < alpha:
                sig_group_tests[''.join(pair)] = pval # Adds the significant pvalues for each applicable test pair

            else:
                nsig_group_tests[''.join(pair)] = pval # Adds the insignificant pvalues for each remaining test pair

        if bool(sig_group_tests):
            sig_col_ptests[ccol] = sig_group_tests # Store the significant pvalues for applicable features

        if bool(nsig_group_tests):
            nsig_col_ptests[ccol] = nsig_group_tests # Store the insignificant pvalues for the remaining features
            
    results = defaultdict(None)

    if anova_pvals.shape[0] != 0: # If there are significant features based on the anova
        # If there significant features based on the pos hoc analysis then store them:
        if bool(sig_col_ptests):
            results['sig_feat_phoc'] = sig_col_ptests

        # If there are not significant results from the post hoc analysis then store them:
        if bool(nsig_col_ptests):
            results['notsig_feat_phoc'] = nsig_col_ptests
            
        results['sig_feat_anova'] = anova # Store the results of the anova if there was significance

        return results

    # If the anova doesn't have significant results, return None:
    else:
        return None
    
def get_columns(sig_feat_phoc):
    """
    Parameters
    ----------
    sig_feat_phoc: Dictionary of dictionaries 
        Returned by the sig_test function, which contains significant genotype pairs (if they exist) for each column.
        
    Returns
    ----------
    cats: Dictionary
        A dictionary of genotype pairs containing the significant column names
    """
    cats = defaultdict(list)
    
    for col, groups in sig_feat_phoc.items():
        for group, _ in groups.items():
            cats[group].append(col)
    
    return cats
