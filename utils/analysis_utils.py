""" This file provides analysis utilities for a variety of tasks """

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

def plot_pca(feats, labels, save_args, loc='lower right', title='Principal component plot of training set'):
    """
    Plots the first two principal components and displays the explained variance.
    
    Parameters
    ----------
    feats: Pandas Dataframe of numerical values
        The dataframe of features to use for the pca plot
        
    labels: Pandas Dataframe of strings
        The dataframe of labels to use for labeling points on the plot
    
    save_args: dictionary
        The arguments to pass to the savefig function (Please see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html for all possible options)
        
    data: pandas dataframe of shape (samples, features)
        The data to be plotted, where the last column contains the labels that will be in the legend
        
    loc: string
    	Location of the legend as specified by matplotlib (optional)
    	
    title : str
        The title of the PC plot. (optional)

    """
    
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
    plt.savefig(**save_args)
    print(f'Explained variance in PC1 and PC2 = {np.sum(pca.explained_variance_ratio_)}')

# Displays a plot of the umap components
def plot_umap(feats, labels, save_args, loc='lower right', title='Embedding of the training set by UMAP'):
    """
    Parameters
    ----------
    feats: Pandas Dataframe of numerical values
        The dataframe of features to use for the pca plot
        
    labels: Pandas Dataframe of strings
        The dataframe of labels to use for labeling points on the plot
    
    save_args: dictionary
        The arguments to pass to the savefig function (Please see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html for all possible options)
    
    data: pandas dataframe of shape (samples, features)
        The data to be plotted, where the last column contains the labels that will be in the legend
        
    loc: string
    	Location of the legend as specified by matplotlib (optional)
    	
    title : str
        The title of the UMAP plot. (optional)

    """
    
    unique_labels = labels.unique()
    labelsdt = {lab: np.nonzero(labels.isin([lab]))[0] for lab in unique_labels}
    
    reducer = umap.UMAP(random_state=rnd_val)
    reducer.fit(feats)
    
    for gene, labs in labelsdt.items():
    	plt.scatter(reducer.embedding_[labs,0],reducer.embedding_[labs,1], label=gene)
    
    plt.title(title, fontsize=24)
    plt.legend(loc=loc)
    plt.savefig(**save_args)
    
class Sig_testing():
    def __init__(self, plates):
        """
        Parameters
        ----------        
        plates : A list, a tuple, or another similar iterable of plate dataframes
            Each plate in the iterable should have the same number of features and be preprocessed to remove columns that arent features. The exception is the 'Metadata_genotype' column. If there is a 'plate' or 'group' column, those will not be considered. Outliers should also be removed.
        """
        self.plates = plates
        self.gtypes = plates[0]['Metadata_genotype'].unique()
        
    def anova_test(self, alpha=0.05):
        """
        Parameters
        ----------
        test: scikit_posthocs test function
        plates : A list, a tuple, or another similar iterable of plate dataframes
            Each plate in the iterable should have the same number of features and be preprocessed to remove columns that arent features. The exception is the 'Metadata_genotype' column. If there is a 'plate' or 'group' column, those will not be considered. Outliers should also be removed.
            
        Returns
        ----------
        
        anova: Pandas Dataframe
            Significant features as determined by the Obnibus (anova) test
            
        pot_feat: Dictionary
            Significant p values of each feature
        """
        
        platesdt = defaultdict(None)

        # Store genotype data in dictionary:
        for i, plate in enumerate(self.plates):
            for gtype in self.gtypes:
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

        # Find the significant features based on the critical value:
        sig_anova = {k: v for k, v in anova.items() if v < alpha}
        anova_pvals = np.array(list(anova.values()))
        sig_ind = np.nonzero(anova_pvals < 0.05)[0]

        pot_feat = posdf[0].iloc[:,sig_ind] # Dataframe with significant features

        return pot_feat, anova
    
    def posthoc_test(self, anova_feats, sig_anova_pvals, test, alpha=0.05):
        """
        Parameters
        ----------
        anova_feats: Pandas Dataframe
            Significant features as determined by the Obnibus (anova) test
            
        sig_anova_pvals: Dictionary
            Significant p values of each feature
            
        test: scikit_posthocs test function
            The posthoc test to be used compare different groups
            
        alpha: float
            The critical value of the tests to be conducted

        Returns
        ----------
        results (or None): Dictionary of dictionary
            Dictionary of anova and post hoc results. What is returned depends on the significance of the tests.
        """
        test_cols = anova_feats.columns # Significant columns to use for post hoc tests

        platesdt = defaultdict(None)

        # Combine the plates data and create groups from plate number and genotype for post hoc tests:
        for i, df in enumerate(self.plates):
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

        if len(anova_feats.columns) != 0: # If there are significant features based on the anova
            # If there significant features based on the pos hoc analysis then store them:
            if bool(sig_col_ptests):
                results['sig_feat_phoc'] = sig_col_ptests

            # If there are not significant results from the post hoc analysis then store them:
            if bool(nsig_col_ptests):
                results['notsig_feat_phoc'] = nsig_col_ptests

            results['sig_feat_anova'] = sig_anova_pvals # Store the results of the anova if there was significance

            return results

        # If the anova doesn't have significant results, return None:
        else:
            return None
        
    def get_columns(self, sig_feat_phoc):
        """
        This function finds the significant column names from the dictionary of groups specified
        
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
