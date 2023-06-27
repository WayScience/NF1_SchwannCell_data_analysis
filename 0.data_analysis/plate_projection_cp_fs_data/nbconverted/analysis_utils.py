""" This file provides analysis utilities for a variety of tasks """

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import itertools

rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator


def plot_pca(
    feats,
    labels,
    save_args,
    loc="lower right",
    title="Principal component plot of training set",
):
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
        plt.scatter(pca_features[labs, 0], pca_features[labs, 1], label=gene)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title, fontsize=24)
    plt.legend(loc=loc)
    plt.savefig(**save_args)
    print(
        f"Explained variance in PC1 and PC2 = {np.sum(pca.explained_variance_ratio_)}"
    )


# Displays a plot of the umap components
def plot_umap(
    feats,
    labels,
    save_args,
    loc="lower right",
    title="Embedding of the training set by UMAP",
):
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
        plt.scatter(
            reducer.embedding_[labs, 0], reducer.embedding_[labs, 1], label=gene
        )

    plt.title(title, fontsize=24)
    plt.legend(loc=loc)
    plt.savefig(**save_args)
