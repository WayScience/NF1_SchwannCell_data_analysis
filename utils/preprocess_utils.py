import pandas as pd
import numpy as np
import pathlib
from sklearn.preprocessing import LabelBinarizer
import os

class Preprocess_data:
    """
    A simplified means to retrieving dataframes from files and removing choosen metadata.
    """

    rnd_val = 0 # Random value for all seeds
    rng = np.random.default_rng(seed=rnd_val) # random number generator

    def __init__(self, path):
        """
        Parameters
        ----------
        path: string or pathlib path
            The path of the data file
        kept_meta_columns: list of strings
            Metadata column names to be kept in the retrieved dataframe (optional)
        """
        path = pathlib.Path(path)
                     
        # If the file isn't found in the path above then raise an error.
        if not path.is_file():
            raise FileNotFoundError(f"File '{path}' does not exist")
            
        if 'parquet' in path.name:
            self.df = pd.read_parquet(path)
            
        elif 'csv' in path.name:
            self.df = pd.read_csv(path)
            
        elif 'tsv' in path.name:
            self.df = pd.read_csv(path, delimiter='\t')
            
        else:
            raise ValueError("The file must be a parquet, csv, or tsv, with the applicable extension included in the filename.")

        self.df = self.df.loc[:,self.df.columns != 'Unnamed: 0'] # Remove the unnamed column

    def remove_meta(self, df, kept_meta_columns=None):
        """
        This function removes all metadata columns, except the ones specified.
        
        Parameters
        ----------
        df: Pandas Dataframe
            The dataframe to be sampled.
            
        kept_meta_columns: List of strings
            The columns in the dataframe that should be retained (optional)

        Returns
        -------
        Pandas Dataframe
            The dataframe without all metadata columns, excluding the specified metadata columns.
        """
        feat_col = [col for col in df.columns if 'Metadata' not in col] # Select all columns that don't contain the Metadata in their name
        if kept_meta_columns is not None:
            kept_col_df = df[kept_meta_columns]
            return pd.concat([kept_col_df, df[feat_col]], axis=1)
        else:
            return df[feat_col]
        return df