import pandas as pd
import numpy as np
import pathlib
from sklearn.preprocessing import LabelBinarizer
import os

class preprocess_data:

    rnd_val = 0 # Random value for all seeds
    rng = np.random.default_rng(seed=rnd_val) # random number generator

    def __init__(self, plate, filename, rel_root, kept_meta_columns=None, cell_method='cellprofiler'):
        cell_methods = ('cellprofiler','deepprofiler')
        plates = np.array([1,2])
        self.kept_meta_columns = kept_meta_columns

        root_folder = 'nf1_data_repo'
        root_dir_path = rel_root / pathlib.Path(root_folder) # Relative path to data repo folder
        data_dir = rel_root / root_folder

        if not root_dir_path.is_dir():
            raise FileNotFoundError(f"The directory {root_folder} is not found in the '{rel_root.resolve()}' path.")

        if not cell_method in cell_methods:
            raise ValueError(f"No Matching cell method can be found. Possible options include:\n{str(cell_methods)}")

        if not bool(np.isin(plate,plates)):
            raise ValueError(f"No Matching plate can be found. Possible options include:\n{str(plates)}")

        if plate == 2:
            if cell_method == cell_methods[0]:
                data_dir = data_dir / "4_processing_features/data/Plate2/CellProfiler"

        if plate == 1:
            if cell_method == cell_methods[0]:
                data_dir = data_dir / "4_processing_features/data/Plate1/CellProfiler"
            elif cell_method == cell_methods[1]:
                data_dir = data_dir / "4_processing_features/data/Plate1/DeepProfiler"

        full_path = pathlib.Path(data_dir) / filename
        
        # If the file isn't found in the path above then raise an error.
        if not full_path.is_file():
            raise FileNotFoundError(f"File '{full_path}' does not exist")
        
        self.df = pd.read_csv(full_path)

        self.df = self.df.loc[:,self.df.columns != 'Unnamed: 0'] # Remove the unnamed column

    def down_sample(self, df, target_column):
        """
        Parameters
        ----------
        df: Pandas Dataframe
            The dataframe to be sampled.
        target_column:
            The column in the dataframe that contains labels of the classification

        Returns
        -------
        Pandas Dataframe
            The sampled dataframe with an equal class distribution, according to the smallest number of classes.
        """
        min_class_samps_size = min(df[target_column].value_counts())
        return df.groupby(target_column, group_keys=False).sample(n=min_class_samps_size, random_state=self.rnd_val)

    def get_raw_df(self):
        """
        Parameters
        ----------
        Self

        Returns
        -------
        Pandas Dataframe
            The class instance of the pandas dataframe
        """
        
        return self.df

    def remove_meta(self, df, kept_meta_columns=None):
        """
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
        feat_col = [col for col in self.df.columns if 'Metadata' not in col] # Select all columns that don't contain the Metadata in their name
        if kept_meta_columns is not None:
            kept_col_df = df[kept_meta_columns]
            return pd.concat([kept_col_df, df[feat_col]], axis=1)
        else:
            return df[feat_col]
        return df

    def get_ml_df(self, df=None, remove_meta = True):
        """
        Parameters
        ----------
        df: pandas Dataframe or nothing
            Can be set if you don't want to use the class's dataframe instance (optional)
            
        remove_meta: Bool
            Will remove the metadata if True, except if a dataframe is supplied to the function (optional)

        Returns
        -------
        Pandas Dataframe
            The dataframe without all metadata columns, excluding any metadata that was previously specifed to keep.
        """
        if df is None:
            df = self.df
        else:
            df = self.remove_meta(df, self.kept_meta_columns)
            
        if remove_meta:
            df = self.remove_meta(df, self.kept_meta_columns)
        return df
