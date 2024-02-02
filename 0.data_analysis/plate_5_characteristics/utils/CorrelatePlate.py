import pandas as pd
from CorrelateAggWells import CorrelateAggWells


class CorrelatePlate:
    """
    Compute correlations for plate data features
    """

    def __init__(self):
        pass

    def correlate_agg_wells(self, _df, _well_col, _feat_cols, _group_cols):
        """
        Computes correlations between and within groups using well feature vectors.
        In this way we correlate each well pair between and within groups.

        Parameters
        ----------
        _df: pd.DataFrame
            Contains the feature and group columns to compare across wells specified by _well_col.

        _well_col: String
            Name of the well column.

        _feat_col: pd.Series or List
            Names of feature columns.

        _group_col: pd.Series or List
            Names of group columns.

        Returns
        -------
        corrs: Dictionary
            The inter-group and intra-group correlated data including well pairs, group pairs, and correlations.
        """

        _df_copy = _df.copy()
        _df_copy.set_index(_well_col, drop=False, inplace=True)

        # Compute aggregated well correlations for the plate data
        caw = CorrelateAggWells()
        return pd.concat(
            [
            caw.inter_correlations(_df, _well_col, _feat_cols, _group_cols),
            caw.intra_correlations(_df, _well_col, _feat_cols, _group_cols)
            ],
            axis=0
        )
