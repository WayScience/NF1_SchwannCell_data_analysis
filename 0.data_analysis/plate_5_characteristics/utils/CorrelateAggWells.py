from collections import defaultdict
from itertools import combinations, product

import pandas as pd


class CorrelateAggWells:
    """
    Compute correlations between aggregated cell features per well
    """

    def __init__(self):
        pass


    def inter_correlations(self, _df, _well_col, _feat_cols, _group_cols):
        """
        Computes correlations between groups using well feature vectors.
        In this way we correlate each well pair between groups.

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
        corrs: pd.DataFrame
            The correlated data including well pairs, inter-group pairs, and correlations.
        """

        groupdf = _df.groupby(_group_cols)

        # Retrieve group names
        gkeys = groupdf.groups.keys()

        # All possible combinations of groups
        group_pairs = list(combinations(gkeys, 2))

        # Groups with the features as columns (for efficiently computing corrleations)
        grouptdfs = {gkey: groupdf.get_group(gkey)[_feat_cols].T.copy() for gkey in gkeys}

        # Store correlations
        corrs = defaultdict(list)

        # Iterate through each group combination
        for gpair in group_pairs:

            # Wells in each group
            group0df_well = groupdf.get_group(gpair[0])[_well_col]
            group1df_well = groupdf.get_group(gpair[1])[_well_col]

            # Iterate through each well group cartesian product and save the data
            for wpair in list(product(group0df_well, group1df_well)):
                corrs["group0"].append(gpair[0])
                corrs["group1"].append(gpair[1])
                corrs["well0"].append(wpair[0])
                corrs["well1"].append(wpair[1])
                corrs["correlation"].append(
                    grouptdfs[gpair[0]][wpair[0]].corr(
                        grouptdfs[gpair[1]][wpair[1]]
                    )
                )

        return pd.DataFrame(corrs)


    def intra_correlations(self, _df, _well_col, _feat_cols, _group_cols):
        """
        Computes correlations within groups using well feature vectors.
        In this way we correlate each well pair within groups.

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
        corrs: pd.DataFrame
            The correlated data including well pairs, intra-group pairs, and correlations
        """

        groupdf = _df.groupby(_group_cols)

        # Retrieve group names
        gkeys = groupdf.groups.keys()

        # Groups with the features as columns (for efficiently computing corrleations)
        grouptdfs = {gkey: groupdf.get_group(gkey)[_feat_cols].T.copy() for gkey in gkeys}


        # Store correlations
        corrs = defaultdict(list)

        # Iterate through each group
        for gpair in gkeys:

            # Wells for the group
            group_well = groupdf.get_group(gpair)[_well_col]

            # Iterate through each well group combination and save the data
            for wpair in list(combinations(group_well, 2)):
                corrs["group0"].append(gpair)
                corrs["group1"].append(gpair)
                corrs["well0"].append(wpair[0])
                corrs["well1"].append(wpair[1])
                corrs["correlation"].append(
                    grouptdfs[gpair][wpair[0]].corr(
                        grouptdfs[gpair][wpair[1]]
                    )
                )

        return pd.DataFrame(corrs)
