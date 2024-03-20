from collections import defaultdict
from itertools import combinations, product

import pandas as pd


class CorrelateData:
    """
    Compute inter or intra group correlations between objects.
    """

    def __init__(self):
        pass

    def save_params(self, _groups, _cols, _drop_cols, _corrs):
        """
        Parameters
        ----------
        _groups: List(Tuple) or Tuple(List) or List or Tuple
            The values for each group (groups 0 and 1) for each column to save.

        _cols: List
            Columns to save for each of the two groups.

        _drop_cols: List
            Columns to not save.

        _corrs: Defaultdict(List)
            Corrlelation data storing correlations and column values for each data pair.

        Returns
        -------
        _corrs: Defaultdict(List)
            Corrlelation data storing correlations and column values for each data pair.
        """

        # Ensure the column value can be indexed if only one column is saved for the group for compatibility
        if (not isinstance(_groups[0], list)) and (not isinstance(_groups[0], tuple)):
            _groups = list(_groups)
            _groups[0], _groups[1] = [_groups[0]], [_groups[1]]

        # Iterate through each column and save the value for that group and column
        for idx, col in enumerate(_cols):
            if col in _drop_cols:
                continue
            _corrs[f"{col}__group0"].append(_groups[0][idx])
            _corrs[f"{col}__group1"].append(_groups[0][idx])

        return _corrs

    def inter_correlations(self, _df, _antehoc_group_cols, _feat_cols, _posthoc_group_cols, _drop_cols = []):
        """
        Computes correlations between two objects using both post hoc and ante groups.
        This is accomplished by computing correlations between objects in different ante groups (inter correlations) and between all possible post hoc correlation comparisons between those groups.

        Parameters
        ----------
        _df: pd.DataFrame
            Contains the features and group columns to use for correlating objects.

        _antehoc_group_cols: List(Objects)
            Columns to groupby before organizing by _posthoc_group_cols.

        _feat_col: List
            Feature columns.

        _posthoc_group_cols: List(Objects)
            Columns to groupby after organizing by _antehoc_group_cols.

        _drop_col: List
            Columns to not save in the final output.

        Returns
        -------
        corrs: pd.DataFrame
            The correlated data including columns from both the ante and post hoc groups less the dropped columns.
        """

        groupdf = _df.groupby(_antehoc_group_cols)

        # Retrieve keys for the ante group
        gkeys = groupdf.groups.keys()

        # Find all possible combinations of size 2 ante group keys
        apairs = list(combinations(gkeys, 2))

        # Store correlations
        corrs = defaultdict(list)

        # Iterate through each ante group combination
        for apair in apairs:

            # Extract the keys for the first post hoc group
            group0df = groupdf.get_group(apair[0]).copy()
            group0df = group0df.groupby(_posthoc_group_cols)[_feat_cols]
            group0_keys = group0df.groups.keys()

            # Extract the keys for the second post hoc group
            group1df = groupdf.get_group(apair[1]).copy()
            group1df = group1df.groupby(_posthoc_group_cols)[_feat_cols]
            group1_keys = group1df.groups.keys()

            # Iterate through each well group cartesian product and save the data
            for ppair in list(product(group0_keys, group1_keys)):

                # Store correlations
                corrs["correlation"].append(group0df.get_group(ppair[0]).iloc[0].corr(
                    group1df.get_group(ppair[1]).iloc[0]
                ))

                # Save the column data for both the ante and post hoc groups less the dropped columns
                corrs = self.save_params(apair, _antehoc_group_cols, _drop_cols, corrs)
                corrs = self.save_params(ppair, _posthoc_group_cols, _drop_cols, corrs)

        return pd.DataFrame(corrs)


    def intra_correlations(self, _df, _antehoc_group_cols, _feat_cols, _posthoc_group_cols, _drop_cols = []):
        """
        Computes correlations between two objects using both post hoc and ante groups.
        This is accomplished by computing correlations between objects only in the same ante group (intra correlations), but between different post hoc groups.

        Parameters
        ----------
        _df: pd.DataFrame
            Contains the features and group columns to use for correlating objects.

        _antehoc_group_cols: List(Objects)
            Columns to groupby before organizing by _posthoc_group_cols.

        _feat_col: List
            Feature columns.

        _posthoc_group_cols: List(Objects)
            Columns to groupby after organizing by _antehoc_group_cols.

        _drop_col: List
            Columns to not save in the final output.

        Returns
        -------
        corrs: pd.DataFrame
            The correlated data including columns from both the ante and post hoc groups less the dropped columns.
        """

        groupdf = _df.groupby(_antehoc_group_cols)

        # Retrieve keys for the ante group
        akeys = groupdf.groups.keys()

        # Store correlations
        corrs = defaultdict(list)

        # Iterate through each ante group combination
        for agroup in akeys:

            # Extract keys for poshoc group columns
            group = groupdf.get_group(agroup).copy()
            group = group.groupby(_posthoc_group_cols)[_feat_cols]
            group_keys = group.groups.keys()

            # Iterate through the combinations pairs of the groups
            for ppair in list(combinations(group_keys, 2)):

                # Store correlations
                corrs["correlation"].append(group.get_group(ppair[0]).iloc[0].corr(
                    group.get_group(ppair[1]).iloc[0]
                ))

                # Save the column data for both the ante and post hoc groups less the dropped columns
                corrs = self.save_params(ppair, _posthoc_group_cols, _drop_cols, corrs)
                corrs = self.save_params([agroup, agroup], _antehoc_group_cols, _drop_cols, corrs)

        return pd.DataFrame(corrs)
