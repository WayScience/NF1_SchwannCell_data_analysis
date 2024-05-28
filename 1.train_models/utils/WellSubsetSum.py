from collections import defaultdict

import numpy as np
import pandas as pd
from PlateTrainingSplits import PlateTrainingSplits


class WellSubsetSum:
    """
    Select wells for the test dataset by maximizing the number of cells in the train-validation set for a given number of wells.
    """

    def __init__(self):
        pass

    def update_test_wells(self, _welldf, _category_col, _well_col, _cell_count_col, _test_well_count):
        """
        Parameters
        ----------
        _welldf: Pandas Dataframe
            Well data with unique wells represented as rows.

        _category_col: List of Strings or String
            The categories to represent equally in the cell population.

        _well_col: String
            Well column name.

        _cell_count_col: String
            The cell count column name.

        _test_well_count: Integer
            Number of wells to sample for testing.

        Returns
        -------
        test_wells: List of Strings
            Test well names.
        """

        # Check if all of the cell counts are integers.
        if not pd.api.types.is_integer_dtype(_welldf[_cell_count_col]):
            raise TypeError(f"{_cell_count_col} column does not only contain integers")

        # Check if all of the integers are greater than zero
        if (_welldf[_cell_count_col] <= 0).all():
            raise ValueError(f"{_cell_count_col} column contains only positive integers")

        plate_split = PlateTrainingSplits()

        catdf = _welldf.groupby(_category_col)

        # Determine the smallest category (reference category) for sampling test wells
        min_cat = plate_split.sum_train_val_category_cell_counts(
            catdf,
            _category_col,
            _cell_count_col,
            _test_well_count
        )

        test_wells = []

        # Make the categorie(s) iterable
        if not isinstance(_category_col, list):
            category_col = [_category_col]

        # Iterate through each group
        for cat, groupdf in catdf:

            # Number of wells in group
            cat_num_wells = groupdf.shape[0]

            # Determine the test wells for the reference category
            if all((groupdf[ref_cat] == min_cat[ref_cat]).all() for ref_cat in category_col):
                base_cat_wells = (
                    groupdf.nsmallest(_test_well_count, _cell_count_col)
                    [_well_col].tolist()
                )

                test_wells.extend(base_cat_wells)

                print(f"{len(base_cat_wells)} wells of {cat_num_wells} wells are test wells for reference group {cat}")

                continue

            # Cumulative number of cells for this group
            tot_cell_count = groupdf[_cell_count_col].sum()

            # The Maximum number of cells (capacity) allowed in the test set
            # until this group becomes the minority group for the train-validation set
            max_test_size = tot_cell_count - min_cat[_cell_count_col]

            # The greatest number of cells using entire wells for the number
            # of of wells (i) at capacity w
            test_well_count = {i: defaultdict(int) for i in np.arange(cat_num_wells + 1)}

            # The well added, if any, at i and capacity w
            test_well_idx = {i: defaultdict(list) for i in np.arange(cat_num_wells + 1)}

            # Iterate through each well
            for df_idx, (_, well) in enumerate(groupdf.iterrows()):

                # Increment all indices by 1 to compute the base case in the first iteration
                df_idx += 1

                # Iterate through all possible capacities
                # Each well must contain at least one cell
                for w in np.arange(1, max_test_size + 1):

                    # The largest number of wells at the current cell capacity (w) is
                    # either at the previous largest number of cells,
                    # or when adding the next well. However, w
                    # cannot be less than the number of cells added from the well.
                    test_well_count[df_idx][w] = test_well_count[df_idx - 1][w]
                    well_val = well[_cell_count_col]
                    if well_val <= w:
                        if well_val + test_well_count[df_idx - 1][w - well_val] > test_well_count[df_idx][w]:
                            # Increment the number of cells if adding the well
                            test_well_count[df_idx][w] = well_val + test_well_count[df_idx - 1][w - well_val]
                            # Track the 0-indexed location of the well
                            test_well_idx[df_idx][w].append(df_idx - 1)


            # Reconstruct the solution (the maximum subset of wells
            # in the test set from this group)
            # Start from the number of wells and number of cells that give the
            # largest number of cells (the optimal value)
            idx = cat_num_wells
            w = max_test_size
            test_wells_int_idx = []

            # Perform the reconstruction to find the test wells indices
            while idx > 0 and w > 0:
                test_well_items = test_well_idx[idx][w]
                if test_well_items:
                    test_wells_int_idx.extend(test_well_items)
                    w -= groupdf.iloc[idx - 1][_cell_count_col]
                idx -= 1

            # Store the names of the wells
            test_wells.extend(groupdf.iloc[test_wells_int_idx][_well_col].tolist())

            # Check that the number of test wells for the group isn't zero
            if len(test_wells) == 0:
                raise ValueError(f"The test well count of group {cat} is zero")

            print(f"{len(test_wells)} wells of {cat_num_wells} wells are test wells for group {cat}")

        return test_wells
