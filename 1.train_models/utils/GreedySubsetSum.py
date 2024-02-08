import pandas as pd
from PlateTrainingSplits import PlateTrainingSplits


class GreedySubsetSum:

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

        # Determine the smalles reference category for sampling test wells
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

            # Determine the wells for the reference category
            if all((groupdf[cat] == min_cat[cat]).all() for cat in category_col):
                base_cat_wells = (
                    groupdf.nsmallest(_test_well_count, _cell_count_col)
                    [_well_col].tolist()
                )
                print(f"{len(base_cat_wells)} wells of {cat_num_wells} wells are test wells for reference group {cat}")

                continue

            # Sort the grouped data by well cell count
            groupdf = groupdf.sort_values(by=_cell_count_col, ascending=False)

            # Create a cumulative sum of the sorted well cell counts
            cell_count_cum_sum = f"{_cell_count_col}_cumsum"
            groupdf[cell_count_cum_sum] = groupdf[_cell_count_col].cumsum()

            # Total cell count for all wells in group
            tot_cell_count = groupdf[cell_count_cum_sum].iloc[-1]

            # Number of train-val cells after adding sorted wells to test set
            groupdf["group_trainval_count"] = tot_cell_count - groupdf[cell_count_cum_sum]

            # Wells if they have a higher train-val cell count
            groupdf = groupdf.loc[min_cat[_cell_count_col] <= groupdf["group_trainval_count"]]

            # Number of cells added to test set
            cat_num_test_wells = groupdf.shape[0]

            # Check that the number of test wells for the group isn't zero
            if cat_num_test_wells == 0:
                raise ValueError(f"The test well count of group {cat} is zero")

            # Add wells to test set
            test_wells.extend(groupdf[_well_col].tolist())

            """
            # Determine which wells should be test wells
            for well, test_cell_count in self.test_well_count(groupdf, _cell_count_col):
                if min_cat[_cell_count_col] <= (tot_cell_count - test_cell_count):
                    test_wells.append(well[_well_col])

                # If too many test wells are added to change the minority train-val group, then stop adding test wells
                else:
                    break
            """

            print(f"{groupdf.shape[0]} wells of {cat_num_wells} wells are test wells for group {cat}")

        return test_wells
