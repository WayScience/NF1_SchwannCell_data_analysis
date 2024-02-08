import pandas as pd
from PlateTrainingSplits import PlateTrainingSplits


class GreedySubsetSum:

    def __init__(self):
        pass

    def test_well_count(self, _welldf, _cell_count_col, _ascending=False):
        """
        Parameters
        ----------
        _welldf: Pandas Dataframe
            Well data with unique wells represented as rows.

        _cell_count_col: String
            The cell count column name.

        _ascending: Boolean
            Order to sort cell counts from wells.

        Yields
        ------
        well: Pandas Series
            Data for one well.

        test_cell_count: Integer
            The number of cells added to the test well set.
        """

        test_cell_count = 0
        welldf = _welldf.sort_values(by=_cell_count_col, ascending=_ascending)
        for _, well in welldf.iterrows():
            test_cell_count += well[_cell_count_col]
            yield well, test_cell_count


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

            # Total cell count for all wells in group
            tot_cell_count = groupdf[_cell_count_col].sum()

            # Initial number of test wells before adding groups test wells
            num_test_wells0 = len(test_wells)

            # Determine which wells should be test wells
            for well, test_cell_count in self.test_well_count(_welldf, _cell_count_col):
                if min_cat[_cell_count_col] <= (tot_cell_count - test_cell_count):
                    test_wells.append(well[_well_col])

                # If too many test wells are added to change the minority train-val group, then stop adding test wells
                else:
                    break

            print(f"{len(test_wells) - num_test_wells0} wells of {cat_num_wells} wells are test wells for group {cat}")

        return test_wells
