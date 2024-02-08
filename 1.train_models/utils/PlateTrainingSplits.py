class PlateTrainingSplits:

    def __init__(self):
        pass

    def sum_train_val_category_cell_counts(self, _catdf, _category_col, _cell_count_col, _test_well_count):
        """
        Parameters
        ----------
        _catdf: DataFrameGroupBy
            Well data with unique wells grouped by _category_col.

        _category_col: List of Strings or String
            The categories to represent equally in the cell population.

        _cell_count_col: String
            The cell count column name.

        _test_well_count: Integer
            Number of wells to sample for testing.

        Returns
        -------
        Pandas Series
            Data about the smallest (reference) category.
        """

        # Calculates the size of the possible train-val set for each category =
        # [the total number of cells - the number of cells in the smallest wells (potential test wells)]
        def well_category_cell_count(well_data):

            # Sorts the well cell count
            sorted_well_data = well_data.sort_values(ascending=False)

            # Exclude the smallest test_well_count values
            min_cell_count_wells = sorted_well_data.iloc[_test_well_count:]

            # Calculate the sum of the remaining values
            return min_cell_count_wells.sum()

        # Calculates the number of wells in each category
        well_countsdf = (
            _catdf[_cell_count_col]
            .size()
            .reset_index(name="Metadata_number_wells")
        )

        # Throw an error if the number of wells is less than or equal to the number of of requested test wells
        if (well_countsdf["Metadata_number_wells"] <= _test_well_count).any():
            raise ValueError(f"The test well count of ({_test_well_count}) exceeds, or is equal to, the number of wells for at least one category in {_category_col}")

        # Calculate the maximum train-val set size for each category
        min_train_val_cats = (
            _catdf
            .agg({_cell_count_col: well_category_cell_count})
            .reset_index()
        )

        # Category with the smallest value
        return min_train_val_cats.loc[
            min_train_val_cats[_cell_count_col].idxmin()
        ]
