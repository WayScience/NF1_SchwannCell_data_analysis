from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


class CreateCorrelations:
    def __init__(self, platedf):
        """
        Parameters
        ----------
        platedf: pandas Dataframe
        Contains all of the siRNA plate data to be analyzed

        aggregate: Boolean
        Whether or not to aggregate the data using the median
        """

        # The Column that corresponds to the concentration
        self.conc_col = "Metadata_Concentration"

        # The Column that corresponds to the construct
        self.construct_col = "Metadata_siRNA"

        # The prefix to use for identifying Metadata columns
        self.meta_prefix = "Metadata"

        # Columns that do not contain Metadata
        self.not_meta_cols = [
            col for col in platedf.columns if self.meta_prefix not in col
        ]

        self.final_construct = []

        self.platedf = platedf

        self.preprocess_data()

        self.compute_correlations()

        self.merge_constructs()

    def merge_constructs(self):
        """
        Merges the correlation data and metadata across comparisons
        """

        conc_comp = list(combinations(self.platedf[self.construct_col].unique(), 2))

        for comp in conc_comp:
            comp = list(comp)

            df = self.platedf.loc[self.platedf[self.construct_col].isin(list(comp))]

            same_construct = self.aggregate_correlations(same_construct=True, constdf=df)

            if "no_treatment" not in comp:
                diff_construct = self.aggregate_correlations(same_construct=False, constdf=df)
                self.final_construct.append(diff_construct)

            self.final_construct.append(same_construct)

        self.final_construct = pd.concat(self.final_construct, axis = 0)

    def preprocess_data(self):
        """
        Preprocess the data so that the constructs are compared for each type of comparison
        """

        # Fill None values with no_treatment
        self.platedf[self.construct_col].fillna(value="no_treatment", inplace=True)

        # Column for filtering correlation data
        self.platedf["Metadata_group"] = self.platedf.apply(
            lambda row: f"{row['Metadata_Well']}_{row[self.conc_col]}_{row[self.construct_col]}",
            axis=1,
        )

        # Set the well to the index of the dataframe
        self.platedf.set_index("Metadata_Well", inplace=True, drop=False)

    def compute_correlations(self):
        """
        Compute the correlations for all of the plate feature data
        """

        # Remove metadata columns and Compute all correlations for all cells
        self.allcorr = self.platedf[self.not_meta_cols].T.corr()

        # Set all of the correlation coeffients between the same cells as nans
        self.allcorr.values[np.diag_indices_from(self.allcorr)] = np.nan

    def aggregate_correlations(self, same_construct, constdf):
        """
        Calculates the cross correlations between cells across wells using the same siRNA concentrations for either the same constructs or different constructs.

        Parameters
        ----------
        same_construct: Boolean
            Whether to consider different construct or the same constructs when calculating cross correlations between cells

        constdf: pandas Dataframe
            The processed data containing all constructs

        Returns
        -------
        Pandas Dataframe
            Contains the cell correlations and other correlation and plate metadata
        """

        # Specify the comparison type for tidy long format reconstruction
        if same_construct:
            comp = "same_construct"

        else:
            comp = "different_construct"

            # If the constructs are different sort them for the results
            constructs = np.sort(constdf[self.construct_col].unique())

        # Groups that have already been considered
        tried_wells = []

        # Dictionary for reconstructing the correlation data
        corr_data = defaultdict(list)

        for group in constdf["Metadata_group"].unique():

            # Find all cells that correspond to the specific (well, siRNA concentration, siRNA construct) group
            welldf = constdf.loc[(constdf["Metadata_group"] == group)]

            dfrow = welldf.iloc[0]

            # The well for this group
            well = dfrow["Metadata_Well"]

            # The concentration of the siRNA construct for this group
            conc = dfrow[self.conc_col]

            # The construct of the siRNA construct for this group
            construct = dfrow[self.construct_col]


            if same_construct:
                # Cells that are used to perform cross correlation. Only cells that have not been cross correlated with the same siRNA construct, the same siRNA concentration from different wells are considered.
                other_welldf = constdf.loc[
                    (constdf[self.conc_col] == conc)
                    & (constdf[self.construct_col] == construct)
                    & (constdf["Metadata_Well"] != well)
                    & ~(constdf["Metadata_Well"].isin(tried_wells)) # Remove duplicate correlation comparisons (eg. if wells A1 and B1 have been compared, don't compare B1 and A1)
                ]

                # Use the same construct in the results
                constructs = [construct] * 2

            else:
                # Cells that are used to perform cross correlation. Cells with different siRNA constructs, and the same siRNA concentration from different wells are considered. Cells that have already been correlated are not tracked, since groups are compared independently
                other_welldf = constdf.loc[
                    (constdf[self.conc_col] == conc)
                    & (constdf[self.construct_col] != construct)
                    & (constdf["Metadata_Well"] != well)
                ]

            num_wellsa = len(welldf)
            num_wellsb = len(other_welldf)

            # If there are no wells from either of these groups to cross correlate, the next group is considered
            if (num_wellsa == 0) or (num_wellsb == 0):
                continue

            # Get the indices corresponding to each cell for the two dataframes to compare, where the indices are also used to reference the columns
            welldf_idx = welldf.index
            other_welldf_idx = other_welldf.index

            # Find the correlation coefficients from the indices
            corr_df = self.allcorr.loc[welldf_idx, other_welldf_idx]

            # Convert the correlation values and flatten them
            corr_vec = corr_df.values.flatten()

            # Number of correlation values
            num_corr = corr_vec.shape[0]

            # Record the correlation data
            corr_data["first_well"].extend(welldf_idx.repeat(num_wellsb).tolist())
            corr_data["second_well"].extend(other_welldf_idx.tolist() * num_wellsa)
            corr_data["pearsons_coef"].append(corr_vec)
            corr_data["first_construct"] += [constructs[0]] * num_corr
            corr_data["second_construct"] += [constructs[1]] * num_corr
            corr_data["concentration"] += [conc] * num_corr

            # Keep track of the wells used for the correlation
            tried_wells.append(well)

        # Combine the correlation data
        corr_data["comparison"] = [comp] * len(corr_data["first_well"])
        corr_data["pearsons_coef"] = np.concatenate(corr_data["pearsons_coef"])

        return pd.DataFrame(corr_data)
