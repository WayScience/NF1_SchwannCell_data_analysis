from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


class CreateCorrelations:
    def __init__(self, platedf, aggregated):
        """
        Parameters
        ----------
        platedf: pandas Dataframe
        Contains all of the siRNA plate data to be analyzed

        aggregated: Boolean
        Whether or not the data is aggregated
        """

        # The Column that corresponds to the concentration
        self.conc_col = "Metadata_Concentration"

        # The Column that corresponds to the construct
        self.construct_col = "Metadata_siRNA"

        # The prefix to use for identifying Metadata columns
        self.meta_prefix = "Metadata"

        # Specify if the data is aggregated or not
        self.aggregated = aggregated

        # Specify the name of the columns depending on if the data is aggregated or not
        if aggregated:
            self.first_entry_name = "first_well"
            self.second_entry_name = "second_well"

        else:
            self.first_entry_name = "first_cell"
            self.second_entry_name = "second_cell"

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

        pos_const = self.platedf[self.construct_col].unique()

        # Get all unique pairs of contruct type comparisons
        conc_comp = list(combinations(pos_const, 2))

        # Iterate through each unique contruct pair to compute across well construct correlations
        for consts in conc_comp:
            consts = list(consts)

            # Filter the plate dataframe with only data from the pair of constructs
            df = self.platedf.loc[self.platedf[self.construct_col].isin(list(consts))]

            diff_construct = self.aggregate_correlations(same_construct=False, constdf=df)
            self.final_construct.append(diff_construct)

        # Iterate through each unique contruct to compute within construct correlations
        for consts in pos_const:
            df = self.platedf.loc[self.platedf[self.construct_col] == consts]

            # calculate the correlation data for the same construct
            same_construct = self.aggregate_correlations(same_construct=True, constdf=df)

            self.final_construct.append(same_construct)

        self.final_construct = pd.concat(self.final_construct, axis = 0)

    def preprocess_data(self):
        """
        Preprocess the data so that each of the constructs are compared for each group
        """

        # Fill None values with no_treatment
        self.platedf[self.construct_col].fillna(value="no_treatment", inplace=True)

        # Find the possible concentrations
        pos_conc = self.platedf[self.conc_col].unique()

        # Remove zero from the possible concentrations
        pos_conc = pos_conc[np.nonzero(pos_conc)]

        # Dataframe with only no_treatment constructs
        no_treatdf = self.platedf.loc[self.platedf[self.construct_col] == "no_treatment"]

        # Store the dataframes to be concatenated so that no_treatement type compared for each concentration
        duplicates = [self.platedf]

        for conc in pos_conc:
            # Create duplicate values for the no_treatment type and set the values to the concentrations to compare
            no_treat_concdf = no_treatdf.copy()
            no_treat_concdf[self.conc_col] = conc
            duplicates.append(no_treat_concdf)

        # Concatenate the duplicate dataframes
        self.platedf = pd.concat(duplicates)

        # Remove entries that correspond to a zero concentration
        self.platedf = self.platedf.loc[self.platedf[self.conc_col] > 0]

        # Column for filtering correlation data
        self.platedf["Metadata_group"] = self.platedf.apply(
            lambda row: f"{row['Metadata_Well']}_{row[self.conc_col]}_{row[self.construct_col]}",
            axis=1,
        )

        if self.aggregated:
            # Set the well to the index of the dataframe if aggregated data is used
            self.platedf.set_index("Metadata_Well", inplace=True, drop=False)

    def compute_correlations(self):
        """
        Compute the correlations for all of the plate feature data
        """

        # Remove redundant samples
        self.allcorr = self.platedf[~self.platedf.index.duplicated(keep='first')]

        # Remove metadata columns and Compute all correlations for all samples
        self.allcorr = self.allcorr[self.not_meta_cols].T.corr()

        # Set all of the correlation coeffients between the same samples as nans
        self.allcorr.values[np.diag_indices_from(self.allcorr)] = np.nan

    def aggregate_correlations(self, same_construct, constdf):
        """
        Calculates the cross correlations between wells using the same siRNA concentrations for either the same constructs or different constructs.
        In this class, only two constructs, or one pair, is compared by filtering constdf prior to calling this function

        Parameters
        ----------
        same_construct: Boolean
            Whether to consider different construct or the same constructs when calculating cross correlations between samples

        constdf: pandas Dataframe
            The processed data containing all constructs

        Returns
        -------
        Pandas Dataframe
            Contains the sample correlations and other correlation and plate metadata
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

            # Find all wells that correspond to the specific (well, siRNA concentration, siRNA construct) group
            welldf = constdf.loc[(constdf["Metadata_group"] == group)]

            dfrow = welldf.iloc[0]

            # The well for this group
            well = dfrow["Metadata_Well"]

            # The concentration of the siRNA construct for this group
            conc = dfrow[self.conc_col]

            # The construct of the siRNA construct for this group
            construct = dfrow[self.construct_col]

            if same_construct:
                # Samples that are used to perform cross correlation. Only wells that have not been cross correlated with the same siRNA construct, the same siRNA concentration from different wells are considered.
                other_welldf = constdf.loc[
                    (constdf[self.conc_col] == conc)
                    & (constdf[self.construct_col] == construct)
                    & (constdf["Metadata_Well"] != well)
                    & ~(constdf["Metadata_Well"].isin(tried_wells)) # Remove duplicate correlation comparisons (eg. if wells A1 and B1 have been compared, don't compare B1 and A1)
                ]

                # Use the same construct in the results
                constructs = [construct] * 2

            else:
                # Samples that are used to perform cross correlation. Samples with different siRNA constructs, and the same siRNA concentration from different wells are considered. Samples that have already been correlated are not tracked, since groups are compared independently
                other_welldf = constdf.loc[
                    (constdf[self.conc_col] == conc)
                    & (constdf[self.construct_col] != construct)
                    & (constdf["Metadata_Well"] != well)
                ]

            num_samplea = len(welldf)
            num_sampleb = len(other_welldf)

            # If there are no samples from either of these groups to cross correlate, the next group is considered
            if (num_samplea == 0) or (num_sampleb == 0):
                continue

            # Get the indices corresponding to each well for the two dataframes to compare, where the indices are also used to reference the columns
            welldf_idx = welldf.index
            other_welldf_idx = other_welldf.index

            # Find the correlation coefficients from the indices
            corr_df = self.allcorr.loc[welldf_idx, other_welldf_idx]

            # Convert the correlation values and flatten them
            corr_vec = corr_df.values.flatten()

            # Number of correlation values
            num_corr = corr_vec.shape[0]

            # Record the correlation data
            corr_data[self.first_entry_name].extend(welldf_idx.repeat(num_sampleb).tolist())
            corr_data[self.second_entry_name].extend(other_welldf_idx.tolist() * num_samplea)
            corr_data["pearsons_coef"].append(corr_vec)
            corr_data["first_construct"] += [constructs[0]] * num_corr
            corr_data["second_construct"] += [constructs[1]] * num_corr
            corr_data["concentration"] += [conc] * num_corr

            # Keep track of the wells used for the correlation if the same construct features are being compared
            tried_wells.append(well)

        # Combine the correlation data
        corr_data["comparison"] = [comp] * len(corr_data[self.first_entry_name])
        corr_data["pearsons_coef"] = np.concatenate(corr_data["pearsons_coef"])

        return pd.DataFrame(corr_data)
