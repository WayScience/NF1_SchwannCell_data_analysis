from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


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

        self.platedf = platedf

        self.preprocess_data()

        self.compute_correlations()

        self.merge_constructs()

    def merge_constructs(self):
        """
        Merges the correlation data and metadata across comparisons

        Returns
        -------
        final_construct: pandas Dataframe
        The final construct from all comparisons including correlation data and metadata
        """

        same_construct = self.aggregate_correlations(same_construct=True)
        diff_construct = self.aggregate_correlations(same_construct=False)

        self.final_construct = pd.concat([same_construct, diff_construct], axis=0)

        return self.final_construct

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

    def compute_correlations(self):
        """
        Compute the correlations for all of the plate feature data
        """

        # Remove metadata columns and Compute all correlations for all cells
        self.allcorr = self.platedf[self.not_meta_cols].T.corr()

        # Set all of the correlation coeffients between the same cells as nans
        self.allcorr.values[np.diag_indices_from(self.allcorr)] = np.nan

    def aggregate_correlations(self, same_construct):
        """
        Calculates the cross correlations between cells across wells using the same siRNA concentrations for either the same constructs or different constructs.

        Parameters
        ----------
        same_construct: Boolean
            Whether to consider different construct or the same constructs when calculating cross correlations between cells

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

        # Groups that have already been considered
        tried_wells = []

        # Dictionary for reconstructing the correlation data
        corr_data = defaultdict(list)

        for group in self.platedf["Metadata_group"].unique():

            # Find all cells that correspond to the specific (well, siRNA concentration, siRNA construct) group
            welldf = self.platedf.loc[(self.platedf["Metadata_group"] == group)]

            dfrow = welldf.iloc[0]

            # The well for this group
            well = dfrow["Metadata_Well"]

            # The concentration of the siRNA construct for this group
            conc = dfrow[self.conc_col]

            # The construct of the siRNA construct for this group
            construct = dfrow[self.construct_col]

            if same_construct:
                # Cells that are used to perform cross correlation. Only cells that have not been cross correlated with the same siRNA construct, the same siRNA concentration from different wells are considered.
                other_welldf = self.platedf.loc[
                    (self.platedf[self.conc_col] == conc)
                    & (self.platedf[self.construct_col] == construct)
                    & (self.platedf["Metadata_Well"] != well)
                    & ~(self.platedf["Metadata_Well"].isin(tried_wells))
                ]

            else:
                # Cells that are used to perform cross correlation. Cells with different siRNA constructs, and the same siRNA concentration from different wells are considered. Cells that have already been correlated are not tracked, since groups are compared independently
                other_welldf = self.platedf.loc[
                    (self.platedf[self.conc_col] == conc)
                    & (self.platedf[self.construct_col] != construct)
                    & (self.platedf["Metadata_Well"] != well)
                ]

            num_cellsa = len(welldf)
            num_cellsb = len(other_welldf)

            # If there are no cells from either of these groups to cross correlate, the next group is considered
            if (num_cellsa == 0) or (num_cellsb == 0) :
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
            corr_data["first_well"].extend(welldf_idx.repeat(num_cellsb).tolist())
            corr_data["second_well"].extend(other_welldf_idx.tolist() * num_cellsa)
            corr_data["pearsons_coef"].append(corr_vec)
            corr_data["construct"] += [construct] * num_corr
            corr_data["concentration"] += [conc] * num_corr

            # Keep track of the wells used for the correlation
            tried_wells.append(well)

        # Combine the correlation data
        corr_data["comparison"] = [comp] * len(corr_data["first_well"])
        corr_data["pearsons_coef"] = np.concatenate(corr_data["pearsons_coef"])

        return pd.DataFrame(corr_data)

    def plot_correlations(self, output_path):

        # Get unique siRNA concentrations and construct pairs
        sirna_pairs = set(zip(self.final_construct['concentration'], self.final_construct['construct']))

        # Define the domain for plotting
        xs = np.linspace(-1, 1, 500)

        # Define the label and the comparison to plot
        corr_map = {
            "same_construct": "Same Construct Across Wells",
            "different_construct": "Different Construct Across Wells"
        }

        # Iterate through concentrations and constructs
        for group, const in sirna_pairs:
            fig, ax = plt.subplots(figsize=(15, 11))

            # Define each combination to reference correlation distributions
            panel = (self.final_construct["construct"] == const) & (
                    self.final_construct["concentration"] == group
                    )

            df = self.final_construct.loc[panel]

            # Get the unique distribution for each panel
            corr_dists = df["comparison"].unique()

            # Plot each distribution for a given panel
            for comp in corr_dists:

                dlabel = corr_map[comp]

                # Get the kde distribtion for the correlation data
                w_x = df.loc[df["comparison"] == comp]["pearsons_coef"].values
                w_density = gaussian_kde(w_x)
                w_y = w_density(xs)

                ax.plot(xs, w_y, label=dlabel)

            ax.set_xlabel("Pairwise Correlation")
            ax.set_ylabel("Probability Density")
            ax.legend()
            ax.set_title(
                    f"KDE of Pairwise Correlations: ({const} Construct at {group}nM)"
                    )
            fig.savefig(f"{output_path}/pdf_{const}_{group}.png")

        # Show the plot
        plt.show()
