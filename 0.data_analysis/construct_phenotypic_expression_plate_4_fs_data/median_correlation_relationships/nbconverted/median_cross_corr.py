#!/usr/bin/env python
# coding: utf-8

# # Determine expression relationships between constructs
# Correlations between aggregate replicates are compared

# ## Imports

# In[1]:


from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import pathlib
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path


# ## Find the git root Directory

# In[2]:


# Get the current working directory
cwd = Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")


# ## Seed and Generator for Reproducibility

# In[3]:


rnd_val = 0  # Random value for all seeds
rng = np.random.default_rng(seed=rnd_val)  # random number generator


# ## Define Paths

# In[4]:


# Change this filename when plate 4 is available
filename = "Plate_4_sc_norm_fs.parquet"
path = pathlib.Path(
    f"{root_dir}/nf1_painting_repo/3.processing_features/data/feature_selected_data/{filename}"
)

# Add the output path here:
output_path = Path("figures")

# Create the directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)


# ## Load the data

# In[5]:


platedf = pd.read_parquet(path)


# ## Specify data columns

# In[6]:


# The Column that corresponds to the concentration
conc_col = "Metadata_Concentration"

# The Column that corresponds to the construct
construct_col = "Metadata_siRNA"

meta_prefix = "Metadata"
# Columns that do not contain Metadata
not_meta_cols = [col for col in platedf.columns if meta_prefix not in col]


# ## Get cells that only contains NF1 siRNA constructs

# In[7]:


platedf[construct_col].fillna(value="None", inplace=True)

platefilt = platedf[construct_col].str.contains("NF1")

platedf = platedf.loc[platefilt]


# ## Column for filtering correlation data

# In[8]:


platedf["Metadata_group"] = platedf.apply(
    lambda row: f"{row['Metadata_Well']}_{row[conc_col]}_{row[construct_col]}", axis=1
)


# ## Aggregate well cells using the Median

# In[9]:


# Columns that should be used to take the median
median_cols = {col_name: "median" for col_name in not_meta_cols}

# Set metadata columns to lambda functions set to the first row
meta_cols = {
    col_name: lambda x: x.iloc[0]
    for col_name in platedf.columns
    if meta_prefix in col_name
}

# Combine the dictionaries
median_cols.update(meta_cols)

# Aggregate the plate data
platedf = platedf.groupby("Metadata_Well").agg(median_cols)


# ## Compute entire correlation matrix

# In[10]:


# Remove metadata columns and Compute all correlations for all cells
allcorr = platedf[not_meta_cols].T.corr()

# Set all of the correlation coeffients between the same cells as nans
allcorr.values[np.diag_indices_from(allcorr)] = np.nan


# ## Function for same and different construct correlations

# In[11]:


def get_correlations(same_construct):
    """
    Calculates the cross correlations between cells across wells using the same siRNA concentrations for either the same constructs or different constructs.

    Parameters
    ----------
    same_construct: Boolean
        Whether to consider different construct or the same constructs when calculating cross correlations between cells

    Returns
    -------
    transwell: Dictionary
    Contains the cell correlation vectors for each group
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

    # Holds correlations for siRNA constructs and concentrations
    transwell = defaultdict(list)

    for group in platedf["Metadata_group"].unique():

        # Find all cells that correspond to the specific (well, siRNA concentration, siRNA construct) group
        welldf = platedf.loc[(platedf["Metadata_group"] == group)]

        dfrow = welldf.iloc[0]

        # The well for this group
        well = dfrow["Metadata_Well"]

        # The concentration of the siRNA construct for this group
        conc = dfrow[conc_col]

        # The construct of the siRNA construct for this group
        construct = dfrow[construct_col]

        if same_construct:
            # Cells that are used to perform cross correlation. Only cells that have not been cross correlated with the same siRNA construct, the same siRNA concentration from different wells are considered.
            other_welldf = platedf.loc[
                (platedf[conc_col] == conc)
                & (platedf[construct_col] == construct)
                & (platedf["Metadata_Well"] != well)
                & ~(platedf["Metadata_Well"].isin(tried_wells))
            ]

        else:
            # Cells that are used to perform cross correlation. Cells with different siRNA constructs, and the same siRNA concentration from different wells are considered. Cells that have already been correlated are not tracked, since groups are compared independently
            other_welldf = platedf.loc[
                (platedf[conc_col] == conc)
                & (platedf[construct_col] != construct)
                & (platedf["Metadata_Well"] != well)
            ]

        num_cellsa = len(welldf)
        num_cellsb = len(other_welldf)

        # If there are no cells from either of these groups to cross correlate, the next group is considered
        if num_cellsa == 0 or num_cellsb == 0:
            continue

        # Get the indices corresponding to each cell for the two dataframes to compare, where the indices are also used to reference the columns
        welldf_idx = welldf.index
        other_welldf_idx = other_welldf.index

        # Find the correlation coefficients from the indices
        corr_df = allcorr.loc[welldf_idx, other_welldf_idx]

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

    return corr_data


# ## Get the correlations for the same constructs

# In[12]:


same_transwell = get_correlations(same_construct=True)

# Create a dataframe for the correlation data
same_construct = pd.DataFrame(same_transwell)


# ## Get the correlations for different constructs

# In[13]:


diff_transwell = get_correlations(same_construct=False)

# Concatenate all correlation values in one numpy array for each group
diff_construct = pd.DataFrame(diff_transwell)


# ## Combine the comparison data

# In[14]:


final_construct = pd.concat([same_construct, diff_construct], axis=0)


# ## Prepare data for plotting

# In[15]:


# Get unique siRNA construct concentrations
pos_covar = platedf[conc_col].unique()

# Get unique siRNA constructs
pos_const = platedf[construct_col].unique()

# Define the domain for plotting
xs = np.linspace(-1, 1, 500)

# Define correlation distributions in a dictionary
pos_comparisons = final_construct["comparison"].unique()
corr_dists = {
    "Same Construct Across Wells": pos_comparisons[0],
    "Different Construct Across Wells": pos_comparisons[1],
}


# ## Plot PDF Curves

# In[16]:


# Iterate through each covariate group (just concentration in this case)
for row, group in enumerate(pos_covar):

    # Iterate through each siRNA construct
    for const in pos_const:
        fig, ax = plt.subplots(figsize=(15, 11))

        # Define each combination to reference correlation distributions
        panel = (final_construct["construct"] == const) & (
            final_construct["concentration"] == group
        )

        df = final_construct.loc[panel]

        # Plot each distribution for a given panel
        for dlabel, comp in corr_dists.items():

            # Get the kde distribtion for the correlation data
            w_x = df.loc[df["comparison"] == comp]["pearsons_coef"].values
            w_density = gaussian_kde(w_x)
            w_y = w_density(xs)

            ax.plot(xs, w_y, label=dlabel)

        ax.set_xlabel("Pairwise Correlation")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.set_title(f"KDE of Pairwise Correlations: ({const} Construct at {group}nM)")
        fig.savefig(f"{output_path}/pdf_{const}_{group}.png")

# Show the plot
plt.show()
