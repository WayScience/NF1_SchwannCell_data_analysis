r"""°°°
# Determine expression relationships between constructs
Correlate post feature selection well-aggregated morphology features across the same concentrations.
°°°"""
#|%%--%%| <4C20coDtNB|Eg9pPBZcbM>
r"""°°°
## Imports
°°°"""
#|%%--%%| <Eg9pPBZcbM|5dKOCfosTM>

import pathlib
import sys

import pandas as pd

# |%%--%%| <5dKOCfosTM|2OsaGp4f3S>
r"""°°°
## Find the root of the git repo on the host system
°°°"""
# |%%--%%| <2OsaGp4f3S|DUsPsvZlzR>

# Get the current working directory
cwd = pathlib.Path.cwd()

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

# |%%--%%| <DUsPsvZlzR|eSQXraMRCI>
r"""°°°
## Define paths
°°°"""
# |%%--%%| <eSQXraMRCI|cZx8xNCyZq>
r"""°°°
### Input paths
°°°"""
# |%%--%%| <cZx8xNCyZq|xdgUvt0md2>

# Path to correlation class
sys.path.append(
    f"{root_dir}/0.data_analysis/utils"
)

# Class for calculating correlations
from CorrelateData import CorrelateData

platedf_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles/Plate_4_bulk_camerons_method.parquet").resolve(strict=True)
platedf = pd.read_parquet(platedf_path)

# |%%--%%| <xdgUvt0md2|lfpMNf0oYw>
r"""°°°
### Output paths
°°°"""
# |%%--%%| <lfpMNf0oYw|z3syxsoQ8C>

data_path = pathlib.Path("construct_correlation_data")
data_path.mkdir(parents=True, exist_ok=True)

#|%%--%%| <z3syxsoQ8C|1KIVewAtgq>
r"""°°°
## Label untreated cells
°°°"""
#|%%--%%| <1KIVewAtgq|7WAqbP1I3I>

platedf["Metadata_siRNA"].fillna("No Construct", inplace=True)
platedf.dropna(inplace=True)

# |%%--%%| <7WAqbP1I3I|rB32cuv8gh>

meta_cols = platedf.filter(like="Metadata").columns
feat_cols = platedf.drop(columns=meta_cols).columns

# |%%--%%| <rB32cuv8gh|CDbIQ6BTAP>
r"""°°°
## Compute Correlations
°°°"""
#|%%--%%| <CDbIQ6BTAP|QCAMpGgBpH>

# Store correlations
corrdfs = []

cp = CorrelateData()

# Include cells with no construct treatment
platedfz = platedf.loc[(platedf["Metadata_Concentration"] == 0.0) & (platedf["Metadata_genotype"] == "WT")].copy()

# Compute correlations for each concentration
for conc, concdf in platedf.groupby("Metadata_Concentration"):

    # Correlates all wells between the same siRNA-genotype combinations
    corrdfs.append(cp.intra_correlations(
        _df = concdf.reset_index(drop=True).copy(),
        _antehoc_group_cols = ["Metadata_siRNA", "Metadata_genotype"],
        _feat_cols = feat_cols,
        _posthoc_group_cols = ["Metadata_Well"],
    )
    )

    # Save the concentration and type of comparison
    corrdfs[-1]["Metadata_Concentration__group0"] = conc
    corrdfs[-1]["Metadata_Concentration__group1"] = conc

    # Don't compute correlations for cells not treated with a construct
    # The cells in this group are already compared to the constructs at every other concentration
    if conc == 0.0:
        continue

    # Include the cells not treated with a construct in the correlation comparisons
    concdf = pd.concat([
        concdf,
        platedfz.copy()
    ], axis=0)

    # Correlates all wells between different siRNA-well combinations
    corrdfs.append(cp.inter_correlations(
        _df = concdf.reset_index(drop=True).copy(),
        _antehoc_group_cols = ["Metadata_siRNA", "Metadata_Concentration"],
        _feat_cols = feat_cols,
        _posthoc_group_cols = ["Metadata_Well"],
    )
    )

# |%%--%%| <QCAMpGgBpH|9cRnSR2wTC>
r"""°°°
## Store Correlation Data
°°°"""
# |%%--%%| <9cRnSR2wTC|HD42qd1jUX>

corrdfs = pd.concat(corrdfs, axis=0)
corrdfs.to_parquet(f"{data_path}/plate_4_sc_post_fs_agg_well_correlations.parquet")

# |%%--%%| <HD42qd1jUX|ZWUk9bCL8r>

corrdfs.head()
