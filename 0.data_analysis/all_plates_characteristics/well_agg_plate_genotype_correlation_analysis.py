r"""°°°
# Well-Aggregated Plate and Genotype Correlation Analysis
Correlations between groups defined by genotype and plate are determined to understand the similarities between group morphologies.
These correlations are computed between cell morphologies aggregated to the well level.
°°°"""
#|%%--%%| <KGGUnjMQyM|CGb0BVUMWf>

import pathlib
import sys

import pandas as pd

# Path to correlation class
sys.path.append(
    "../utils"
)

# Class for calculating correlations
from CorrelateData import CorrelateData

#|%%--%%| <CGb0BVUMWf|kcR7QgN82F>
r"""°°°
## Find the root of the git repo on the host system
°°°"""
# |%%--%%| <kcR7QgN82F|T1ixkk0ex3>

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

#|%%--%%| <T1ixkk0ex3|jS6asi9T8F>
r"""°°°
# Inputs
°°°"""
#|%%--%%| <jS6asi9T8F|fv2xmp9AeZ>

data_path = pathlib.Path(root_dir / "nf1_painting_repo/3.processing_features/data/single_cell_profiles").resolve(strict=True)

plate3df_path = pathlib.Path(root_dir / data_path / "Plate_3_bulk_camerons_method.parquet").resolve(strict=True)
plate3pdf_path = pathlib.Path(root_dir / data_path / "Plate_3_prime_bulk_camerons_method.parquet").resolve(strict=True)
plate5df_path = pathlib.Path(root_dir / data_path / "Plate_5_bulk_camerons_method.parquet").resolve(strict=True)

plate3df = pd.read_parquet(plate3df_path)
plate3pdf = pd.read_parquet(plate3pdf_path)
plate5df = pd.read_parquet(plate5df_path)

#|%%--%%| <fv2xmp9AeZ|0xXYIze28O>
r"""°°°
# Outputs
°°°"""
#|%%--%%| <0xXYIze28O|pZMK12ZIUc>

plate_correlation_path = pathlib.Path("construct_correlation_data/well_agg_plate_genotype_correlations.parquet")
plate_correlation_path.mkdir(parents=True, exist_ok=True)

#|%%--%%| <pZMK12ZIUc|6y63uIAumE>
r"""°°°
# Process Bulk Plate Data
°°°"""
#|%%--%%| <6y63uIAumE|6LdIFMuOwi>
r"""°°°
## Combine data
Concat plate data and retain common columns.
°°°"""
#|%%--%%| <6LdIFMuOwi|F3pA4Cv4nE>

plates_cols = plate3df.columns.intersection(plate3pdf.columns).intersection(plate5df.columns)
platesdf = pd.concat([plate3df, plate3pdf, plate5df], axis=0)
platesdf = platesdf[plates_cols]

#|%%--%%| <F3pA4Cv4nE|7F2REC8Dbk>

# Morphology and metadata columns
morph_cols = [col for col in platesdf.columns if "Metadata" not in col]
meta_cols = platesdf.columns.difference(morph_cols)

#|%%--%%| <7F2REC8Dbk|YFdbO4KkoA>
r"""°°°
# Correlate wells
Wells are correlated between plate and genotype.
°°°"""
#|%%--%%| <YFdbO4KkoA|YPxjtndLAN>

cd = CorrelateData()
correlationsdf = []

#|%%--%%| <YPxjtndLAN|FlFnFIA36R>

cd.intra_correlations(
    _df=plate3df.loc[plate3df["Metadata_genotype"] == "WT"].copy(),
    _antehoc_group_cols=["Metadata_Plate", "Metadata_genotype"],
    _feat_cols=morph_cols,
    _posthoc_group_cols=["Metadata_Well"],
    _drop_cols=["Metadata_Well"]
)

#|%%--%%| <FlFnFIA36R|ddISX0Yw8M>
r"""°°°
## Well Correlations (same genotypes different plates)
°°°"""
#|%%--%%| <ddISX0Yw8M|Du6jPm5xP5>

for genotype in platesdf["Metadata_genotype"].unique():

    correlation_params = {
    }

    correlationsdf.append(
        cd.inter_correlations(
            _df=platesdf.loc[platesdf["Metadata_genotype"] == genotype].copy(),
            _antehoc_group_cols=["Metadata_Plate"],
            _feat_cols=morph_cols,
            _posthoc_group_cols=["Metadata_Well", "Metadata_genotype"],
            _drop_cols=["Metadata_Well"]
        )
    )

#|%%--%%| <Du6jPm5xP5|jkWJ9qYkUw>
r"""°°°
## Well Correlations (different genotypes and all possible plates)
°°°"""
#|%%--%%| <jkWJ9qYkUw|bfMQ1sVcMz>

correlationsdf.append(
    cd.inter_correlations(
        _df=platesdf.copy(),
        _antehoc_group_cols=["Metadata_genotype"],
        _feat_cols=morph_cols,
        _posthoc_group_cols=["Metadata_Plate", "Metadata_Well"],
        _drop_cols=["Metadata_Well"]
    )
)

#|%%--%%| <bfMQ1sVcMz|UfHLEfoctO>
r"""°°°
## Well Correlations (same genotype and same plate)
°°°"""
#|%%--%%| <UfHLEfoctO|EDzbNe3PgY>

correlationsdf.append(
    cd.intra_correlations(
        _df=platesdf.copy(),
        _antehoc_group_cols=["Metadata_Plate", "Metadata_genotype"],
        _feat_cols=morph_cols,
        _posthoc_group_cols=["Metadata_Well"],
        _drop_cols=["Metadata_Well"]
    )
)

#|%%--%%| <EDzbNe3PgY|Gp5LlIa6Il>
r"""°°°
# Save Plate Correlations
°°°"""
#|%%--%%| <Gp5LlIa6Il|Kz1w4ViFmv>

correlationsdf = pd.concat(correlationsdf, axis=0)

correlationsdf.to_parquet()

#|%%--%%| <Kz1w4ViFmv|A0fjXCKcdj>

correlationsdf.head()
