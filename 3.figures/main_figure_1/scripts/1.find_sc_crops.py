#!/usr/bin/env python
# coding: utf-8

# # Generate random single-cell crops of cells per genotype from Plate 5 for main figure

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd
import numpy as np
import cv2


# ## Set paths and variables

# In[2]:


# Images are accessible in the nf1_schwanncell_data repo
path_to_images_dir = pathlib.Path(
    "../../../nf1_cellpainting_data/1.cellprofiler_ic/Corrected_Images/Corrected_Plate_5"
)  # Focus on plate 5

# Path to wear single-cell crops are saved
path_to_sc_dir = pathlib.Path("./sc_crops")
path_to_sc_dir.mkdir(exist_ok=True)

# URL path to annotated parquet file from Plate 5 (versioned)
url = "https://github.com/WayScience/nf1_cellpainting_data/raw/main/3.processing_features/data/single_cell_profiles/Plate_5_sc_annotated.parquet"


# ## Load in annotated data frame and only include metadata 
# 
# NOTE: We normally use random seed = 0 but we have changed it here to find best random cells for viewing that are not cells going through mitosis or cell death.

# In[3]:


# This random seed value does not follow the conventions of the lab, but yields the best visualizations of single-cells
random_seed_value = 58
# Set a seed for reproducibility
np.random.seed(random_seed_value)

# Load in plate 5 data frame
plate5_df = pd.read_parquet(
    url,
    columns=[
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_genotype",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_Cells_Location_Center_X",
        "Metadata_Cells_Location_Center_Y",
    ],
)

# Exclude rows where "Metadata_genotype" is "HET" due to not using during the training of the model
plate5_df = plate5_df[plate5_df["Metadata_genotype"] != "HET"]

# Select one random row per "Metadata_genotype"
plate5_df = (
    plate5_df.groupby("Metadata_genotype")
    .apply(lambda x: x.sample(1, random_state=random_seed_value))
    .reset_index(drop=True)
)

print(plate5_df.shape)
plate5_df.head(5)


# ## Set up dictionary to hold info to find random single-cells per genotype

# In[4]:


# B1_01_1_1_DAPI_001_illumcorrect.tiff

# Create dictionary to run through each single-cell to find crop
random_sc_dict = {}
for _, row in plate5_df.head().iterrows():
    genotype_key = f"{row['Metadata_genotype']}_genotype"
    random_sc_dict[genotype_key] = {
        "well": row["Metadata_Well"],
        "site": row["Metadata_Site"],
        "location_center_x": row["Metadata_Nuclei_Location_Center_X"],
        "location_center_y": row["Metadata_Nuclei_Location_Center_Y"],
    }

# Check the created dictionary
print(random_sc_dict)


# ## Generate single-cell crops

# In[5]:


# Define a mapping for the suffixes
channel_mapping = {1: "DAPI", 2: "GFP", 3: "CY5", 4: "RFP"}

for genotype, info in random_sc_dict.items():
    # Initialize a list to store file paths
    file_paths = []

    # Create file paths with well, site, and channel
    for i in range(1, 5):  # Update the range to start from 1
        channel = channel_mapping[i]
        filename = f"{path_to_images_dir}/{info['well']}_01_{i}_{info['site']}_{channel}_001_illumcorrect.tiff"
        file_paths.append(filename)

        # Read the image
        channel_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # Use the location_center_x and location_center_y to create a crop
        center_x = info.get("location_center_x")
        center_y = info.get("location_center_y")

        # Crop dimensions
        crop_size = 250
        half_crop = crop_size // 2

        # Ensure the center coordinates are valid
        if center_x is not None and center_y is not None:
            # Calculate crop boundaries
            top_left_x = max(int(center_x - half_crop), 0)
            top_left_y = max(int(center_y - half_crop), 0)
            bottom_right_x = min(int(center_x + half_crop), channel_image.shape[1])
            bottom_right_y = min(int(center_y + half_crop), channel_image.shape[0])

            # Perform cropping
            cropped_channel = channel_image[
                top_left_y:bottom_right_y, top_left_x:bottom_right_x
            ]

            # Ensure the cropped image is of size 250x250
            cropped_channel = cv2.resize(cropped_channel, (crop_size, crop_size))

            # Save the cropped image with single_cell and channel information
            output_filename = f"{path_to_sc_dir}/{genotype}_{channel}_cropped.png"
            cv2.imwrite(output_filename, cropped_channel)

