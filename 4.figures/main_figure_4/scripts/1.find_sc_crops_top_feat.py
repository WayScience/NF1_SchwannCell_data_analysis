#!/usr/bin/env python
# coding: utf-8

# # Generate min/max repesentative single-cell images per top two Null features from the model
# 
# 1. Average edge intensity of GFP in cytoplasm
# 2. Radial distribution of RFP in cytoplasm

# ## Import libraries

# In[1]:


import pathlib
from pprint import pprint

import cv2
import pandas as pd
from typing import List, Dict


# ## Define functions

# In[2]:


# Function for formatting min/max row data frames into dictionaries
def create_sc_dict(dfs: List[pd.DataFrame], names: List[str]) -> dict:
    """Format lists of data frames and names into a dictionary with all relevant metadata to find single-cell images.

    Args:
        dfs (List[pd.DataFrame]): List of data frames each containing a single cell and relevant metadata.
        names (List[str]): List of names corresponding to the data frames.

    Returns:
        dict: Dictionary containing info relevant for finding single-cell crops.
    """
    sc_dict = {}
    for df, name in zip(dfs, names):
        for i, (_, row) in enumerate(df.iterrows()):
            key = f"{name}_{i + 1}"
            sc_dict[key] = {
                "plate": row["Metadata_Plate"],
                "well": row["Metadata_Well"],
                "site": row["Metadata_Site"],
                "location_center_x": row["Metadata_Nuclei_Location_Center_X"],
                "location_center_y": row["Metadata_Nuclei_Location_Center_Y"],
            }
    return sc_dict


# In[3]:


# Function for generating and saving single-cell crops per channel as PNGs
def generate_sc_crops(
    sc_dict: Dict,
    channel_mapping: Dict[int, str], 
    images_dir: pathlib.Path,
    output_img_dir: pathlib.Path,
    crop_size: int,
) -> None:
    """Using a dictionary with single-cell metadata info per image set, single-cell crops per channel are generated
    and saved as PNGs in an image set folder.

    Args:
        sc_dict (Dict): Dictionary containing info relevant for finding single-cell crops.
        channel_mapping (Dict[int, str]): Dictionary mapping integer to channel name for generating paths.
        images_dir (pathlib.Path): Directory where illumination corrected images are found.
        output_img_dir (pathlib.Path): Main directory to save each image set single-cell crops
        crop_size (int): Size of the box in pixels (example: setting crop_size as 250 will make a 250x250 pixel crop around the single-cell center coordinates)
    """
    for key, info in sc_dict.items():
        # Initialize a list to store file paths for every image set
        file_paths = []

        # Create file paths with well, site, and channel
        for i in range(1, 5):  # Update the range to start from 1
            channel = channel_mapping[i]
            filename = f"{images_dir}/{info['well']}_01_{i}_{info['site']}_{channel}_001_illumcorrect.tiff"
            file_paths.append(filename)

            # Read the image
            channel_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # Use the location_center_x and location_center_y to create a crop
            center_x = info.get("location_center_x")
            center_y = info.get("location_center_y")

            # Crop dimensions (including crop_size)
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

                # Make directory for the key to keep all channels for an image in one folder
                key_dir = pathlib.Path(f"{output_img_dir}/{key}")
                key_dir.mkdir(exist_ok=True, parents=True)

                # Save the cropped image with single_cell and channel information
                output_filename = str(pathlib.Path(f"{key_dir}/{key}_d{i}_cropped.png"))
                cv2.imwrite(output_filename, cropped_channel)


# ## Set paths and variables

# In[4]:


# Images directory for plate 5 (using for finding single-cells)
images_dir = pathlib.Path(
    "../../../nf1_cellpainting_data/1.cellprofiler_ic/Corrected_Images/Corrected_Plate_5"
).resolve(strict=True)

# Output dir for cropped images
output_img_dir = pathlib.Path("./sc_crops")
output_img_dir.mkdir(exist_ok=True)

# Define the size of the cropping box (250x250 pixels)
crop_size = 250

# Define a mapping for the suffixes
channel_mapping = {1: "DAPI", 2: "GFP", 3: "CY5", 4: "RFP"}

# Create open list for one row data frames for each top feature per channel per cell type
list_of_dfs = []

# Create open list of names to assign each data frame in a list relating to the feature, channel, and cell type
list_of_names = []


# ## Load in Plate 5 data to generate repesentative images from

# In[5]:


# Load in normalized + feature selected data as data frame
plate5_df = pd.read_parquet(
    pathlib.Path(
        "../../../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_5_sc_feature_selected.parquet"
    )
)

# Load in annotated dataframe to extract neighbors
annot_df = pd.read_parquet(
    pathlib.Path(
        "../../../nf1_cellpainting_data/3.processing_features/data/single_cell_profiles/Plate_5_sc_annotated.parquet"
    ),
    columns=[
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_Nuclei_Number_Object_Number",
        "Cells_Neighbors_NumberOfNeighbors_Adjacent",
    ],
)

plate5_df = plate5_df.merge(
    annot_df,
    on=["Metadata_Well", "Metadata_Site", "Metadata_Nuclei_Number_Object_Number"],
    how="inner",
)

plate5_df.rename(
    columns={
        "Cells_Neighbors_NumberOfNeighbors_Adjacent": "Metadata_Number_of_Cells_Neighbors_Adjacent"
    },
    inplace=True,
)

print(plate5_df.shape)
plate5_df.head()


# ## Load in feature importance data and determine the top two differential features for Null

# In[6]:


feat_import_df = pd.read_parquet(
    pathlib.Path(
        "../../2.evaluate_models/classify_genotypes/model_evaluation_data/feature_importances.parquet"
    )
)

# Find the top positive feature (get the second highest feature that is not correlation)
intensity_feature = feat_import_df.sort_values(by='feature_importances', ascending=True).iloc[1]['feature_names']

# Find the top negative feature
radial_feature = feat_import_df.loc[feat_import_df['feature_importances'].idxmin(), 'feature_names']

# Print the features
print(intensity_feature)
print(radial_feature)


# ## Filter plate 5 single-cells to only include isolated cells that are not near the edge of the FOV

# In[7]:


# Filter the DataFrame directly
filtered_plate5_df = plate5_df[
    (plate5_df['Metadata_Number_of_Cells_Neighbors_Adjacent'].isin([0])) &
    (plate5_df['Metadata_Nuclei_Location_Center_X'] > crop_size // 2) &
    (plate5_df['Metadata_Nuclei_Location_Center_X'] < (plate5_df['Metadata_Nuclei_Location_Center_X'].max() - crop_size // 2)) &
    (plate5_df['Metadata_Nuclei_Location_Center_Y'] > crop_size // 2) &
    (plate5_df['Metadata_Nuclei_Location_Center_Y'] < (plate5_df['Metadata_Nuclei_Location_Center_Y'].max() - crop_size // 2))
]

print(filtered_plate5_df.shape)
filtered_plate5_df.head()


# ### Max single-cells for Intensity feature (represent Null)

# In[8]:


# Get data frame with the top 3 single-cells from the top WT coefficient
max_int_feature = filtered_plate5_df[filtered_plate5_df["Metadata_genotype"] == "Null"].nlargest(
    3, intensity_feature
)[
    [
        intensity_feature,
        "Metadata_genotype",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(max_int_feature)
list_of_names.append("max_int_feature")

print(max_int_feature.shape)
max_int_feature


# ### Min single-cells for Intensity feature (represent WT)

# In[9]:


# Get data frame with the top 3 single-cells from the top WT coefficient
min_int_feature = filtered_plate5_df[filtered_plate5_df["Metadata_genotype"] == "WT"].nsmallest(
    3, intensity_feature
)[
    [
        intensity_feature,
        "Metadata_genotype",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(min_int_feature)
list_of_names.append("min_int_feature")

print(min_int_feature.shape)
min_int_feature


# ### Max single-cells for Radial Distribution feature (represent Null)

# In[10]:


# Get data frame with the top 3 single-cells from the top Null coefficient
max_radial_feature = filtered_plate5_df[filtered_plate5_df["Metadata_genotype"] == "Null"].nlargest(
    3, radial_feature
)[
    [
        radial_feature,
        "Metadata_genotype",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(max_radial_feature)
list_of_names.append("max_radial_feature")

print(max_radial_feature.shape)
max_radial_feature


# ### Min single-cells for Radial Distribution feature (represent WT)

# In[11]:


# Get data frame with the top 3 single-cells from the top Null coefficient
min_radial_feature = filtered_plate5_df[filtered_plate5_df["Metadata_genotype"] == "WT"].nsmallest(
    3, radial_feature
)[
    [
        radial_feature,
        "Metadata_genotype",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Number_of_Cells_Neighbors_Adjacent",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(min_radial_feature)
list_of_names.append("min_radial_feature")

print(min_radial_feature.shape)
min_radial_feature


# ## Merge feature info into dictionary for processing

# In[12]:


sc_dict = create_sc_dict(dfs=list_of_dfs, names=list_of_names)

# Check the created dictionary for the first two items
pprint(list(sc_dict.items())[:2], indent=4)


# ## Generate single-cell crops 

# In[13]:


generate_sc_crops(
    sc_dict=sc_dict,
    channel_mapping=channel_mapping,
    images_dir=images_dir,
    output_img_dir=output_img_dir,
    crop_size=crop_size,
)

