#!/usr/bin/env python
# coding: utf-8

# # Here the features are seperated according to compartments for later analysis

# In[1]:


# %%--%%| <qGnYViiwRD|SyZ3qa8iz3>
r"""°°°
## Imports
°°°"""


# In[ ]:


import itertools
import pandas as pd
from pathlib import Path


# ## Find the git root Directory

# In[ ]:


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


# ## Specify Paths

# In[ ]:


output_path = Path("data")
feature_file = "feature_importances.tsv"

output_path.mkdir(
    parents=True, exist_ok=True
)  # Create the parent directories if they don't exist

comparments_output_path = output_path / "feature_compartments.tsv"


# ## Import the model data as a dataframe

# In[ ]:


feature_properties = pd.read_csv(output_path / feature_file, sep="\t")


# ## Seperate cell data by channel

# In[ ]:


# A map for comparments to channel
compartment2channel = {"actin": "RFP", "er": "GFP", "nucleus": "DAPI"}

# Create a Dictionary to hold the comparment data as dataframes
compartment_data = {
    compartment: feature_properties[feature_properties["feature"].str.contains(channel)]
    for compartment, channel in compartment2channel.items()
}


# ## Find the features that use more than one compartment

# In[ ]:


# Create a list of all possible comparment pairs
pairs = list(itertools.combinations(compartment_data.keys(), 2))

# Use a list of possible compartments for find compartment agnostic features
pos_compartments = list(compartment2channel.values())

# Get the features that do not belong to any compartment specifically
other_compartment = feature_properties[
    ~feature_properties["feature"].str.contains("|".join(pos_compartments))
]

# Find intersected rows between each pair of compartments
for pair in pairs:
    # Create placeholder dataframes for each comparment in the pair
    df1 = compartment_data[pair[0]]
    df2 = compartment_data[pair[1]]

    # Get duplicate features between the two compartment dataframes
    intersection = pd.concat([df1, df2], axis=0)
    intersection = intersection[intersection.duplicated(subset="feature", keep=False)]

    # Remove any duplicate features already added to the other compartment dataframe
    other_compartment = pd.concat([other_compartment, intersection], axis=0)
    other_compartment = other_compartment.drop_duplicates(subset="feature")


# ## Organize the data according to compartment in one dataframe

# In[ ]:


# Remove features from each compartment dataframe if they are duplicates in other dataframes, or if the features only exist in the other_compartment dataframe
compartment_data = {
    compartment: compartment_data[compartment][
        ~compartment_data[compartment]["feature"].isin(other_compartment["feature"])
    ]
    for compartment, channel in compartment2channel.items()
}

compartment_data["other"] = other_compartment

# Create a compartment column for each comparment dataframe
for compartment, df in compartment_data.items():
    compartment_data[compartment]["compartment"] = len(df) * [compartment]

# Concatenate the rows of DataFrames to create the plot
concatenated_df = pd.concat(compartment_data.values(), ignore_index=True)


# ## Save the dataframes

# In[ ]:


concatenated_df.to_csv(comparments_output_path, sep="\t", index=False)
