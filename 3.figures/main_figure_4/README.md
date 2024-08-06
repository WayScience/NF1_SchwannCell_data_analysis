# Creating main figure 4 - Model coefficient plot and image montage

To generate the fourth main figure of the manuscript, there are 4 steps to follow:

1. [1.find_sc_crops_top_feat.ipynb](./1.find_sc_crops_top_feat.ipynb): Find the 6 top representative single cells (3 max and 3 min) for each of the two most weighted features, specifically for the Null genotype  (as the top WT feature was a correlation feature which we decided was harder to visualize).
   
2. We manually stack the channels together into one composite image where blue is nuclei, red is actin, green is ER, and magenta is mitochondria. Then, we add 25 uM scales to each crop using 3.1065 uM/pixel in the `Analyze > Set Scale... module` (as identified from the metadata of the raw images). The composite images are saved as PNGs back into the same folder.

3. [2.generate_image_montage.ipynb](./2.generate_image_montage.ipynb): Using the composite single cell crops, we can now merge them together to make an image montage figure that labels each crop per feature and as either the min/max of the feature.

4. [3.main_figure_4.ipynb](./3.main_figure_4.ipynb): Patch together the coefficient plots and image montage into one main figure.
