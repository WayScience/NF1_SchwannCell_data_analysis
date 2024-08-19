# Creating main figure 1 - Workflow and genotype image montage

To generate the first main figure of the manuscript, there are 5 steps to follow:

1. [1.find_sc_crops.ipynb](./1.find_sc_crops.ipynb): Find randomly selected single-cells (one per genotype) and create crops around the single-cell for each channel. 
2. [update_channel_colors.ijm](./update_channel_colors.ijm): Using an ImageJ macro (first developed by [Mike Lippincott for the Interstellar_Analysis repo](https://github.com/MikeLippincott/Interstellar_Analysis/blob/main/figures/S13/imageJ_macros/channel_change.ijm)), we can change the channel single-cell crops from greyscale to assigned RGB colors (DAPI = blue, GFP = green, CY5 = magenta, RFP = red). This macro can be dragged and dropped into ImageJ.
3. While the crops are still in ImageJ, we manually stack the channels together into one composite image (make sure to keep source images). Then, we add 25 uM scales to each crop using 3.1065 uM/pixel in the `Analyze > Set Scale... module` (as identified from the metadata of the raw images). All crops are saved as PNGs back into the same folder.
4. [2.create_image_montage.ipynb](./2.create_image_montage.ipynb): Using the updated and colored crops, we can now merge them together to make an image montage figure that labels each crop per channel and per genotype.
5. [3.main_figure_1.ipynb](./3.main_figure_1.ipynb): Patch together the workflow image and image montage into one main figure.

All steps can be ran with the bash script using the command below:

```bash
source main_figure_1.sh
```
