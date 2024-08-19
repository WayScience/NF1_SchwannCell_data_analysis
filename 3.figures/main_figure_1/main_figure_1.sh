#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the python based analysis env
conda activate nf1_analysis

# convert all notebooks to script files into the scripts folder
jupyter nbconvert --to script --output-dir=scripts/ *.ipynb

# run the notebook for finding single-cell crops
python scripts/1.find_sc_crops.py

# deactivate python env and activate R env
conda deactivate
conda activate nf1_figures

# run notebooks to generate image montage and main figure 1
Rscript scripts/2.create_image_montage.r
Rscript scripts/3.main_figure_1.r
