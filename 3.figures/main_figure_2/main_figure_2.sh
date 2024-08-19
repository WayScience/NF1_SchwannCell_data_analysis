#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the python based analysis env
conda activate nf1_analysis

# convert all notebooks to script files into the scripts folder
jupyter nbconvert --to script --output-dir=scripts/ *.ipynb

# run the notebook for running t-test
python scripts/correlation_t_test.py

# deactivate python env and activate R env
conda deactivate
conda activate nf1_figures

# run notebooks to generate main figure 2
Rscript scripts/main_figure_2.r
