#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the R based analysis env
conda activate nf1_figures

# convert all notebooks to script files into the scripts folder
jupyter nbconvert --to script --output-dir=scripts/ *.ipynb

# run notebooks to generate supplemental figure 2
Rscript scripts/SuppFigure2_qaulitycontrol.r
