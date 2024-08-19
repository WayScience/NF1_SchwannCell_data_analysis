# Generate manuscript figures
After evaluation results are extracted, we generate figures describing the results of our experiment.
There are a total of six figures (four main and two supplemental).
All figure PNGs are found in the [figures](./figures/) folder.

1. [*Main figure 1*](./main_figure_1/): This figure describes our workflow and displays an image montage of the wildtype and null *NF1* genotype single cells, which are hard to distinguish just by eye.
2. [Main figure 2](./main_figure_2/): This figure shows how subtle the morphological differences are between *NF1* genotypes at both the well-population and single-cell levels, which supports are reasoning to pursue a machine learning methodology.
3. [Main figure 3](./main_figure_3/): This figure shows the results of the model evaluations (precision-recall, accuracy, and confusion matrices) as extracted in the second module of this repository.
4. [Main figure 4](./main_figure_4/): This figure looks at the feature importances of the model when predicting *NF1* genotype. There are two image montages that show six example single-cells (three with the highest values of the feature and three with the lowest), one for each of the top features for predicting each genotype.
5. [Supplemental figure 1](./supp_figure_1/): This figure is an extension of main figure 2, which facets the plot by plate to show that the subtle differences between *NF1* genotype are consistent. 
6. [Supplemental figure 2](./supp_figure_2/): This figure shows the distributions of FOVs across blur (PowerLogLogSlope) and saturation (PercentMaximal) metrics and where the thresholds were assigned to detect poor-quality images.
