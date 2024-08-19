# Creating main figure 2 - Morphology differences at single-cell and well-population levels

To generate the second main figure of the manuscript, there are 2 steps to follow:

1. [correlation_t_test.ipynb](./correlation_t_test.ipynb): For Panel C of the figure, we perform a t-test to evaluate if the means of the Pearson's correlation distributions (either wells are same or different genotype) are significantly different.
2. [main_figure_2.ipynb](./main_figure_2.ipynb): Generate counts, UMAP, and density plot of the correlations and patch together to make one figure.

All steps can be ran with the bash script using the command below:

```bash
source main_figure_2.sh
```
