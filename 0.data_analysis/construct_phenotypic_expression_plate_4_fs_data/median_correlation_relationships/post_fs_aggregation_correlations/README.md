# Aggregate Correlation Analysis

In this analysis, we compare the effect of different treatments on aggregated cell morphology profiles using correlation.
We summarize these treatments as follows:

- siRNA constructs 1 and 2 target the mRNA of the NF1 gene.
- Scramble is a random siRNA sequence.
- No Treatment is the state without any perturbation

### Correlation Comparisons

We aggregated single cells post normalization and feature selection to calculate pairwise correlations across wells between:

- The same siRNA constructs within each construct concentration
- Different siRNA constructs within each construct concentration

### Visualizing Comparisons

Correlations were then plotted as boxplots for each concentration in the following groups:

- (Construct 1) and (Construct 2)
- (Construct 1) and (Constrcut 1)
- (Construct 2) and (Construct 2)
- (Construct 1) and (Scramble)
- (Construct 2) and (Scramble)
- (Scramble) and (Scramble)
- (Construct 1) and (No Treatment)
- (Construct 2) and (No Treatment)
- (Scramble) and (No Treatment)
- (No Treatment) and (No Treatment)
