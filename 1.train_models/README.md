# Develop the NF1 model
In this module, we extract patterns to discern between cells with WT and Null genotypes using cell morphologies from plates 3, 3 prime, and 5.
After substantial development we have developed and optimized our final NF1 model with the same architecure as a logistic regression.
The training and testing sets are choose to create a 90-10 train-test split between wells, while approximately retaining a uniform genotype class distribution of cells within each data split.
This minimizes the loss of cells when downsampling to the minority genotype class in each data split.
We optimize the choosen model by performing a random search with stratified k-fold cross validation using the class-balanced training set.
After optimization we retrain the NF1 model with the optimal regularization strength using the entire dataset, but excluding the test set.

## Saving the model and results
During optimization of the NF1 model we save the predicted genotype probabilities, true genotype labels, and the predicted genotype labels of the training, validation, and testing datasets for later evaluation.
