# Evaluate NF1 Model
After training the NF1 model described in [1.train_models]("../1.train_models), and saving the results, we evaluate the performance of the NF1 model.
We evaluate the final NF1 model on each split (train, validation, and test), each plate, and across all plates using the following metrics:

- Precision
- Recall
- Accuracy
- F1 score
- Confusion matrices

> **NOTE:** The precision and recall thresholded data includes results from the stratified cross-validation test dataset, saved while searching for the optimal NF1 model hyper-parameter.  Data without the data split label `val` were not evaluated during hyper-parameter optimization.

In addition to these changes, we save the feature importances of the model to gather insights about key morphology differences.
