# Evaluate NF1 Model
After training the NF1 model described in [1.train_models]("../1.train_models), and saving the results, we evaluate the performance of the NF1 model.
We evaluate the final NF1 model on each split (train, validation, and test), each plate, and across all plates using the following metrics:

- Precision
- Recall
- Accuracy
- F1 score
- Confusion matrices

> **NOTE:** The precision and recall data cover the results from all the different parameter settings tested during the hyperparameter search. All other files contain the results from the final model (best hyperparameters).

In addition to these changes, we save the feature importances of the model to gather insights about key morphology differences.
