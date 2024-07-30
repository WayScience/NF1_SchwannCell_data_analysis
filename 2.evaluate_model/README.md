# Evaluate NF1 Model
After training the NF1 model described in [1.train_models]("../1.train_models), and saving the results, we evaluate the performance of the NF1 model.
We evaluate the NF1 model on each split (train, validation, and test), the entire dataset, and each plate using the following metrics:
- Precision
- Recall
- Accuracy
- F1 score
- Confusion matrices
- Threshold precision and recall scores

In addition to these changes, we save the feature importances of the model to gather insights about key morphology differences.
