import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

rnd_val = 0


def get_model_data(traindf, lr):
    """
    Coordinates all of the actions to get the model data

    Parameters
    ----------
    traindf: Pandas Dataframe
        The training and validation dataset

    lr: Sklearn Logistic Regression Model
        An untrained Logistic Regression model

    Returns
    -------
    lr: Logistic Regression
        The best trained logistic regression model

    testdf: Pandas Dataframe
        The test set with labels

    le: The label encoder used to create the labels
    """

    traindf, testdf, le = split_plate(traindf)
    models = train_models(traindf)
    lr = get_best_model(models)

    return lr, testdf, le


def train_models(traindf, lr):
    """
    Trains the model and returns a dictionary of model outputs. Please see https://scikit-learn.org/stable/modules/cross_validation.html for more details

    Parameters
    ----------
    traindf: Pandas Dataframe
        The training and validation dataset

    Returns
    -------
    models: Dictionary
        Please see this link on model outputs returned (https://scikit-learn.org/stable/modules/cross_validation.html)

    lr: Sklearn Logistic Regression Model
        An untrained Logistic Regression model
    """

    # Default number of splits
    num_splits = 5

    lr = LogisticRegression(
        max_iter=1000, solver="sag", random_state=rnd_val, n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=rnd_val)

    xvalidate_params = {
        "estimator": lr,
        "X": traindf.drop("label", axis="columns"),
        "y": traindf["label"],
        "n_jobs": -1,
        "return_estimator": True,
        "scoring": "average_precision",
        "cv": skf,
    }

    models = cross_validate(**xvalidate_params)

    return models


def get_best_model(models):
    """
    Gets the best model

    Parameters
    ----------
    models: Dictionary
        Please see this link on model outputs returned (https://scikit-learn.org/stable/modules/cross_validation.html)

    Returns
    -------
    lr: Sklearn Logistic Regression Model
        The best trained logistic regression model
    """
    # Get the best performing model
    test_score_argmax = models["test_score"].argmax()
    lr = models["estimator"][test_score_argmax]
    return lr


def split_plate(platedf):
    """
    Splits the plate data by stratifying using the stratify_column for each of the target_column values for both the train and the test sets.

    Parameters
    ----------
    platedf: Pandas Dataframe
        The dataframe of plate data

    Returns
    -------
    traindf: Pandas Dataframe
        The split training data for the plate

    testdf: Pandas Dataframe
        The split testing data for the plate

    le: label encoder
        The label encoder of genotypes
    """

    target_column = "Metadata_genotype"
    stratify_column = "Metadata_Well"

    # The fraction of the dataset accounting for the train and validation sets
    train_val_frac = 0.85

    # Find the gene with the least number of cells
    smallest_gene = platedf[target_column].value_counts().min()

    platedata = pd.DataFrame()

    # Stratify using the stratify_column for each unique value in target_column
    for gene in platedf[target_column].unique():
        df = platedf.loc[platedf[target_column] == gene]
        df_frac = smallest_gene / len(df)
        stratwell = df.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(frac=df_frac, random_state=rnd_val)
        )
        platedata = pd.concat([platedata, stratwell], axis="rows")

    # Create train/test split using the data by both the target_column and the stratify_column
    traindf, testdf = train_test_split(
        platedata,
        train_size=train_val_frac,
        stratify=platedata[[target_column, stratify_column]],
        shuffle=True,
        random_state=rnd_val,
    )

    # Encode Labels
    le = LabelEncoder()
    testdf["label"] = le.fit_transform(testdf[target_column].values)
    traindf["label"] = le.transform(traindf[target_column].values)

    # Remove unecessary columns
    feat_col = [col for col in traindf.columns if "Metadata" not in col]

    traindf = traindf[feat_col]
    testdf = testdf[feat_col]

    return traindf, testdf, le
