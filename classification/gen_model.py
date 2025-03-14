"""
Generate a model with optimized hyperparameters via Optuna. Loads the mels and labels from 
`--input_dir`.
"""


import argparse
from functools import partial
import logging
import os

import numpy as np
import optuna
from rich.logging import RichHandler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MELS_LEN = 20  # number of mels per melvector
MELVECS_LEN = 20  # number of melvectors in a melmatrix


logger: logging.Logger


def load_db(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """
    Load database (mels and labels) from a given directory.

    Args:
        args (argparse.Namespace): the args to the script

    Returns:
        tuple[np.ndarray, np.ndarray]: db_mels, db_labels
    """

    db_mels = np.load(os.path.join(args.input_dir, "db_mels.npy"))
    db_labels = np.load(os.path.join(args.input_dir, "db_labels.npy"), allow_pickle=True)

    return db_mels, db_labels


def objective(trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Optuna objective function to optimize (maximize). Applies a pipeline with given hyperparameters
    and returns the cross validation score.

    Args:
        trial (Any): Optuna trial object
        X_train (np.ndarray): mels for training
        y_train (np.ndarray): labels for training

    Returns:
        float: the cross validation score
    """

    n_components = trial.suggest_int("n_components", 5, MELS_LEN * MELVECS_LEN)  # PCA components
    n_neighbors = trial.suggest_int("n_neighbors", 1, 20)  # k-NN neighbors

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Normalize data
        ("pca", PCA(n_components=n_components)),  # Dimensionality reduction
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))  # Classification
    ])

    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()

    return score


def main(args: argparse.Namespace) -> None:
    """
    The main function.

    Args:
        args (argparse.Namespace): the argument of the script
    """

    # [1] Load database
    db_mels, db_labels = load_db(args)
    db_mels_train, db_mels_test, db_labels_train, db_labels_test = train_test_split(
        db_mels, db_labels, stratify=db_labels)

    # [2] Perform hyperparameter optimization
    hyperparam_study = optuna.create_study(
        study_name="Hyperparam Optimizer", direction="maximize")  # Maximize accuracy
    hyperparam_study.optimize(
        partial(objective, X_train=db_mels_train, y_train=db_labels_train),
        n_trials=50
    )

    # [3] Test best parameters on real set
    best_hyperparams = hyperparam_study.best_params
    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=best_hyperparams["n_components"])),
        ("knn", KNeighborsClassifier(n_neighbors=best_hyperparams["n_neighbors"]))
    ])
    # TODO: save scaler and PCA components
    best_pipeline.fit(db_mels_train, db_labels_train)

    # [4] Evaluate the model
    # FIXME: use real melvecs to test the model in real conditions
    y_pred = best_pipeline.predict(db_mels_test)
    test_accuracy = accuracy_score(db_labels_test, y_pred)

    logger.info("Accuracy is %f", test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new model on optimized hyperparameters.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory containing the .npy dataset arrays")
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="If the verbose mode should be activated")

    main_args = parser.parse_args()

    handler = RichHandler(keywords=RichHandler.KEYWORDS)
    logging.basicConfig(level=main_args.verbose, format="%(message)s",
                        datefmt="[%X]", handlers=[handler])
    logger = logging.getLogger("LELEC210X")

    main(main_args)
