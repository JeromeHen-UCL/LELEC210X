"""
Generate a model with optimized hyperparameters via Optuna. Loads the mels and labels from 
`--input_dir`.
"""


import argparse
from collections import Counter
from functools import partial
import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import optuna
from rich.logging import RichHandler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MELS_LEN = 20  # number of mels per melvector
MELVECS_LEN = 20  # number of melvectors in a melspec

CLASS_NAMES = ("background", "chainsaw", "fire", "fireworks", "gunshot")

OBJ_SCORE_SYNTH_FRAC = .2
OBJ_SCORE_REAL_FRAC = 1 - OBJ_SCORE_SYNTH_FRAC
OBJ_PCA_PENALTY = 0.1  # penalty for too many PCA components

SHOW_SPE_TRUE = "fireworks"
SHOW_SPE_PRED = "chainsaw"

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


def load_data(args: argparse.Namespace) -> tuple[np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray,
                                                 list[np.ndarray],
                                                 list[np.ndarray],
                                                 list[str]]:
    """
    Load all of the needed data for the model optimization. This includes the database mels and
    labels, the test mels and labels, the real mels and labels.

    Args:
        args (argparse.Namespace): the args to the script

    Returns:
        tuple: X_train, db_mels_test, y_train, y_test, melvecs_classes, X_real, y_real
    """

    db_mels, db_labels = load_db(args)
    X_train, db_mels_test, y_train, y_test = train_test_split(
        db_mels, db_labels, stratify=db_labels)

    melvecs_classes = []
    for class_name in CLASS_NAMES:
        melvecs_classes.append(np.load(os.path.join(
            "real_melspecs", f"{class_name}_mels.npy")))

    X_real = []
    y_real = []
    for melvecs_class, class_name in zip(melvecs_classes, CLASS_NAMES):
        X_real.extend(melvecs_class)
        y_real.extend([class_name] * len(melvecs_class))
    return X_train, db_mels_test, y_train, y_test, melvecs_classes, X_real, y_real


def get_real_accuracy(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy of the model on real data, ignoring background predictions
    when the true label is not 'background'. This is done by filtering out background predictions
    when the true label is not 'background', and then assigning per-sample weights so that
    each class contributes equally to the accuracy score.

    Args:
        y_real (np.ndarray): the real labels
        y_pred (np.ndarray): the predicted labels

    Returns:
        float: the real accuracy
    """

    # Filter out 'background' predictions when true label isn't 'background'
    y_real_filtered = []
    y_pred_filtered = []
    for yt, yp in zip(y_real, y_pred):
        if yt != "background" and yp == "background":
            continue
        y_real_filtered.append(yt)
        y_pred_filtered.append(yp)

    # Count valid samples per class (after filtering)
    class_counts = Counter(y_real_filtered)

    # Assign per-sample weights so each class contributes equally
    weights = np.array([
        1.0 / class_counts[yt] if class_counts[yt] > 0 else 0.0
        for yt in y_real_filtered
    ])

    return accuracy_score(y_real_filtered, y_pred_filtered, sample_weight=weights)


def objective(
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_real: np.ndarray,
        y_real: np.ndarray) -> float:
    """
    Optuna objective function to optimize (maximize). Applies a pipeline with given hyperparameters
    and returns the cross validation score.

    Args:
        trial (Any): Optuna trial object
        X_train (np.ndarray): mels for training
        y_train (np.ndarray): labels for training
        X_train (np.ndarray): mels for testing on real data
        y_real (np.ndarray): labels testing on real data

    Returns:
        float: the cross validation score
    """

    n_components = trial.suggest_int("n_components", 5, MELS_LEN * MELVECS_LEN)  # PCA components
    n_neighbors = trial.suggest_int("n_neighbors", 1, 200)  # k-NN neighbors

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Normalize data
        ("pca", PCA(n_components=n_components)),  # Dimensionality reduction
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))  # Classification
    ])

    score_synth = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
    pipeline = pipeline.fit(X_train, y_train)
    y_real_pred = pipeline.predict(X_real)
    score_real = get_real_accuracy(y_real, y_real_pred)

    pca_penality = OBJ_PCA_PENALTY * n_components / (MELS_LEN * MELVECS_LEN)

    return OBJ_SCORE_SYNTH_FRAC * score_synth + OBJ_SCORE_REAL_FRAC * score_real - pca_penality


def show_cm(cm: np.ndarray, label: str) -> None:
    """
    Show the confusion matrix for a given label.

    Args:
        cm (np.ndarray): confusion matrix
        label (str): label for the confusion matrix
    """

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap="jet",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    # plt.xticks(np.arange(len(CLASS_NAMES))+.5, CLASS_NAMES, rotation=45)
    plt.ylabel("Actual")
    # plt.yticks(np.arange(len(CLASS_NAMES))+.5, CLASS_NAMES, rotation=45)
    plt.title(f"Confusion matrix on {label.capitalize()} data")
    plt.savefig(f"cm_{label}.svg")
    plt.show()


def show_specifics(
        X_real: np.ndarray,
        y_real: np.ndarray,
        y_pred_test: np.ndarray,
        true_label: str,
        pred_label) -> None:
    """
    Show specific melspecs for a given true label and predicted label.

    Args:
        X_real (np.ndarray): real melspecs
        y_real (np.ndarray): real melspecs labels
        y_pred_test (np.ndarray): predictions on real melspecs
        true_label (str): column name to show
        pred_label (str): row name to show
    """

    for xr, yt, yp in zip(X_real, y_real, y_pred_test):
        if yt == true_label and yp == pred_label:
            plt.imshow(xr.reshape(20, 20), cmap="jet", aspect="auto", extent=(0, 20, 20, 0))
            plt.gca().invert_yaxis()
            plt.title(f"Real {yt.capitalize()}, pred {yp.capitalize()}")
            plt.show()


def main(args: argparse.Namespace) -> None:
    """
    The main function.

    Args:
        args (argparse.Namespace): the argument of the script
    """

    # [1] Load database
    X_train, db_mels_test, y_train, y_test, melvecs_classes, X_real, y_real = load_data(args)

    # [2] Perform hyperparameter optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    hyperparam_study = optuna.create_study(
        study_name="Hyperparam Optimizer",
        direction="maximize",
        sampler=sampler
    )  # Maximize accuracy
    hyperparam_study.optimize(
        partial(objective, X_train=X_train, y_train=y_train, X_real=X_real, y_real=y_real),
        n_trials=30, show_progress_bar=True
    )

    # [3] Extract best pipeline
    best_hyperparams = hyperparam_study.best_params
    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=best_hyperparams["n_components"])),
        ("knn", KNeighborsClassifier(n_neighbors=best_hyperparams["n_neighbors"]))
    ])

    logger.info("PCA: %d\tNeighbours: %d",
                best_hyperparams["n_components"], best_hyperparams["n_neighbors"])

    # Save scaler and PCA components
    best_pipeline.fit(X_train, y_train)
    scaler = best_pipeline.steps[0][1]
    best_pca = best_pipeline.steps[1][1]
    best_model = best_pipeline.steps[2][1]

    pickle.dump(best_pipeline, open("pipeline.pickle", "wb"))
    pickle.dump(scaler, open("scaler.pickle", "wb"))
    pickle.dump(best_pca, open("pca.pickle", "wb"))
    pickle.dump(best_model, open("model.pickle", "wb"))

    # [4] Evaluate the model on synthetic data
    y_pred_test = best_pipeline.predict(db_mels_test)
    real_accuracy = accuracy_score(y_test, y_pred_test)
    cm_synth = confusion_matrix(y_test, y_pred_test, normalize="true")
    logger.info("Accuracy on synthetic data is %f", real_accuracy)

    # [5] Evaluate the model on real data
    cm_real = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
    for i, (melvecs_class, true_class) in enumerate(zip(melvecs_classes, CLASS_NAMES)):
        y_real_pred_class = list(best_pipeline.predict(melvecs_class))

        # Filter out background predictions for non-background classes
        if true_class != "background":
            y_real_pred_class = [p for p in y_real_pred_class if p != "background"]

        # Fill confusion matrix (normalized per row)
        total = len(y_real_pred_class) if y_real_pred_class else 1
        for j, class_name in enumerate(CLASS_NAMES):
            count = y_real_pred_class.count(class_name)
            cm_real[i, j] = round(count / total, 2) if total else 0

    y_real_pred = best_pipeline.predict(X_real)
    real_accuracy = get_real_accuracy(y_real, y_real_pred)
    logger.info("Accuracy on real data (ignoring background) is %.2f", real_accuracy)

    # [6] Show synthetic and real confusion matrices
    for cm, label in zip((cm_synth, cm_real), ("synthetic", "real")):
        show_cm(cm, label)

    # [7] Show specific melspecs
    # show_specifics(X_real, y_real, y_pred_test, SHOW_SPE_TRUE, SHOW_SPE_PRED)


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
