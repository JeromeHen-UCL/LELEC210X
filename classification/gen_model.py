"""
Generate a model with optimized hyperparameters via Optuna. Loads the mels and labels from 
`--input_dir`.
"""


import argparse
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
MELVECS_LEN = 20  # number of melvectors in a melmatrix

CLASS_NAMES = ("background", "chainsaw", "fire", "fireworks", "gunshot")


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
    n_neighbors = trial.suggest_int("n_neighbors", 3, 200)  # k-NN neighbors

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

    # TODO: save scaler and PCA components
    best_pipeline.fit(db_mels_train, db_labels_train)
    best_model = KNeighborsClassifier(n_neighbors=best_hyperparams["n_neighbors"])
    best_model.fit(db_mels_train, db_labels_train)

    # TODO: change when PCA is on MCU
    pickle.dump(best_pipeline, open("model.pickle", "wb"))

    # [4] Evaluate the model on synthetic data
    y_pred_test = best_pipeline.predict(db_mels_test)
    real_accuracy = accuracy_score(db_labels_test, y_pred_test)
    cm_test = confusion_matrix(db_labels_test, y_pred_test, normalize="true")
    logger.info("Accuracy on synthetic data is %f", real_accuracy)

    # Visualize the synthetic melvecs
    # for melvec, label in zip(db_mels_train, db_labels_train):
    #     plt.imshow(melvec.reshape(20, 20), cmap="jet", aspect="auto", extent=(0, 20, 20, 0))
    #     plt.gca().invert_yaxis()
    #     plt.title(f"Train {label.capitalize()}")
    #     plt.show()

    y_pred_real = []
    y_true_real = []
    melvecs_classes = []
    for class_name in CLASS_NAMES:
        melvecs_classes.append(np.load(os.path.join(
            "real_melspecs", f"{class_name}_mels.npy")))

    min_len = np.min([len(melvecs) for melvecs in melvecs_classes])
    melvecs_classes = [melvecs[:min_len] for melvecs in melvecs_classes]

    # [5] Evaluate the model on real data
    cm_real = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
    y_pred_real = []
    y_true_real = []
    for i, (melvecs, class_name) in enumerate(zip(melvecs_classes, CLASS_NAMES)):
        y_pred_real_class = list(best_pipeline.predict(melvecs))

        if i != 0:
            y_pred_real_class = list(filter(lambda x: x != "background", y_pred_real_class))
            y_pred_real += list(y_pred_real_class)
            y_true_real += [class_name] * len(y_pred_real_class)

        for j, class_name in enumerate(CLASS_NAMES):
            cm_real[i, j] = y_pred_real_class.count(class_name)

            # Do not consider background predictions when the actual class is not background
            if i != 0:
                cm_real[i, 0] = 0
                cm_real[i, j] /= len(y_pred_real_class) - y_pred_real_class.count("background")
            else:
                cm_real[i, j] /= len(y_pred_real_class)

            cm_real[i, j] = round(cm_real[i, j], 2)

        # if class_name in ("background"):
        #     continue
        # for i, melvec in enumerate(melvecs):
        #     plt.imshow(melvec.reshape(20, 20), cmap="jet", aspect="auto", extent=(0, 20, 20, 0))
        #     plt.gca().invert_yaxis()
        #     plt.title(f"{class_name.capitalize()} {i}")
        #     plt.show()

    real_accuracy = accuracy_score(y_true_real, y_pred_real)
    logger.info("Accuracy on real data (ignoring background) is %f", real_accuracy)

    # [6] Show synthetic and real confusion matrices
    for cm, label in zip((cm_test, cm_real), ("synthetic", "real")):
        print(cm)

        sns.heatmap(cm, annot=True, fmt='.2f', cmap="jet",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("Predicted")
        # plt.xticks(np.arange(len(CLASS_NAMES))+.5, CLASS_NAMES, rotation=45)
        plt.ylabel("Actual")
        # plt.yticks(np.arange(len(CLASS_NAMES))+.5, CLASS_NAMES, rotation=45)
        plt.title(f"Confusion matrix on {label.capitalize()} data")
        plt.show()


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
