import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = ("birds", "chainsaw", "fire", "handsaw", "helicopter")


def show_cm(cm: np.ndarray) -> None:
    """
    Show the confusion matrix for a given label.

    Args:
        cm (np.ndarray): confusion matrix
    """

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap="jet",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.xticks(np.arange(len(CLASS_NAMES))+.5, CLASS_NAMES, rotation=45)
    plt.ylabel("Actual")
    plt.yticks(np.arange(len(CLASS_NAMES))+.5, CLASS_NAMES, rotation=45)
    plt.title("Confusion matrix of the old classifier")
    plt.tight_layout()
    plt.savefig("cm_old.svg")
    plt.show()


cm_matrix = np.array(((0, 8, 0, 3, 2),
                     (0, 0, 0, 0, 0),
                     (0, 0, 0, 0, 1),
                     (20, 12, 19, 17, 17),
                     (0, 0, 0, 0, 0))).T / 100

show_cm(cm_matrix)
