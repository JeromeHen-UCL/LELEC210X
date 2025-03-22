"""
Filters an .npy ndarray containing mels. Shows each melspec and asks if it should be kept or not.
"""


import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from rich.logging import RichHandler


CLASS_NAMES = ("background", "chainsaw", "fire", "fireworks", "gunshot")


logger: logging.Logger


def main(args: argparse.Namespace) -> None:

    plt.ion()

    for class_name in CLASS_NAMES:
        logger.info("Class: %s", class_name)

        skip = input("Skip? (y/N)").lower() not in ("N", "")
        if skip:
            continue

        melspecs = np.load(os.path.join(args.input_dir, f"{class_name}_mels.npy"))

        melspecs_to_keep = []
        for i, melspec in enumerate(melspecs):
            plt.cla()
            plt.imshow(melspec.reshape(20, 20), cmap="jet", aspect="auto", extent=(0, 20, 20, 0))
            plt.gca().invert_yaxis()
            plt.title(f"{class_name.capitalize()} {i}")
            plt.show()

            keep = input(f"Keep {i}/{len(melspecs)}? (Y/n): ").lower() in ('y', '')
            if keep:
                logger.debug("Kept")
                melspecs_to_keep.append(melspec/np.max(melspec))

        np.save(os.path.join(args.input_dir,
                f"{class_name}_mels_filtered.npy"), np.array(melspecs_to_keep).squeeze(axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the dataset for the classification task.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="real_melspecs",
        help="Directory containing the mels files to be used")
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
