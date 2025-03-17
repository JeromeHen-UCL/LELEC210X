import os
from pathlib import Path
import pickle
from typing import Optional

import click
import numpy as np
import requests

from auth import PRINT_PREFIX
import common
from common.env import load_dotenv
from common.logging import logger

from .utils import payload_to_melvecs

load_dotenv()

MELVEC_LENGTH = 20
N_MELVECS = 20
HOSTNAME_LOCAL = "http://localhost:5000"
HOSTNAME_CONTEST = "http://lelec210x.sipr.ucl.ac.be/lelec210x"


@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    "model_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
@click.option(
    "--host",
    "host",
    default=None,
    type=click.Choice(["local", "contest"]),
    help="Choice of host for the contest. If none specified there will be no server submission.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model_path: Optional[Path],
    host: Optional[str],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).

    Most likely, you want to pipe this script after running authentification
    on the packets:

        rye run auth | rye run classify

    This way, you will directly receive the authentified packets from STDIN
    (standard input, i.e., the terminal).
    """

    logger.info("Starting classification...")

    if model_path:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    else:
        model = None

    # Why python 3.9? could have used match case :'(
    # get the server address
    # get the server key in order to send guesses
    if host == "local":
        hostname = HOSTNAME_LOCAL
        key = os.getenv("LOCAL_KEY")
    elif host == "contest":
        hostname = HOSTNAME_CONTEST
        key = os.getenv("CONTEST_KEY")
    elif host is None:
        hostname = None
        key = None
    else:
        logger.error("No correct host chosen, got %s", host)
        hostname = None
        key = None

    if key == "":
        logger.error("No key found, please set the key in the .env file")

    for packet in _input:
        if not PRINT_PREFIX in packet:
            continue

        payload = packet[len(PRINT_PREFIX):]

        melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
        melvecs_flattened = melvecs.flatten().reshape(1, -1)
        logger.debug("Parsed payload into Mel vectors: %s", melvecs)
        logger.debug("\tflattened into: %s", melvecs_flattened)

        max_condition = np.max(melvecs_flattened) > 0.010
        logger.info("Max is %s", np.max(melvecs_flattened))
        if max_condition:
            logger.info("Max trigerred")

        norm_condition = np.linalg.norm(melvecs_flattened) > 0.010
        logger.info("Norm is %s", np.linalg.norm(melvecs_flattened))
        if norm_condition:
            logger.info("Norm trigerred")

        if model and (max_condition or norm_condition):
            # predictions = model.predict(melvecs_flattened)
            predictions = ["fireworks"]  # HACK: fixme later
            # predictions_proba = model.predict_proba(melvecs_flattened)
            predictions_proba = [1]  # HACK: fixme later

            logger.info("%s : %s ", str(predictions[0]), predictions_proba[0])

            if hostname:
                logger.info("Submitting prediction %s to %s", predictions[0], hostname)
                try:
                    requests.post(
                        f"{hostname}/lelec210x/leaderboard/submit/{key}/{predictions[0]}", timeout=1)
                except requests.exceptions.RequestException as e:
                    logger.error("Failed to submit prediction: %s", e)
