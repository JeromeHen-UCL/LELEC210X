import os
from pathlib import Path
import pickle
from typing import Optional

import click
import requests

from auth import PRINT_PREFIX
import common
from common.env import load_dotenv
from common.logging import logger

from .utils import payload_to_melvecs

load_dotenv()

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

    for packet in _input:
        if PRINT_PREFIX in packet:
            payload = packet[len(PRINT_PREFIX):]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info("Parsed payload into Mel vectors: %s", melvecs)

            if model:
                predictions = model.predict(melvecs)
                prediction_proba = model.predict_proba(melvecs)
                logger.debug("%s : %s ", str(predictions[0]), prediction_proba[0])

                if hostname:
                    logger.debug("Submitting prediction %s to %s", predictions[0], hostname)
                    requests.post(f"{hostname}/lelec210x/leaderboard/submit/\
                                {key}/{predictions}", timeout=1)
