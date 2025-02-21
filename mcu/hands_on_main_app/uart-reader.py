"""
uart-reader.py
ELEC PROJECT - 210x
"""

import sys
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import requests
import pickle  # for loading the model
import argparse
import sys


import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from classification.utils.plots import plot_specgram

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

HOSTNAME_LOCAL = "http://localhost:5000"
HOSTNAME_CONTEST = "http://lelec210x.sipr.ucl.ac.be/lelec210x"

dt = np.dtype(np.uint16).newbyteorder("<")


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])
    else:
        # print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                "ascii"
            )
            # print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array / np.linalg.norm(buffer_array)


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-p", "--port", help="Port for serial communication")
    argParser.add_argument(
        "-m", "--model", help="Classification pickle filename")
    argParser.add_argument(
        "--host", help="Contest host (\"local\" or \"contest\")")
    argParser.add_argument(
        "--key", help="Key to use for the contest board (local or remote)")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print(
            "Launch this script with [-p PORT_REF] to access the communication port")
        sys.exit(0)

    if args.model:
        # load the trained model
        with open(args.model, 'rb') as model_file:
            classifier: ClassifierMixin | GaussianProcessClassifier = pickle.load(
                model_file)

    if args.host == "local":
        hostname = HOSTNAME_LOCAL
    elif args.host == "contest":
        hostname = HOSTNAME_CONTEST
    else:
        print("No host chosen, selecting local")
        hostname = HOSTNAME_LOCAL

    if args.key is None:
        print("No key specified")
        sys.exit(0)

    key = args.key

    input_stream = reader(port=args.port)

    for i, packet in enumerate(input_stream):
        melvec = packet[4:-8]  # remove header and CRC from packet
        # melvec /= np.linalg.norm(melvec, keepdims=True)

        print(f"MEL Spectrogram #{i}")

        plt.figure()
        plot_specgram(
            melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            ax=plt.gca(),
            is_mel=True,
            title=f"MEL Spectrogram #{i}",
            xlabel="Mel vector",
        )
        plt.draw()
        plt.pause(0.001)
        plt.clf()

        if args.model:
            # print(f"{melvec.shape=}")
            prediction = classifier.predict((melvec,))
            prediction_proba = classifier.predict_proba((melvec,))
            print(f"{str(prediction[0])} : {prediction_proba[0]}")

            if args.host:
                requests.post(f"{hostname}/lelec210x/leaderboard/submit/\
                              {key}/{prediction}", timeout=1)

        # print(repr(melvec))


if __name__ == "__main__":
    main()
