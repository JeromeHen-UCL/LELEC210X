"""
Get the frequency response of the AFE + ADC by playing a sine wave recording the output for
multiple frequencies, extracting the amplitude of the recorded sine wave and interpolating the
frequency response.
"""


# TODO:
# have the same fs for the sounds to convert and the recordings


import argparse
import glob
import logging
import os
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler
import scipy as sp
import serial
from serial.tools import list_ports
import sounddevice as sd
import soundfile as sf


# Constants
PRINT_PREFIX = "SND:HEX:"
FREQ_SAMPLING = int(32e6/((15+1)*(195+1)))
VAL_MAX_ADC = 4096

N_REPETITIONS = 2
N_FREQUENCIES = 15

# Via datasheet
FREQ_START = 100
FREQ_STOP = FREQ_SAMPLING / 2

FILTER_WIDTH = 0.05

MEAN_MIN = 0.1

# Global variables
logger: logging.Logger

playback_stop_flag = threading.Event()


def check_args(args: argparse.Namespace) -> None:
    """
    Check the arguments of the script.

    Args:
        args (argparse.Namespace): The arguments of the script.
    """

    if not args.recordings_dir:
        logger.info("Recordings directory not specified. Saving in %s/", args.recordings_dir)

    if args.port is None and not args.load_recordings:
        ports = list_ports.comports()
        if not ports:
            logger.critical("No port specified and not loading recordings. No port found.")
            raise ValueError("No serial ports found.")

        ports_printable = "\n".join(port.device for port in ports)
        logger.critical("No port specified and not loading recordings. Please specify a port.\n"
                        "Available ports:\n%s", ports_printable)

        sys.exit(1)

    if os.path.exists(args.recordings_dir):
        logger.debug("Recordings directory %s/ exists.", args.recordings_dir)
    else:
        os.makedirs(args.recordings_dir)
        logger.debug("Recordings directory %s/ created.", args.recordings_dir)

    if args.load_recordings and args.keep_recordings:
        logger.warning("Both load_recordings and keep_recordings are set. "
                       "This combination of args is non coherent. Exiting.")
        sys.exit(1)


def sine_wave(frequency: int,
              t: np.array,
              amplitude: float = 1.0,
              phase_shift: float = 0.0) -> np.ndarray:
    """
    Generate a sine wave with specified frequency, amplitude and phase shift.

    Args:
        frequency (int): Frequency of the sine wave in Hz.
        t (np.array): Time vector.
        amplitude (float, optional): Amplitude of the wave. Defaults to 1.0.
        phase_shift (float, optional): Phase shift from 0 to 2pi. Defaults to 0.0.

    Returns:
        np.ndarray: The sine wave evaluated at times `t`.
    """

    return amplitude * np.sin(2 * np.pi * frequency * t + phase_shift).astype(np.float32)


def start_playback(frequency: int,
                   amplitude=1.0,
                   sample_rate=44100,
                   chunk_duration=0.1) -> threading.Thread:
    """
    Start playing a sine wave continuously in a background thread.

    Args:
        frequency (int): Frequency of the sine wave in Hz.
        amplitude (float, optional): Amplitude of the sine wave (0.0 to 1.0). Defaults to 1.0.
        sample_rate (int, optional): Sampling rate in Hz. Defaults to 44100.
        chunk_duration (float, optional): Duration of each chunk in seconds. Defaults to 0.1.

    Returns:
        threading.Thread: The thread playing the sine wave.
    """

    logger.info("Playing sine wave at %d Hz.", frequency)

    # Reset the stop event in case it was set earlier
    playback_stop_flag.clear()

    # Defint threading task
    def playback():
        chunk_size = int(sample_rate * chunk_duration)
        t0 = 0.0

        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            while not playback_stop_flag.is_set():
                t = np.linspace(t0, t0 + chunk_duration, chunk_size, endpoint=False)
                t0 += chunk_duration

                # wave = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
                wave = sine_wave(frequency, t, amplitude)
                stream.write(wave)

    _playback_thread = threading.Thread(target=playback, daemon=True)
    _playback_thread.start()

    return _playback_thread


def stop_playback(playback_thread: threading.Thread) -> None:
    """
    Stop the continuous sine wave playback.
    """

    playback_stop_flag.set()
    if playback_thread is not None:
        playback_thread.join()  # Wait for the thread to finish


def parse_buffer(line) -> bytes:
    """
    Parse the buffer from uart.

    Args:
        line (str): The line to parse.

    Returns:
        bytes: The parsed buffer.
    """

    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])

    return None


def reader(port):
    """
    Read the buffer from the uart.

    Args:
        port (_type_): Port to read from.

    Yields:
        str: The buffer.
    """

    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1042).decode("ascii")
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder("<")
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


def recording_filename(args: argparse.Namespace, frequency: int, repetition: int) -> str:
    """
    Get the filename of the recording.

    Args:
        args (argparse.Namespace): The arguments of the script.
        frequency (int): The frequency of the recording.
        repetition (int): The number of the current repetition.

    Returns:
        str: The filename of the recording.
    """

    return (f"{args.recordings_dir}/{args.recordings_prefix}-{repetition}-{frequency:05.0f}"
            f"{args.recordings_suffix}")


def main(args: argparse.Namespace) -> None:
    """
    Get the amplitude for multiple frequencies and interpolate the frequency response.

    Args:
        args (argparse.Namespace): The arguments of the script.
    """

    # Define the frequencies to play at
    frequencies = np.logspace(np.log10(FREQ_START), np.log10(
        FREQ_STOP), num=N_FREQUENCIES, base=10)

    # Play the sine wave at each frequency and record the output in a file
    if not args.load_recordings:
        if args.keep_recordings:
            start_repetition = len(glob.glob(f"{args.recordings_dir}/*.ogg")) // N_FREQUENCIES
        else:
            start_repetition = 0
        stop_repetition = start_repetition + N_REPETITIONS

        logger.info("Starting to record %d samples.", N_REPETITIONS * N_FREQUENCIES)
        for repetition in range(start_repetition, stop_repetition):
            for frequency in np.random.permutation(frequencies):
                playback_thread = start_playback(frequency, amplitude=0.3)

                output = None
                for msg in reader(args.port):
                    if isinstance(msg, np.ndarray):
                        output = msg
                        stop_playback(playback_thread)
                        break

                # save the output to a file
                buf = np.asarray(output, dtype=np.float64)
                buf = buf - np.mean(buf)
                sf.write(recording_filename(args, frequency, repetition), buf, FREQ_SAMPLING)

    n_repetitions = len(glob.glob(f"{args.recordings_dir}/*.ogg")) // N_FREQUENCIES

    # Extract amplitudes from the recordings
    freqs_amplitudes = np.empty((N_FREQUENCIES, 2))  # [mean, std] for every frequency
    for i, frequency in enumerate(frequencies):
        # extract the amplitude of the sine wave and append it to the list
        amplitudes = np.zeros(n_repetitions)
        for repetition in range(n_repetitions):
            filename = recording_filename(args, frequency, repetition)
            buf, _ = sf.read(filename)

            # Get the value of the recorded sine wave
            fft = np.abs(np.fft.fft(buf))
            x_fft = np.fft.fftfreq(len(fft), 1/FREQ_SAMPLING)

            # filter out low frequencies
            filtered_fft = fft.copy()
            filtered_fft[np.abs(x_fft) < frequency * (1-FILTER_WIDTH/2)] = 0  # low pass filter
            filtered_fft[np.abs(x_fft) > frequency * (1+FILTER_WIDTH/2)] = 0  # high pass filter

            major_freq = np.abs(x_fft[np.argmax(filtered_fft)])

            amplitudes[repetition] = np.max(filtered_fft)

            if args.plot_recordings:
                _, axes = plt.subplots(1, 2, figsize=(10, 4))

                # Time plot
                x = np.linspace(0, len(buf) - 1, len(buf)) * 1 / FREQ_SAMPLING
                axes[0].plot(x, buf, color="b")
                axes[0].set_title(f"File {filename} content")
                axes[0].set_xlabel("Time (s)")
                axes[0].set_ylabel("[/]")
                # axes[0].legend()
                axes[0].grid(True)

                # Frequency plot
                axes[1].plot(x_fft, fft,
                             label=f"FFT ({np.abs(x_fft[np.argmax(fft)]):5.0f} Hz)", color="r")
                axes[1].plot(x_fft, filtered_fft,
                             label=f"Filtered FFT ({major_freq:5.0f} Hz)",
                             color="g")
                axes[1].set_title("FFT of the file content")
                axes[1].set_xlabel("Frequency (Hz)")
                axes[1].set_ylabel("Amplitude")
                axes[1].legend()
                axes[1].grid(True)

                plt.tight_layout()
                plt.show()

        # major_freqs.append(np.abs(x_fft[np.argmax(fft)]))
        freqs_amplitudes[i] = (np.mean(amplitudes), np.std(amplitudes))

    means = freqs_amplitudes[:, 0]  # Extract means
    means[means < MEAN_MIN] = MEAN_MIN
    logger.info("Means: %s", means)

    stds = freqs_amplitudes[:, 1]  # Extract standard deviations
    logger.info("Stds: %s", stds)

    # Normalize the means
    means_normalized = means / np.max(means)
    stds_normalized = stds / np.max(means)

    if args.plot_response:
        plt.loglog(frequencies, means_normalized, "o")
        plt.fill_between(frequencies, means_normalized - stds_normalized,
                         means_normalized + stds_normalized, color="blue", alpha=0.3)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency response of the AFE + ADC")
        plt.grid(True)
        plt.show()

    means_normalized[-1] = 0  # Set the last value to 0 to respect Type II FIR filter

    # Get the FIR coefficients of the frequency response
    means_interp = np.interp(
        np.linspace(0, FREQ_SAMPLING/2, 512),
        np.hstack(([0], frequencies, [FREQ_SAMPLING/2])),  # Add 0 Hz and Nyquist
        # Assume constant end values
        np.hstack(([means_normalized[0]], means_normalized, [means_normalized[-1]]))
    )

    fir_coefficients = sp.signal.firwin2(
        300, np.linspace(0, FREQ_SAMPLING/2, 512) / (FREQ_SAMPLING/2), means_interp)
    logger.info("FIR coefficients: %s", fir_coefficients)

    if args.plot_FIR:
        plt.stem(sp.signal.minimum_phase(fir_coefficients))
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("FIR coefficients")
        plt.grid(True)
        plt.show()

    # Save the FIR coefficients
    np.save("fir_coefficients.npy", sp.signal.minimum_phase(fir_coefficients))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the frequency response of the AFE + ADC.")
    parser.add_argument("--port", type=str, default=None,
                        help="The port to read the data from.")
    parser.add_argument("--keep_recordings", action='store_true',
                        help="If the previous recording should be kept.")
    parser.add_argument("--load_recordings", action='store_true',
                        help="If the recordings should be loaded from a file.")
    parser.add_argument("--recordings_dir", type=str, default="recordings",
                        help="The directory to store/load the recordings from.")
    parser.add_argument("--recordings_prefix", type=str, default="recording",
                        help="The prefix of the recordings file.")
    parser.add_argument("--recordings_suffix", type=str, default="Hz.ogg",
                        help="The suffix of the recordings file.")
    parser.add_argument("--plot_recordings", action='store_true',
                        help="If the recordings should be ploted")
    parser.add_argument("--plot_response", action='store_true',
                        help="If the frequency response should be ploted")
    parser.add_argument("--plot_FIR", action='store_true',
                        help="If the FIR coefficients should be ploted")
    parser.add_argument("--verbose", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="If the verbose mode should be activated")
    main_args = parser.parse_args()

    handler = RichHandler(keywords=RichHandler.KEYWORDS)
    logging.basicConfig(level=main_args.verbose, format="%(message)s",
                        datefmt="[%X]", handlers=[handler])
    logger = logging.getLogger("LELEC210X")

    check_args(main_args)

    main(main_args)
