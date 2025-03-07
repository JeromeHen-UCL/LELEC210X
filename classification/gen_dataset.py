"""
Generate the dataset for the classification task. Has the ability to generate new feature vectors
using data augmentation techniques like:
    - delay of the audio
    - varying the SNR (sound to background ratio)
    - adding white noise (white before filtering!)

Each augmentation possibility is performed in combination with the others. The output is saved in
the `output_dir` directory as numpy ndarrays, with non scaled values being in mels (like the ones
received from the MCU).
"""


# TODO:
# - Implement noise (take distribution of noise from Nucleo)


import argparse
import glob
import logging
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler
from scipy import signal as sg
import soundfile as sf


CLASSES = {"chainsaw", "fire", "fireworks", "gun"}

INPUT_AUDIO_DURATION = 5  # seconds
SOUND_TARGET_DB = 50  # dB

MELS_NFFT = 512  # number of fft components in transform
MELS_LEN = 20  # number of mels per melvector
MELVECS_LEN = 20  # number of melvectors in a melmatrix

SOUND_FS = 10200  # Hz
SOUND_DURATION = MELS_NFFT * MELVECS_LEN / SOUND_FS  # seconds
SOUND_LEN = int(SOUND_FS * SOUND_DURATION)

AUG_DELAY_MIN_FRAC = -.8  # fraction of the sound length
AUG_DELAY_MAX_FRAC = .1  # fraction of the sound length
AUG_DELAY_NUM = 20  # number of possible delays to apply

AUG_NOISE_MIN = 0  # ?
AUG_NOISE_MAX = 1  # ?
AUG_NOISE_NUM = 1  # number of possible noise levels to apply

AUG_ECHO_MIN = 0  # ?
AUG_ECHO_MAX = 1  # ?
AUG_ECHO_NUM = 1  # number of possible echo levels to apply

AUG_SNR_MIN = 10  # dB
AUG_SNR_MAX = 30  # dB
AUG_SNR_NUM = 20  # number of possible SNR levels to apply

AUG_PINK_SNR_MIN = -30  # Pink noise SNR relative to background (in dB)
AUG_PINK_SNR_MAX = 0   # Maximum pink noise SNR (relative to background)
AUG_PINK_SNR_NUM = 20   # Number of pink noise SNR values

DELAY_PARAMS = np.linspace(AUG_DELAY_MIN_FRAC * INPUT_AUDIO_DURATION,
                           AUG_DELAY_MAX_FRAC * INPUT_AUDIO_DURATION,
                           AUG_DELAY_NUM)
NOISE_PARAMS = np.linspace(AUG_NOISE_MIN, AUG_NOISE_MAX, AUG_NOISE_NUM)
ECHO_PARAMS = np.linspace(AUG_ECHO_MIN, AUG_ECHO_MAX, AUG_ECHO_NUM)
SNR_PARAMS = np.linspace(AUG_SNR_MIN, AUG_SNR_MAX, AUG_SNR_NUM)
PINK_SNR_PARAMS = np.linspace(AUG_PINK_SNR_MIN, AUG_PINK_SNR_MAX, AUG_PINK_SNR_NUM)


logger: logging.Logger
logging_period: float


def load_audios(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the audios from filenames.

    Args:
        args (argparse.Namespace): the args to the script

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: the input audios, the input labels and the
        background
    """

    if args.load:
        logger.info("Loading background")
        background = np.load(os.path.join(args.input_dir, "background.npy"))

        logger.info("Loading input audios")
        input_audios = np.load(os.path.join(args.input_dir, "input_audios.npy"), allow_pickle=True)
        input_labels = np.load(os.path.join(args.input_dir, "input_labels.npy"), allow_pickle=True)
    else:
        logger.info("Loading background")
        background = get_clean_audio(args.background_filepath)
        np.save(os.path.join(args.input_dir, "background.npy"), background)

        input_filenames = glob.glob(os.path.join(args.input_dir, "*.wav"))
        blacklist_filenames = {"background.wav"}
        input_filenames = [f for f in input_filenames
                           if os.path.basename(f) not in blacklist_filenames
                           and os.path.basename(f).split('_')[0] in CLASSES]

        input_len = len(input_filenames)

        input_audios = np.ndarray(input_len, dtype=object)
        input_labels = np.ndarray(input_len, dtype=object)
        for i, input_filename in enumerate(input_filenames):
            if i % logging_period == 0 or i == input_len - 1:
                logger.info("Loading (%.0f%%) %s", 100 * (i + 1)/len(input_filenames),
                            os.path.basename(input_filename))
            # Load, normalize and resample the audio
            sig = get_clean_audio(input_filename)

            # Store the normalized and resampled audio
            input_audios[i] = sig

            # Extract the label from filename
            input_labels[i] = os.path.basename(input_filename).split('_')[0]

        np.save(os.path.join(args.input_dir, "input_audios.npy"), input_audios)
        np.save(os.path.join(args.input_dir, "input_labels.npy"), input_labels)

    assert isinstance(input_audios, np.ndarray)
    assert isinstance(input_labels, np.ndarray)
    assert isinstance(background, np.ndarray)

    return (input_audios, input_labels, background)


def get_clean_audio(filepath: str) -> np.ndarray:
    """
    Load, normalize and resample an audio file.

    Args:
        filepath (str): The path to the audio file.

    Returns:
        np.ndarray: The audio signal.
    """

    # Load the audio
    sig, sr = sf.read(filepath)

    # Convert to mono if needed
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)

    # Normalize to prevent clipping
    sig /= np.sqrt(np.sum(np.abs(sig) ** 2))
    sig *= np.sqrt(10 ** (SOUND_TARGET_DB / 10))

    # Resample
    if sr != SOUND_FS:
        gcd = np.gcd(sr, SOUND_FS)  # Greatest common divisor
        up = SOUND_FS // gcd
        down = sr // gcd

        # Better simulation of the ADC anti-aliasing filter than sg.resample
        sig = sg.resample_poly(sig, up, down, window=('kaiser', 5.0))

    assert isinstance(sig, np.ndarray)

    return sig


def gen_pink_noise(length: int) -> np.ndarray:
    """
    Generate pink noise using an FFT-based method.

    Args:
        length (int): Number of samples

    Returns:
        np.ndarray: Pink noise at given 
    """

    # Generate white noise in frequency domain
    white_noise = np.random.randn(length)

    # Compute FFT
    fft_spectrum = np.fft.rfft(white_noise)

    # Create 1/f filter (inverse frequency scaling)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(freqs)

    # Apply filter and transform back
    pink_spectrum = fft_spectrum * pink_filter
    pink_noise = np.fft.irfft(pink_spectrum)

    # Normalize to unit variance
    pink_noise /= np.std(pink_noise)

    return pink_noise


def gen_audio(input_audio: np.ndarray,
              background: np.ndarray,
              delay: float,
              snr_db: float,
              add_background=True) -> np.ndarray:
    """
    Generate a dataset audio from an input audio, a delay and background to audio SNR.

    Args:
        input_audio (np.ndarray): the pure input audio from
        background (np.ndarray): the background noise
        delay (float): delay to apply to the input audio
        snr_db (float): the SNR (audio to background power) in db
        add_background (bool, optional): add the background sound or not. Defaults to True.

    Raises:
        ValueError: if the background is too small

    Returns:
        np.ndarray: the generated audio
    """
    # TODO: add AWGN and Pink noises

    # Ensure background is long enough
    if len(background) < SOUND_LEN:
        raise ValueError("Background noise must be at least as long as the desired output length.")

    # Convert delay to samples
    delay_samples = int(delay * SOUND_FS)

    if delay_samples > 0:
        # Apply positive delay by padding at the beginning
        delayed_audio = np.concatenate((np.zeros(delay_samples), input_audio))
    else:
        # Apply negative delay by trimming the start
        delayed_audio = input_audio[abs(delay_samples):] if abs(
            delay_samples) < len(input_audio) else np.array([])

    # Trim or pad delayed_audio to match the desired length
    if len(delayed_audio) > SOUND_LEN:
        delayed_audio = delayed_audio[:SOUND_LEN]
    else:
        delayed_audio = np.pad(delayed_audio, (0, SOUND_LEN - len(delayed_audio)))

    if add_background:
        # Pick a random portion of background noise
        background_start_idx = np.random.randint(0, len(background) - SOUND_LEN + 1)
        background = background[background_start_idx:background_start_idx + SOUND_LEN]

        # Compute input and background power
        # Avoid division by zero
        input_power = np.mean(delayed_audio ** 2) if np.any(delayed_audio) else 1e-10
        background_power = np.mean(background ** 2) if np.any(background) else 1e-10

        # Compute scaling factor for background noise to achieve desired SNR
        snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
        noise_scaling_factor = np.sqrt(input_power / (snr_linear * background_power))

        # Scale background noise
        adjusted_background = background * noise_scaling_factor

        # Mix input audio with background noise
        output_audio = delayed_audio + adjusted_background

        return output_audio

    return delayed_audio


def get_db_audios(args: argparse.Namespace,
                  input_audios: np.ndarray,
                  input_labels: np.ndarray,
                  background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the database of generated sounds from input audios and modifyers.

    Args:
        args (argparse.Namespace): the args to the script
        input_audios (np.ndarray): array of the input audios
        input_labels (np.ndarray): array of the input labels
        background (np.ndarray): background audio

    Returns:
        tuple[np.ndarray, np.ndarray]: the database audios and their labels
    """

    input_len = len(input_labels)

    db_audios = np.empty((args.db_len, int(SOUND_DURATION * SOUND_FS)), dtype=float)
    db_labels = np.empty(args.db_len, dtype=object)

    background_frac = 1/(len(CLASSES)+1)

    # Add background audios
    for i in range(int(background_frac * args.db_len)):

        delay_param = - np.random.randint(0, len(background) - SOUND_LEN + 1) / SOUND_FS
        noise_param = np.random.choice(NOISE_PARAMS)

        # generate a time shifted with noise
        if i % logging_period == 0 or i == args.db_len - 1:
            logger.info("Processing (%2.0f%%)\t\tdelay=%.2f,\tnoise=%.2f",
                        100 * (i + 1)/args.db_len, delay_param, noise_param)

        db_audios[i] = gen_audio(background, background, delay_param, 1, add_background=False)
        db_labels[i] = "background"

    # Add event audios
    for i in range(int(background_frac * args.db_len), args.db_len):
        # If need to reshuffle the audio and labels
        if i % input_len == 0:
            logger.debug("Reshuffling at %d", i)

            shuffled_indices = np.random.permutation(input_len)

            input_audios = input_audios[shuffled_indices]
            input_labels = input_labels[shuffled_indices]

        # Compute an index to prevent overflow
        index = i % input_len

        input_audio = input_audios[index]
        input_label = input_labels[index]

        # Choose augmentation args
        delay_param = np.random.choice(DELAY_PARAMS)
        noise_param = np.random.choice(NOISE_PARAMS)
        echo_param = np.random.choice(ECHO_PARAMS)
        snr_param = np.random.choice(SNR_PARAMS)

        if i % logging_period == 0 or i == args.db_len - 1:
            logger.info("Processing (%.0f%%)\t\tdelay=%.2f,\tnoise=%.2f,\techo=%.2f,\tSNR=%.2f",
                        100 * (i + 1)/args.db_len, delay_param,
                        noise_param, echo_param, snr_param)

        output_audio = gen_audio(input_audio, background, delay_param, snr_param)

        db_audios[i] = output_audio
        db_labels[i] = input_label

    return db_audios, db_labels


def filter_audio(audio: np.ndarray, fir_coefficients: np.ndarray) -> np.ndarray:
    """
    Filter an audio using the given FIR coefficients.

    Args:
        audio(np.ndarray): the content of the audio to filter
        fir_coefficients(np.ndarray): the FIR filter coefficients

    Returns:
        np.ndarray: the filtered audio
    """

    # Normalize to prevent clipping
    audio = audio / np.linalg.norm(audio, keepdims=True)

    # Apply FIR filter using lfilter (causal)
    filtered_audio = sg.lfilter(fir_coefficients, [1.0], audio)

    assert isinstance(filtered_audio, np.ndarray)

    # Normalize to prevent clipping
    # FIXME?
    filtered_audio = filtered_audio / np.linalg.norm(filtered_audio, keepdims=True)

    return filtered_audio


def get_mels(audio: np.ndarray) -> np.ndarray:
    """
    Transform an audio in a melsmatrix representation.

    Args:
        audio (np.ndarray): the audio to modify

    Returns:
        np.ndarray: the melsmatrix of the audio
    """

    # STEP 1: Windowing
    # Reshape the signal with a piece for each row
    audiomat = np.reshape(audio, (MELS_LEN, MELS_NFFT))
    audioham = audiomat * np.hamming(MELS_NFFT)  # Windowing. Hamming, Hanning, Blackman,..

    # STEP 2: Compute FFT
    # FFT row by row
    stft = np.fft.fft(audioham, axis=1)

    # STEP 3: Normalize
    # Taking only positive frequencies and computing the magnitude
    stft = stft[:, : MELS_NFFT // 2].T
    max_value = np.max(np.abs(stft))

    stft /= max_value

    stft = np.abs(stft)

    stft *= max_value

    # STEP 4: Apply MEL transformation
    hz2mel_mat = librosa.filters.mel(sr=SOUND_FS, n_fft=MELS_NFFT, n_mels=MELS_LEN)
    hz2mel_mat = hz2mel_mat / np.max(hz2mel_mat)

    melspec = hz2mel_mat[:, : MELS_NFFT // 2] @ stft

    return melspec


def main(args: argparse.Namespace) -> None:
    """
    The main function.

    Args:
        args(argparse.Namespace): The arguments of the script.
    """

    logger.info("Loading FIR coefficients")
    fir_coefficients = np.load(args.fir_filename)

    # [1] Extract the sound features
    input_audios, input_labels, background = load_audios(args)

    # [2] Perform data augmentations
    # Number of available parameter combinations
    combinations_len = len(DELAY_PARAMS) * len(NOISE_PARAMS) * len(ECHO_PARAMS) * len(SNR_PARAMS)
    logger.info("Params have %d combinations", combinations_len)

    db_audios, db_labels = get_db_audios(args, input_audios, input_labels, background)

    # [3] Apply FIR response
    for i, audio in enumerate(db_audios):
        if i % logging_period == 0 or i == len(db_audios) - 1:
            logger.info("Filtering (%.0f%%)", 100 * (i + 1)/args.db_len)

        db_audios[i] = filter_audio(audio, fir_coefficients)

    # [4] Transform into melvectors
    db_mels = np.empty((args.db_len, MELVECS_LEN * MELS_LEN))
    for i, audio in enumerate(db_audios):
        if i % logging_period == 0 or i == len(db_audios) - 1:
            logger.info("Converting (%.0f%%)", 100 * (i + 1)/args.db_len)

        db_mels[i] = get_mels(audio).flatten()
        if args.plot_mels:
            plt.imshow(db_mels[i].reshape(MELS_LEN, MELVECS_LEN), cmap="jet",
                       aspect="auto", extent=(0, MELVECS_LEN, MELS_LEN, 0))
            plt.gca().invert_yaxis()
            plt.title(f"{db_labels[i].capitalize()}")
            plt.show()

    # [5] Save the feature matrix and labels
    np.save(os.path.join(args.output_dir, "db_mels.npy"), db_mels)
    np.save(os.path.join(args.output_dir, "db_labels.npy"), db_labels)


if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser(
        description="Generate the dataset for the classification task.")

    parser.add_argument(
        "--fir_filename",
        type=str,
        default="fir_coefficients.npy",
        help="Filename of the FIR filter coefficients")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="src/classification/datasets/soundfiles",
        help="Directory containing the audio files to be used")
    parser.add_argument(
        "--background_filepath",
        type=str,
        default="src/classification/datasets/soundfiles/background.wav",
        help="Filepath of thebackground noise")
    parser.add_argument(
        "--db_len",
        type=int,
        default=1000,
        help="Number of elements in the generated database")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the input sounds from saved .npy")
    parser.add_argument(
        "--plot_mels",
        action="store_true",
        help="Plot the melspecs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save the feature matrix")
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="If the verbose mode should be activated")

    main_args = parser.parse_args()

    logging_period = main_args.db_len * .05

    handler = RichHandler(keywords=RichHandler.KEYWORDS)
    logging.basicConfig(level=main_args.verbose, format="%(message)s",
                        datefmt="[%X]", handlers=[handler])
    logger = logging.getLogger("LELEC210X")

    main(main_args)
