"""
Simulator for an AGC gain controll. This will put as input a pulse of 0.2 s of a sine wave of
frequency 1kHz every second.
"""


import argparse

import numpy as np
import matplotlib.pyplot as plt


SAMPLE_RATE = 1e3  # sample rate of the recordings
SINE_FREQ = 1e2  # freq of the sine wave
SINE_AMPLITUDE = .2  # amplitude of the sine pulse
PULSE_DURATION = .2  # duration of the sine pulse, in seconds


def gen_input_signal(period_number: int) -> np.ndarray:
    """
    Generate a signal with noise and a pulse of high frequency sine wave of .02 s every second, for
    a specified overall time.

    Args:
        period_number (int): the number of periods to generate

    Returns:
        np.ndarray: a numpy array with specified length (number of periods)
    """

    total_samples = int(SAMPLE_RATE * period_number)
    noise_samples = int(SAMPLE_RATE * (1-PULSE_DURATION))
    pulse_samples = int(SAMPLE_RATE * PULSE_DURATION)

    t = np.arange(total_samples) / SAMPLE_RATE

    sine = SINE_AMPLITUDE * np.sin(2 * np.pi * SINE_FREQ * t)

    output = np.random.normal(0, 0.005, total_samples)
    for second in range(period_number):
        start_idx = int(second * SAMPLE_RATE + noise_samples)
        end_idx = start_idx + pulse_samples

        output[start_idx: end_idx] += sine[start_idx: end_idx]

    return output, t


def agc_transform(input_signal: np.ndarray, k: int, avg: int, a_desired: float) -> np.ndarray:
    """
    Apply an agc transformation to the specified input signal.

    Args:
        input_signal (np.ndarray): the input signal to modify
        k (int): the gain of the feedback loop, divided by ???
        avg (int): number of samples to average over (2**(7+avg))
        a_desired (float): the desired RMS mean output amplitude

    Returns:
        np.ndarray: the filtered signal
    """

    mean_buffer_len = 2**(7 + avg)
    mean_buffer = np.zeros((mean_buffer_len))

    gain = 1.0

    output_signal = np.zeros_like(input_signal)

    gains = np.zeros_like(input_signal)
    means = np.zeros_like(input_signal)

    # first fill the buffer without update process
    # for i, sample in enumerate(input_signal[:mean_buffer_len]):
    #     # Set the output signal sample
    #     output_signal[i] = gain * sample
    #     gains[i] = gain

    #     mean_buffer[i % mean_buffer_len] = (gain * sample) ** 2

    for i, sample in enumerate(input_signal):
        # Set the output signal sample
        output_signal[i + 0] = gain * sample
        gains[i + 0] = gain

        # Update the gain via feedback
        mean_buffer[i % mean_buffer_len] = (gain * sample) ** 2

        current_mean = np.sqrt(np.mean(mean_buffer) + 1e-8)
        means[i + 0] = current_mean

        delta = a_desired - current_mean

        gain += k/(2**17) * delta
        gain = np.clip(gain, 0.01, 1000.0)

    return output_signal, gains, means


def main(args: argparse.Namespace) -> None:
    """
    Main function.

    Args:
        args (argparse.Namespace): the args of the script.
    """

    k = args.k  # gain of the AGC
    avg = args.avg  # number of samples to average over
    a_desired = args.a_desired  # desired amplitude of the output signal

    initial_signal, t = gen_input_signal(6)  # generate the input signal

    final_signal, gains, means = agc_transform(
        initial_signal, k=k, avg=avg, a_desired=a_desired)

    plt.figure(figsize=(10, 5))
    plt.plot(t, final_signal, color="blue", label="final_signal", alpha=1)
    plt.plot(t, initial_signal, color="green", label="initial_signal", alpha=1)
    plt.axhline(y=a_desired, color="red", linestyle="--", linewidth=1)
    plt.plot(t, means, color="red", label="RMS mean")
    # plt.plot(t, gains, label="gains", alpha=.5)
    plt.ylim((0, plt.ylim()[1]))
    plt.ylabel("Amplitude [AU]")
    plt.xlabel("Time [s]")

    plt.legend(loc=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig("agc_sim_optimal.svg")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="agc simulator")

    parser.add_argument(
        "--k",
        type=int,
        default=2000,
        help="feedback gain of the AGC divided by ???"
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=3,
        help="log of the number of samples to average over. This value is 2**(7 + avg) samples."
    )
    parser.add_argument(
        "--a_desired",
        type=float,
        default=.32,
        help="desired amplitude of the output signal"
    )

    main_args = parser.parse_args()

    main(main_args)
