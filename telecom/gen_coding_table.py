"""
Generate a coding table from synthetic payloads. This table will be used to reduce the size of the
packets sent while keeping the same information content. This generation will use the alrogithm of 
TODO.
"""


import argparse
import heapq
import pickle
from struct import pack, unpack
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline


def float_to_q15_bytes(float_values: list[float]) -> bytes:
    """
    Convert a list of floats to Q15 bytes.

    Args:
        float_values (list[float]): the values to convert.

    Returns:
        bytes: the converted values
    """

    q15_vals = []
    for f in float_values:
        # Convert to Q15 integer
        q15 = int(round(f * 2**15))

        # Clip to Q15 range
        q15 = max(-2**15, min(2**15 - 1, q15))

        # Pack as signed 16-bit little-endian
        q15_vals.append(q15)
    # Pack into bytes

    return b''.join(pack('>h', val) for val in q15_vals)  # '>h' = big endian signed short


def q15_bytes_to_floats(q15_values: bytes) -> list[float]:
    """
    Convert a list of Q15 bytes to floats.

    Args:
        q15_values (bytes): the values to convert.

    Returns:
        list[float]: the converted values
    """

    count = len(q15_values) // 2
    ints = unpack('>' + 'h' * count, q15_values)  # Big endian signed shorts
    return [i / 32768 for i in ints]


def gen_payload(melspec: np.ndarray, pipeline: Pipeline) -> np.ndarray:
    """
    Generate a payload from a melspectrogram using a given pipeline.

    Args:
        melspec (np.ndarray): the payload to transform.
        pipeline (Pipeline): the pipeline to use for transformation.

    Returns:
        np.ndarray: a correctly formatted packet
    """

    #   Field           Length (bytes)  Encoding    Description
    #
    #   r               1                           Reserved, set to 0.
    #   emitter_id      1               BE          Unique id of the sensor node.
    #   payload_length  2               BE          Length of app_data (in bytes).
    #   packet_serial   4               BE          Unique and incrementing id of the packet.
    #   app_data        any                         The feature vectors.
    #   tag             16                          Message authentication code (MAC).

    melspec_transf = pipeline.transform((melspec,))[0]
    payload = float_to_q15_bytes(melspec_transf)

    return payload


def get_freq_table(payloads: np.ndarray, key_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency table from the payloads.

    Args:
        payloads (np.ndarray): the payloads to use for generation. Values should be as Q15 bytes.
        key_length (int): the size of the key in the coding table (in bytes).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - symbols (np.ndarray): the symbols of the coding table.
            - frequencies (np.ndarray): the matching frequencies of the symbols.
    """

    num_payloads, payload_len = payloads.shape
    chunk_count = int(np.ceil(payload_len / key_length))
    padded_len = chunk_count * key_length

    # [1] Pad payloads if needed
    if padded_len > payload_len:
        pad_width = padded_len - payload_len
        payloads = np.pad(payloads, ((0, 0), (0, pad_width)), constant_values=0)

    # [2] Exract the chunks by reshaping
    chunks = payloads.reshape(num_payloads, chunk_count, key_length)
    all_chunks = chunks.reshape(-1, key_length)  # Flatten across all payloads

    # [3] Generate the coding table
    # Convert each chunk to a fixed-size composite dtype for fast uniqueness lookup
    key_dtype = np.dtype((np.void, key_length))  # Treat each row as a raw byte string
    chunk_keys = all_chunks.view(dtype=key_dtype).ravel()

    # Use np.unique for faster frequency counting
    unique_keys, counts = np.unique(chunk_keys, return_counts=True)

    # Reconstruct symbols as uint8 arrays from the unique void records
    symbols = np.frombuffer(b''.join(unique_keys.tolist()), dtype=np.uint8).reshape(-1, key_length)

    # Normalize frequencies
    frequencies = counts.astype(np.float32)
    frequencies /= frequencies.sum()

    return symbols, frequencies


def gen_huffman_table(symbols: np.ndarray, frequencies: np.ndarray) -> dict:
    """
    Generate a Huffman coding table from symbols and their frequencies.

    Args:
        symbols (np.ndarray): shape (N, key_length), each row is a symbol (array of uint8)
        frequencies (np.ndarray): shape (N,), normalized frequencies summing to 1.0

    Returns:
        dict: mapping from symbol (as bytes) to Huffman code (as bitstring, e.g., '0110')
    """

    # [1] Initialize heap with (prob, id, node), where node is either symbol or tuple of nodes
    heap = [(freq, i, symbol) for i, (freq, symbol) in enumerate(zip(frequencies, symbols))]
    heapq.heapify(heap)

    # [2] Build tree
    while len(heap) > 1:
        freq1, _, left = heapq.heappop(heap)
        freq2, _, right = heapq.heappop(heap)
        merged_node = (left, right)
        heapq.heappush(heap, (freq1 + freq2, id(merged_node), merged_node))

    # [3] DFS explore and assign codes
    huffman_codes = {}

    def assign_codes(node, code):
        if isinstance(node, np.ndarray):  # It's a symbol
            huffman_codes[bytes(node)] = code
        else:
            left, right = node
            assign_codes(left, code + '0')
            assign_codes(right, code + '1')

    _, _, root = heap[0]
    assign_codes(root, '')

    return huffman_codes


def main(args: argparse.Namespace) -> None:
    """
    Main function.

    Args:
        args (argparse.Namespace): the args of the script.
    """

    # [0] Load the database and the pipeline
    db_mels = np.load(args.db_mels)
    with open(args.pipeline, "rb") as pipeline_file:
        pipeline = pickle.load(pipeline_file)

    payload_size = pipeline[1].n_components * 2  # 2 bytes per Q15 value

    # [1] Generate payloads
    if args.load:
        payloads = np.load(args.payloads)
    else:
        payloads = np.zeros((len(db_mels), payload_size), dtype=np.uint8)

        for i, melspec in enumerate(db_mels):
            payload = gen_payload(melspec, pipeline[:2])
            payloads[i] = np.frombuffer(payload, dtype=np.uint8)

        np.save(args.payloads, payloads)

    # [2] Generate the coding table
    payload_gains = []
    for key_length in range(1, 33):
        symbols, frequencies = get_freq_table(payloads, key_length)
        coding_table = gen_huffman_table(symbols, frequencies)

        fixed_rate = 8 * key_length  # bits in non-compressed payload
        huffman_rate = sum(frequencies[i] * len(coding_table[bytes(symbols[i])])
                           for i in range(len(symbols)))  # mean bits num per compressed payload
        gain = fixed_rate - huffman_rate

        chunks_per_payload = int(np.ceil(payload_size / key_length))
        payload_gain = gain * chunks_per_payload  # bits gained by encoding on a payload
        payload_gains.append(payload_gain)

        print(f"[key_length={key_length}]")
        print(f"\tRate (Huffman): {huffman_rate:.3f} bits/chunk")
        print(f"\tFixed Rate:     {fixed_rate} bits/chunk")
        print(f"\tGain:           {gain:.3f} bits/chunk")
        print(f"\tPayload Gain:   {payload_gain:.3f} bits/payload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source code generation for encoding packets.")
    parser.add_argument(
        "--db_mels",
        type=str,
        default="../classification/db_mels.npy",
        help="Path to the synthetic melspectrograms database.")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="../classification/pipeline.pickle",
        help="Path to the PCA model.")
    parser.add_argument(
        "--payloads",
        type=str,
        default="payloads.npy",
        help="Path to the previously generated payloads.")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the input sounds from saved .npy")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results.")

    main_args = parser.parse_args()

    main(main_args)
