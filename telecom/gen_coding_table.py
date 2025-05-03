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


EPSILON = 1e-6  # Small value to avoid q15 conversion overflow


def float_to_q15_bytes(float_values: list[float]) -> bytes:
    """
    Convert a list of floats to Q15 bytes.

    Args:
        float_values (list[float]): the values to convert.

    Returns:
        bytes: the converted values
    """

    # Clip values to Q15 range [-1.0, 1.0 - EPSILON]
    clipped = np.clip(float_values, -1.0, 1.0 - EPSILON)

    # Scale and round to Q15 range
    q15_vals = np.round(clipped * (2 ** 15 - 1)).astype(np.int16)

    # Convert to big-endian bytes
    return q15_vals.astype('>i2').tobytes()


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


def normalize_payload(payload: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize a payload to the range [-1, 1].
    This is done by subtracting the minimum value and dividing by the range (max - min).

    Args:
        payload (np.ndarray): the payload to normalize.

    Returns:
        Tuple[np.ndarray, float, float]: the normalized payload
    """

    min_val = payload.min()
    max_val = payload.max()

    if max_val == min_val:
        return np.zeros_like(payload), 0.0, 1.0  # Avoid division by zero

    scaled = 2 * (payload - min_val) / (max_val - min_val + EPSILON) - 1

    return scaled, min_val, max_val


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
    #   conv_addr       4               BE          Multiplicative factor to q15 conversion
    #   app_data        any                         The feature vectors.
    #   tag             16                          Message authentication code (MAC).

    # melspec_transf is not from -1 to 1 (not included),
    # and thus a transformation to q15 would overflows
    melspec_transf = pipeline.transform((melspec,))[0]

    # Normalize the payload to Q15 range [-1, 1]
    melspec_transf, _, _ = normalize_payload(melspec_transf)

    assert np.all(melspec_transf < 1.0), "Some values still exceed 1.0!"
    assert np.all(melspec_transf >= -1.0), "Some values are below -1.0!"

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
    # [3.1] Convert each chunk to a fixed-size composite dtype for fast uniqueness lookup
    key_dtype = np.dtype((np.void, key_length))  # Treat each row as a raw byte string
    chunk_keys = all_chunks.view(dtype=key_dtype).ravel()

    # Use np.unique for faster frequency counting
    unique_keys, counts = np.unique(chunk_keys, return_counts=True)

    # [3.2] Check if every possible symbol at least present once
    unique_keys_set = set(bytes(key) for key in unique_keys)
    missing_symbols = []
    missing_counts = []

    for symbol in range(256**key_length):
        symbol_bytes = symbol.to_bytes(key_length, byteorder='big')
        if symbol_bytes not in unique_keys_set:
            # print(f"Warning: Symbol {symbol} ({symbol_bytes.hex()}) not found in the payloads.")
            missing_symbols.append(symbol_bytes)
            missing_counts.append(0)  # Assign count 0 for missing symbol

    if missing_symbols:
        # Convert missing symbols to a NumPy array of the same type as unique_keys
        missing_symbols_array = np.array(missing_symbols, dtype=unique_keys.dtype)
        missing_counts_array = np.array(missing_counts, dtype=counts.dtype)

        # Concatenate existing symbols with the missing symbols
        unique_keys = np.concatenate([unique_keys, missing_symbols_array])
        counts = np.concatenate([counts, missing_counts_array])

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


def save_huffman_table_to_h(
        coding_table: dict[bytes, str],
        key_length: int,
        var_name="huffman_table"):
    """
    Export a Huffman table as a C const array.

    Args:
        coding_table (dict[bytes, str]): The Huffman coding table: symbol → bitstring
        key_length (int): The number of bytes in each symbol
        var_name (str): The variable name for the C array
    """

    header = f"""\
#if !defined({var_name.upper()}_H)
#define {var_name.upper()}_H

#include <stdint.h>

typedef struct
{{
    uint8_t symbol[{key_length}];
    uint32_t codeword;
    uint8_t bit_length;
}} HuffmanEntry;

const HuffmanEntry {var_name}[] = {{\n"""

    body_lines = []
    for symbol, bitstr in sorted(coding_table.items(), key=lambda x: x[0]):
        # Convert symbol to comma-separated hex
        symbol_hex = ', '.join(f'0x{b:02X}' for b in symbol)
        codeword_int = int(bitstr, 2)
        bit_length = len(bitstr)
        line = f"    {{ {{ {symbol_hex} }}, 0x{codeword_int:X}, {bit_length} }},"
        body_lines.append(line)

    footer = f"""\
}};

#endif // {var_name.upper()}_H

"""

    # Combine everything
    full_output = header + "\n".join(body_lines)[:-1] + "\n" + footer

    # Write to .h
    with open("coding_table.h", "w", encoding="utf-8") as f:
        f.write(full_output)


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
    key_lengths = (1,)

    payload_gains = []
    for key_length in key_lengths:
        symbols, frequencies = get_freq_table(payloads, key_length)

        # Testing
        # symbols, frequencies = ("A", "B", "C"), (.5, .3, .2)

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

        # Plot the results
        if args.plot:
            plt.figure(figsize=(6, 4))
            plt.hist(np.arange(len(frequencies)), bins=len(frequencies) //
                     2**2, weights=frequencies, label="Chunks Distribution")
            plt.title("Chunks Distribution")
            plt.xlabel("Chunk Content")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"chunks_distribution_{key_length}.svg")
            plt.show()

    # [3] Save the coding table
    save_huffman_table_to_h(coding_table, key_length)

    np.save("coding_table.npy", coding_table)


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
