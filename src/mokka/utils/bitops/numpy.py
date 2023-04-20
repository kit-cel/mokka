"""Module to implement bit operations in NumPy."""

import numpy as np


def idx2bits(idxs, num_bits_per_symbol):
    """
    Convert integer representations of bit vectors to bits in LSB-first.

    :param idxs: Vector of indices
    :param num_bits_per_symbol: size of the bit vectors
    :returns: vector of bit vectors
    """
    bits = np.remainder(
        np.right_shift(
            idxs[:, np.newaxis],
            np.arange(num_bits_per_symbol),
        ),
        2,
    )
    return bits


def bits2idx(bits):
    """
    Convert bits in LSB-first to their integer representation.

    :param bits: Vectors of bits with the last dimension representing the size of words
    :returns: Vector of integers
    """
    coeff = 2 ** np.arange(bits.shape[-1])
    B = np.sum(bits * np.flip(coeff, dims=(0,)), -1)
    return B


def generate_all_input_bits(m):
    """Generate all :math:`2^m` possible bitstrings.

    :param m: Length of the bitstring
    :returns: Array with all possible bitstrings in LSB

    """
    nums = np.arange(2**m, dtype=np.uint8)
    bits = idx2bits(nums, m)
    return bits


def gray(m):
    """Generate a sequence of bit strings with binary-reflected Gray coding.

    :param m: length of the bitstrings
    :returns: Tuple of :math:`2^m` Gray-encoded bitstrings

    """
    if m == 1:
        return ((0,), (1,))
    prev = gray(m - 1)
    return tuple((0,) + p for p in prev) + tuple((1,) + p for p in reversed(prev))
