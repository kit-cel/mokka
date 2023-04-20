"""NumPy implementation of Mappers and Demappers."""

import numpy as np
from ..utils.bitops import gray
from scipy import interpolate as interp
from ..utils.generators.numpy import generate_all_bits
import logging

logger = logging.getLogger(__name__)


class QAM:
    """
    Arbitrarily sized QAM constellation with gray mapping.

    :param m: Information per symbol [bit/symbol]
    """

    def __init__(self, m):
        """Construct QAM class."""
        if m % 2:
            raise ValueError(
                "Only bits/per Symbol which are multiple of 2 are supported"
            )
        self.m = m
        self.bit_sequence = np.array(gray(m // 2))

        self.one_idx = np.nonzero(self.bit_sequence)
        self.zero_idx = np.nonzero(np.abs(self.bit_sequence - 1))

        scaling = np.sqrt(3 / 2) * (np.sqrt(2**m) - 1) / (np.sqrt(2**m - 1))
        self.symbol_amplitudes = np.linspace(-1, 1, 2 ** (m // 2)) * scaling
        self.symbol_amplitudes_rep = self.symbol_amplitudes[:, np.newaxis].repeat(
            m // 2, axis=1
        )
        self.demap_interp = interp.interp1d(
            self.symbol_amplitudes,
            np.arange(2 ** (m // 2)),
            bounds_error=None,
            fill_value="extrapolate",
        )

    def get_constellation(self):
        """
        Return all constellation symbols.

        The index represents the integer representation of
        the encoded bit-string (LSB first).

        :returns: all constellation symbols
        """
        bits = generate_all_bits(self.m)
        symbols = self.map(bits)
        return symbols

    def map(self, bits):
        """
        Perform mapping of bitstrings.

        :returns: complex symbol per bitstring of length `self.m`
        """
        output = []
        for seq in bits.reshape(-1, self.m):
            # Take first half for real and second half for imag
            real = self.symbol_amplitudes[
                (self.bit_sequence == seq[: self.m // 2]).all(axis=1).nonzero()
            ]
            imag = self.symbol_amplitudes[
                (self.bit_sequence == seq[-(self.m // 2) :]).all(axis=1).nonzero()
            ]
            output.append(real + 1j * imag)
        return np.array(output)
