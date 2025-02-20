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
    

class PSK:
    """
    Arbitrarily sized PSK constellation with gray mapping.

    :param m: Information per symbol [bit/symbol]
    """

    def __init__(self, m):
        """Construct PSK class."""
        if m % 2:
            raise ValueError(
                "Only bits/per Symbol which are multiple of 2 are supported"
            )
        self.m = m
        self.bit_sequence = np.array(gray(m))

        self.one_idx = np.nonzero(self.bit_sequence)
        self.zero_idx = np.nonzero(np.abs(self.bit_sequence - 1))

        phase_step_size = 2*np.pi/(2**m)
        self.symbol_phases = np.arange(2**m)*phase_step_size
        self.symbols = np.exp(1j*self.symbol_phases)

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
            output.append(self.symbols[(self.bit_sequence == seq).all(axis=1).nonzero()])
        return np.array(output)        


class r_phi_PSK:
    """
    Arbitrarily sized r_phi_PSK constelllations.

    :param num_bits_radial: Number of bits encoded in radial direction
    :param num_bits_phase: Number of bits in phase/angular direction
    """

    def __init__(self, num_bits_radial, num_bits_phase):
        """Construct r_phi_PSK class."""
        if num_bits_radial % 1:
            raise ValueError(
                "Number of bits must be integer"
            )
        if num_bits_phase % 1:
            raise ValueError(
                "Number of bits must be integer"
            )
        self.num_bits_radial = num_bits_radial
        self.num_radii = int(2**num_bits_radial)    
        self.num_bits_phase = num_bits_phase
        self.num_phases = int(2**num_bits_phase)
        
        self.bit_sequence_radius = np.array(gray(num_bits_radial))
        self.bit_sequence_angle = np.array(gray(num_bits_phase))

        self.m = num_bits_radial + num_bits_phase

        scaling = np.sqrt(6/((self.num_radii+1)*(2*self.num_radii+1)))
        self.symbol_radii = (np.arange(self.num_radii)+1)*scaling
        self.symbol_phases = np.arange(self.num_phases)/self.num_phases*2*np.pi

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
            # Take first num_bits_radial bits for amplitude and last num_bits_phase bits for phase
            amplitude = self.symbol_radii[
                (self.bit_sequence_radius == seq[: self.num_bits_radial]).all(axis=1).nonzero() if self.num_bits_radial>0 else 0
            ]
            phase = self.symbol_phases[
                (self.bit_sequence_angle == seq[-self.num_bits_phase :]).all(axis=1).nonzero() if self.num_bits_phase>0 else 0
            ]
            output.append(amplitude*np.exp(1j*phase))
        return np.array(output)