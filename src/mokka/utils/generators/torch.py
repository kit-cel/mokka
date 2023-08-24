"""Module for data generators in PyTorch."""

import torch  # noqa


def generate_bits(shape):
    """Generate uniform random bits.

    :param shape: tuple with resulting shape
    :returns: array with uniform random bits in the requested shape

    """
    bits = torch.randint(0, 2, shape)
    return bits
