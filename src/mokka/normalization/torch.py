"""Normalization for communication systems in PyTorch."""

import torch


def energy(c, p=None):
    """
    Perform normalization on average energy.

    :param c: complex constellation points
    :param p: probabilities, if None: uniform probability is assumed
    :returns: constellation with average energy 1
    """
    if p is not None:
        scaling = torch.sqrt(1.0 / torch.sum(p * (torch.abs(c) ** 2), -1))
    else:
        scaling = torch.sqrt((c.size()[-1]) / torch.sum(torch.abs(c) ** 2, -1))
    scaling = torch.unsqueeze(scaling, -1).repeat(*((1,) * len(c.size())), c.size()[-1])
    scaling = torch.squeeze(scaling, 0)
    c = torch.multiply(c, scaling)
    return c


def centered_energy(c, p=None):
    """Perform centered (zero-mean) normalization of the complex constellation.

    :param c: complex constellation points
    :param p: probabilities, if None: uniform probability is assumed
    :returns: centered constellation with average energy of 1

    """
    if p is not None:
        c = (c - (torch.sum(p * c, -1))) / (
            torch.sqrt(
                (
                    (torch.sum(p * torch.abs(c) ** 2, -1))
                    - torch.abs((torch.sum(p * c, -1))) ** 2
                )
            )
        )
    else:
        c = (c - (torch.sum(c, -1) / (c.size()[-1]))) / (
            torch.sqrt(
                (
                    (torch.sum(torch.abs(c) ** 2, -1) / c.size()[-1])
                    - torch.abs((torch.sum(c, -1) / c.size()[-1])) ** 2
                )
            )
        )
    return c


def center_constellation(constellation_points, constellation_probabilities=None):
    """Performs centering (zero-mean) of the complex constellation

    :param constellation_points: complex constellation points
    :param constellation_probabilites: probabilities, if None: uniform probability is assumed
    :returns: centered complex constellation

    Updated by:
    - Benedikt Geiger, 21.11.2024
    """
    if constellation_probabilities is not None:
        mass_center = torch.sum(constellation_probabilities * constellation_points, -1)
    else:
        mass_center = torch.mean(constellation_points, -1)
    if len(constellation_points.shape) > 1:
        mass_center = mass_center.unsqueeze(1)
    centered_constellation_points = constellation_points - mass_center
    return centered_constellation_points


def normalize_constellation(constellation_points, constellation_probabilities=None):
    """Perform normalization (unit power) of the complex constellation.

    :param constellation_points: complex constellation points
    :param constellation_probabilites: probabilities, if None: uniform probability is assumed
    :returns: unit power complex constellation

    Updated by:
    - Benedikt Geiger, 21.11.2024
    """
    if constellation_probabilities is not None:
        power = torch.sum(
            constellation_probabilities * torch.abs(constellation_points) ** 2, -1
        )
    else:
        power = torch.mean(torch.abs(constellation_points) ** 2, -1)
    if len(constellation_points.shape) > 1:
        power = power.unsqueeze(1)
    normalized_constellation = constellation_points / torch.sqrt(power)
    return normalized_constellation
