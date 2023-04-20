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
