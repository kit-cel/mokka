"""Implementation of information theoretic functions in pyTorch for MOKKa."""

import torch
import logging

logger = logging.getLogger(__name__)


def BMI(m, N, b_ij, L_ij, p=None):
    """
    Calculate the bitwise mutual information.

    Originally written by Benedikt Geiger

    Parameters
    ----------
    m           : int
        number of bits
    N           : int
        number of sent/received symbols
    b_ij        : 2D-matrix_like
        value of the i-th (Sent) modulation symbol
        at the j-th position in the bit-string
    L_ij        : 2D-matrix_like
        LLRs
    p           : 1D-matrix_like
        probability of occurence of the sent symbols
        to adjust source entropy.

    Returns
    -------
    BMI         : float
        bitwise mutual information
    """
    if p is not None:
        entropy = torch.sum(-p * torch.log2(p))
    else:
        entropy = m
    BMI = entropy - 1 / N * torch.sum(
        torch.log2(
            1
            + torch.exp(
                torch.clip(
                    -1 * torch.pow(-1, b_ij) * L_ij,
                    max=torch.tensor(88.0).to(b_ij.device),
                )
            )
        )
    )  # Positive L_ij corresponds to 0 bit
    return BMI


def MI(M, PX, N, symbol_idx, Q_YX):
    """
    Calculate the mutual information.

    Numerically Computing Achievable Rates of Memoryless Channels

    Parameters
    ----------
    M           : int
        number of symbols
    PX        : 2D-matrix_like
        1 x M
    N           : int
        number of sent/received symbols
    symbol_idx  : 1D-matrix_like
        sent symbol indices
    Q_YX        : 2D-matrix_like
        N x M conditonal probabilities

    Returns
    -------
    MI         : float
        mutual information
    """
    MI = torch.mean(
        torch.log2(
            Q_YX[torch.arange(0, N, device=Q_YX.device), symbol_idx.long()]
            / torch.sum(Q_YX * PX, 1)
        ),
        0,
    )
    return MI


def hMI(m, N, symbol_idx, demapped_symbol_idx):
    """
    Calculate the mutual information according to \
    "Achievable Rates for Probabilistic Shaping" (Eq. 87).

    If probabilistic shaping is applied this has to be reflected
    in the transmit symbols indices

    Parameters
    ----------
    m           : int
        number of bits per symbol
    N           : int
        number of sent/received symbols
    symbol_idx  : 1D-matrix_like
        sent symbol indices
    demapped_symbol_idx : 1D-matrix_like
        demapped symbol indices

    Returns
    -------
    BMI         : float
        bitwise mutual information

    """
    epsilon = 1 - torch.count_nonzero(demapped_symbol_idx == symbol_idx) / N
    epsilon = torch.clamp(epsilon, 1e-8, 1 - 1e-8)
    logger.debug("epsilon: %s", epsilon.item())
    hard_mi = m - (
        -(1 - epsilon) * torch.log2(1 - epsilon)
        - epsilon * torch.log2(epsilon)
        + epsilon * torch.log2(torch.tensor(2**m - 1))
    )

    return hard_mi


def SNR(M, sym_idx, rx_syms):
    """
    Calculate the signal to noise ratio.

    Parameters
    ----------
    M           : int
        number of constellation points
    sym_idx        : Vector
        Transmitted symbol indices between 0 and M-1
    rx_syms        : Vector
        Complex received symbols

    Returns
    -------
    SNR        : float
        Signal to noise ratio in linear units
    SNR_dB        :float
        Signal to noise ratio in decibels

    """
    assert sym_idx.shape == rx_syms.shape, "Dimension mismatch"
    rx_vals_mean = torch.zeros(M, dtype=torch.complex64, device=rx_syms.device)
    rx_vals_sigma2 = torch.zeros(M, dtype=torch.float64, device=rx_syms.device)
    for i in torch.arange(M):
        idx = (sym_idx == i).nonzero()
        rx_vals_mean[i] = torch.mean(rx_syms[idx])
        rx_vals_sigma2[i] = torch.mean(
            torch.square(torch.abs(rx_syms[idx] - rx_vals_mean[i]))
        )
    SNR = torch.mean(torch.square(torch.abs(rx_vals_mean))) / torch.mean(rx_vals_sigma2)
    SNR_dB = 10 * torch.log10(SNR)
    return SNR, SNR_dB
