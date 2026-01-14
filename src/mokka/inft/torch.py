"""Implementation of information theoretic functions in pyTorch for MOKKa."""

import torch
import numpy as np
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
        entropy = torch.sum(-p * torch.log2(p + 1e-12))
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


def GMI(m, N, b_ij, L_ij, p=None):
    """
    Calculate the generalized mutual information.

    Written by Benedikt Geiger

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
    GMI         : float
        generalized mutual information
    """
    # Store all conditionals
    conditional_probs = {}
    conditional_entropies = {}

    for i in range(m):
        target = b_ij[:, i]  # B_{i+1}

        if i == 0:
            # Unconditional: P(B1)
            counts = torch.bincount(target, minlength=2).float()
            probs = counts / counts.sum()
            conditional_probs[f"P(B1)"] = probs

            # H(B1)
            probs = conditional_probs["P(B1)"]
            entropy = -(probs * torch.log2(probs.clamp(min=1e-12))).sum()
            conditional_entropies["H(B1)"] = entropy.item()
        else:
            # Conditional: P(B_{i+1} | B1,...,B_i)
            context = b_ij[:, :i]  # shape N x i
            context_indices = (
                context * (2 ** torch.arange(i - 1, -1, -1, device=b_ij.device))
            ).sum(dim=1)

            num_contexts = 2**i
            joint_counts = torch.zeros(
                num_contexts, 2, device=b_ij.device
            )  # 2 for bit values 0,1

            for b_val in [0, 1]:
                mask = target == b_val
                selected_contexts = context_indices[mask]
                joint_counts[:, b_val] = torch.bincount(
                    selected_contexts, minlength=num_contexts
                )

            # Normalize each row to get P(B_{i+1} | context)
            probs = joint_counts / joint_counts.sum(dim=1, keepdim=True)
            conditional_probs[f"P(B{i+1}|B1..B{i})"] = probs

            key = f"P(B{i+1}|B1..B{i})"
            cond_probs = conditional_probs[key]  # shape: [2^i, 2]

            # Compute entropy per context
            entropy_per_context = -(
                cond_probs * torch.log2(cond_probs.clamp(min=1e-12))
            ).sum(dim=1)

            # Get context probabilities: count how often each context occurred
            context = b_ij[:, :i]
            context_indices = (
                context * (2 ** torch.arange(i - 1, -1, -1, device=b_ij.device))
            ).sum(dim=1)
            context_counts = torch.bincount(context_indices, minlength=2**i).float()
            context_probs = context_counts / context_counts.sum()

            # Total conditional entropy = weighted average
            weighted_entropy = (context_probs * entropy_per_context).sum()
            conditional_entropies[f"H(B{i+1}|B1..B{i})"] = weighted_entropy.item()

    # print("Conditional entropy")
    entropy = sum(
        (
            conditional_entropies[f"H(B{i+1}|B1..B{i})"]
            if i > 0
            else conditional_entropies["H(B1)"]
        )
        for i in range(m)
    )
    # print(entropy)
    # print("Bitwise entropy")
    P_b = torch.sum(b_ij, 0) / b_ij.shape[0]
    # print(P_b)
    # print(torch.sum(-P_b*torch.log2(P_b+1e-12)-(1-P_b)*torch.log2(1-P_b+1e-12)))
    # print("Entropy gap")
    # print(torch.sum(-P_b*torch.log2(P_b+1e-12)-(1-P_b)*torch.log2(1-P_b+1e-12))-entropy)

    GMI = entropy - 1 / N * torch.sum(
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
    return GMI


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


def MI_AWGN(
    received_symbols: torch.tensor,
    transmitted_sybmols: torch.tensor,
    constellation_points: torch.tensor,
    constellation_probabilites: torch.tensor,
    noise_power: torch.tensor,
) -> torch.tensor:
    """
    Calculate the Mutual infomation like a boss (LS) under the assumption of an AWGN channel

    :param received_symbols: t.tensor of received symbols
    :param transmitted_sybmols: t.tensor of received symbols
    :param constellation_points: position of the constellation points
    :param constellation_probabilites: probabilites of the constellation points
    :param noise_power: linear noise power (sum of both dimensions)
    :returns: estimate of the mutual inforamation (MI)

    Written by:
    - Benedikt Geiger, 11.12.2024
    """
    assert (
        received_symbols.ndim
        + transmitted_sybmols.ndim
        + constellation_points.ndim
        + constellation_points.ndim
    ) == 4, f"Expected all inputs to have dim 1"
    numerator = torch.exp(
        -torch.abs(received_symbols - transmitted_sybmols) ** 2 / (noise_power)
    )
    denominator = torch.sum(
        torch.exp(
            -torch.abs(
                received_symbols.unsqueeze(1) - constellation_points.unsqueeze(0)
            )
            ** 2
            / (noise_power)
        )
        * constellation_probabilites.unsqueeze(0),
        dim=1,
    )
    MI = torch.mean(torch.log2((numerator + 1e-12) / (denominator + 1e-12)))
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


def calculate_BER(predictions, labels):
    """
    Calculates the bit error rate

    Parameters
    ----------
    predictions    : Vector
        Log-likelihood ratios (LLRs)
    labels         : Vector
        Transmitted bits

    Returns
    -------
    BER            : float
        Bit Error Rate (BER)
    """
    BER = torch.sum((predictions > 0.5) != labels) / labels.numel()
    return BER


def calculate_SER(predictions, labels):
    """
    Calculates the symbol error rate

    Parameters
    ----------
    predictions    : Vector
        Probabilites that a certain symbol was transmitted
    labels         : Vector
        Transmitted symbols

    Returns
    -------
    SER            : float
        Symbols Error Rate (SER)
    """
    return torch.sum(torch.argmax(predictions, 1) != labels) / predictions.shape[0]
