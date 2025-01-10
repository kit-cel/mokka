"""Functions that do not fit anywhere else - in PyTorch."""

import torch
import numpy as np
from .. import utils


def convolve(signal, kernel, mode="full"):
    """Calculate the 1-D convolution using the torch.conv1d function.

    This function is a convenience to assist using the torch convolution function
    within a signal processing context. The convolution function provided by
    PyTorch has image processing in mind and therefore some parameters of this function
    can be simplfied.

    :param signal: signal in the time domain.
    :param kernel: second signal in the time domain, must be shorter.
    :param mode: either "full", "same" or "valid".
    :returns: convolution between signal and kernel.

    """
    if mode == "full":
        padding = kernel.shape[0] - 1
    elif mode == "same":
        padding = "same"
    elif mode == "valid":
        padding = "valid"
    return torch.squeeze(
        torch.conv1d(
            signal[None, :],
            torch.flip(kernel[None, None, :], (2,)),
            padding=padding,
        )
    )


def convolve_overlap_save(signal, kernel, mode="full"):
    """Calculate the 1-D convolution using FFT overlap-save method.

    This is  only  efficient for very  long signals,
    where signal length >> 10*kernel length
    """
    original_signal_len = signal.shape[0]
    conv_padding = 0
    if mode == "full":
        conv_padding = kernel.shape[0] - 1
        output_len = original_signal_len + kernel.shape[0] - 1
    elif mode == "valid":
        conv_padding = 0
        output_len = original_signal_len - kernel.shape[0] + 1
    elif mode == "same":
        conv_padding = int((kernel.shape[0]) // 2)
        output_len = original_signal_len
    else:
        raise Exception(f"Convolution mode {mode} not supported!")
    signal = torch.nn.functional.pad(signal, (conv_padding, conv_padding))
    # Approximate  length of FFT:
    fft_len = int(2 ** np.ceil(np.log2(kernel.shape[0] * 10)))
    kernel_f = torch.fft.fft(kernel, fft_len, dim=0).unsqueeze(0)
    # Generate overlapping view on signal
    n_fft = np.ceil((signal.shape[0] - fft_len) / (fft_len - kernel.shape[0] + 1))
    pad_signal_len = int((n_fft + 1) * fft_len + n_fft * (1 - kernel.shape[0]))
    padding_value = pad_signal_len - signal.shape[0]
    split_signal = torch.nn.functional.pad(signal, (0, padding_value)).unfold(
        0, fft_len, fft_len - kernel.shape[0] + 1
    )
    signal_f = torch.fft.fft(split_signal, fft_len, dim=1)
    filtered_signal = torch.fft.ifft(kernel_f * signal_f, fft_len, dim=1)
    output_signal = filtered_signal[:, kernel.shape[0] - 1 : fft_len].reshape(-1)[
        :output_len
    ]
    return output_signal


def distribution_quantization(probs, n):
    """
    Compute a quantized distribution for given probabilities.

    This function uses an algorithm introduced for calculating
    a discrete distribution minimizing the KL divergence for CCDM.

    :param probs: probabilities of occurence for each element.
    :param n: length of the full sequence.
    :returns: number of ocurrences for each element in the sequence.

    """
    c = torch.floor(probs * n)
    k = c + 1
    klogk = k * torch.log(k)
    clogc = c * torch.log(c)
    clogc[torch.isnan(clogc)] = 0
    inc_cost = klogk - clogc - torch.log(probs)
    for i in torch.arange(n - torch.sum(c).item()):
        idx = torch.argmin(inc_cost)
        c[idx] += 1
        k = c[idx] + 1
        tmp = k * torch.log(k)
        inc_cost[idx] = inc_cost[idx] - klogk[idx] - klogk[idx] + tmp + clogc[idx]
        clogc[idx] = klogk[idx]
        klogk[idx] = tmp
    return c


def distribution_quant_gumbel_softmax(probs, n, temp=0.1):
    """
    Compute a quantized distribution for given probabilities with Gumbel-Softmax.

    This uses the Gumbel-Softmax sampling method to provide a differentiable
    way to embed the information about the distribution into the sequence.
    We use the straight-through trick to retain the gradients of softmax but
    actually use the values from a argmax for calculation.

    :param probs: probabilities of occurence for each element
    :param n: length of the sequence to sample
    :param temp: temperature of Softmax.
    :returns: sampled indices in as sequence of length n
    """
    probs = torch.atleast_2d(probs)

    oh_soft = torch.nn.functional.gumbel_softmax(
        torch.tile(torch.log(probs), dims=(n, 1)), tau=temp, dim=1, hard=False
    )
    oh_hard = torch.nn.functional.one_hot(
        torch.argmax(oh_soft, dim=1), oh_soft.shape[1]
    )
    oh = oh_hard - oh_soft.detach() + oh_soft
    idx = (
        torch.arange(0, probs.shape[1], dtype=torch.float, device=oh.device)[None, :]
        @ oh.T
    )
    return idx


def unwrap(p, discont=np.pi, dim=-1, period=2 * np.pi):
    """
    Perform phase unwrapping in PyTorch.

    This implementation follows the NumPy implementation closely.

    :param p: tensor with angle values
    :param discont: discontinuity to unwrap
    :param dim: dimension to unwrap
    :param period: which periodicity the signal has
    """
    # NP unwrap implemented in Torch
    nd = p.ndim
    dd = torch.diff(p, dim=dim)
    slice1 = [slice(None, None)] * nd  # full slices
    slice1[dim] = slice(1, None)
    slice1 = tuple(slice1)
    ddmod = torch.remainder(dd + np.pi, 2 * np.pi) - np.pi
    ph_correct = ddmod - dd
    up = p.clone()
    up[slice1] = p[slice1] + torch.cumsum(ph_correct, dim=dim)
    return up


@utils.decorators.deprecated("Use unwrap() instead")
def unwrap_torch(*args):
    """
    Perform phase unwrapping in PyTorch.

    This implementation follows the NumPy implementation closely.

    :param p: tensor with angle values
    :param discont: discontinuity to unwrap
    :param dim: dimension to unwrap
    :param period: which periodicity the signal has
    """
    return unwrap(*args)


def Jones2Mueller(J):
    """Take a 2x2 complex Jones matrix and return a 4x4 real Mueller matrix."""
    # point-wise multiply
    U = (
        1
        / torch.sqrt(torch.as_tensor(2.0))
        * torch.tensor(
            [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -1j, 1j, 0]],
            dtype=torch.complex64,
        )
    )
    U_inv = (
        1
        / torch.sqrt(torch.as_tensor(2.0))
        * torch.tensor(
            [[1, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j], [1, -1, 0, 0]],
            dtype=torch.complex64,
        )
    )
    J_expanded = torch.kron(J, J.conj().resolve_conj())
    M = torch.matmul(torch.matmul(U, J_expanded), U_inv).real
    return M
