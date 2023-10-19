"""
Utilities sub-module for MOKKa.

This sub-module contains a lot of useful utiltiies used throughout
the MOKKa package.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging

from . import decorators  # noqa
from . import generators  # noqa
from . import bitops  # noqa


# Plotting utilities


def setup_plot(title):
    """
    Create a figure and axes.

    :params title: Title of the figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(title=title)
    return fig, ax


def plot_scatter(ax, samples, labels):
    """
    Scatter plot a bunch of samples with labels.

    :params ax: matplotlib axis
    :params samples: complex samples to plot
    :params labels: list of labels assigning a label per sample
    """
    classes = np.unique(labels)
    for c in classes:
        c_idx = np.nonzero(labels == c)
        ax.scatter(samples[c_idx].real, samples[c_idx].imag)


def plot_classifier(ax, class_fn, range_x, range_y, points_x, points_y):
    """
    Plot classifier regions.

    :params ax: matplotlib axis
    """
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(range_x[0], range_x[1], points_x),
        torch.linspace(range_y[0], range_y[1], points_y),
    )
    class_input = torch.cat(
        (torch.reshape(grid_x, (-1, 1)), torch.reshape(grid_y, (-1, 1))), dim=1
    )
    class_output = class_fn(class_input)

    plot_value = torch.reshape(class_output, (points_x, -1)).detach().numpy()
    ax.contourf(grid_x.detach().numpy(), grid_y.detach().numpy(), plot_value, levels=10)


def plot_constellation(
    ax,
    constellation,
    label=True,
    keep=False,
    color=None,
    size=2,
    probabilities=None,
    label_type="hex",
):
    """Plot constellation points.

    :param ax: matplotlib axis
    :param constellation: Constellation points in the order of their bitlabels
    :param label: Switch to plot labels
    :param keep: Keep existing drawing on the axis
    :param color: Color for the constellation points
    :param size: Size of the constellation points
    :param probabilities: scale the size of the constellation according
                          to their probability.
    :param label_type: Hexadecimal or bitstring labels
    :returns: axis

    """
    const_cpu = constellation.detach().cpu().numpy()

    bitlabels = ints2bits(np.arange(len(const_cpu)), int(np.log2(len(const_cpu))))
    if label_type == "hex":
        labels = bits2hex(bitlabels)
    else:
        labels = bitlabels
    if not keep:
        ax.clear()
    ax.set_facecolor("white")
    if probabilities is not None:
        size = probabilities * 600
    if color is not None:
        ax.scatter(
            const_cpu.real,
            const_cpu.imag,
            alpha=0.8,
            edgecolor=None,
            s=size,
            color=color,
        )
    else:
        ax.scatter(const_cpu.real, const_cpu.imag, alpha=0.8, edgecolor=None, s=size)
    if label:
        for point, label in zip(const_cpu, labels):
            ax.annotate(
                label,
                (point.real, point.imag),
                textcoords="offset points",
                xytext=(1, 4),
                size=8,
                color="black",
            )


def plot_bitwise_decision_regions(
    axs,
    demapper,
    num_samples=10000,
    meshgrid=False,
    additional_args=None,
    sample_radius=1.2,
):
    """
    Plot the bitwise decision regions of the demapper.

    This is done by either sampling random points within a radius or using a meshgrid
    to sample points along the real and imaginary axis with a fixed spacing.

    :param axs: Matplotlib axis
    :param demapper: demapper object
    :param num_samples: number of samples in total or per axis is meshgrid is used
    :param meshgrid: Use a meshgrid to sample the 2D area
    :param additional_args: additional arguments to provide to the demapper
    :param sample_radius: either the radius or the maximum amplitude for meshgrid

    """
    if meshgrid:
        pp_axis = num_samples
        sample_real, sample_imag = torch.meshgrid(
            (
                torch.linspace(-sample_radius, sample_radius, pp_axis),
                torch.linspace(-sample_radius, sample_radius, pp_axis),
            )
        )
        sample_points = sample_real + 1j * sample_imag
        sample_points = sample_points.flatten().to(next(demapper.parameters()).device)

        if additional_args is not None:
            args_expanded = tuple(
                arg.expand(num_samples**2, 1) for arg in additional_args
            )

    else:
        sample_points = (
            sample_radius
            * torch.rand((num_samples,))
            * torch.exp(2j * np.pi * torch.rand((num_samples,)))
        ).to(next(demapper.parameters()).device)
        if additional_args is not None:
            args_expanded = tuple(arg.expand(num_samples, 1) for arg in additional_args)

    if additional_args is None:
        sample_demapped_bits = demapper(sample_points)
    else:
        sample_demapped_bits = demapper(sample_points, *args_expanded)
    for m in range(demapper.m.item()):
        ax = axs[m]
        axs[m].clear()
        _ = ax.tricontourf(
            sample_points.real.detach().cpu().numpy(),
            sample_points.imag.detach().cpu().numpy(),
            np.clip(sample_demapped_bits[:, m].detach().cpu().numpy(), -20, 19.9),
            levels=np.linspace(-20, 20, 100),
            cmap="RdBu",
        )
        # plt.colorbar(tcf, ax=ax)
    return sample_points, sample_demapped_bits


def setup_logging(level, component=""):
    """
    Configure the Python logging module to print logs for given component \
    and minimum level on stdout.

    :param level: A level from the logging module as minimum level
    :param component: The parent component for which all logs should be displayed.
    """
    logger = logging.getLogger(component)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def sigma_phi(linewidth, symbolrate):
    r"""Calculate :math:`\sigma_{\phi}` given a laser linewidth and symbol rate.

    :param linewidth: laser linewidth :math:`\Delta f`  [Hz]
    :param symbolrate: symbol rate :math:`R_{\mathrm{sym}}` [Bd]
    :returns: :math:`\sqrt{2 \pi \Delta f / R_{\mathrm{sym}}}`

    """
    return np.sqrt(2 * np.pi * linewidth / symbolrate)


def N0(SNR):
    r"""Calculate :math:`N_0` from SNR.

    :param SNR: signal-to-noise ratio :math:`s` [dB]
    :returns: :math:`10^{-\frac{s}{10}}`

    """
    return 10 ** (-SNR / 10)


def export_constellation(output_file, constellation):
    """Export a constellation to a csv file.

    :param output_file: path to file
    :param constellation: tensor of complex constellation points
    :returns:

    """
    labels = bits2hex(
        ints2bits(np.arange(len(constellation)), int(np.log2(len(constellation))))
    )
    result = ["real\timag\tlabel\n"]
    for point, label in zip(constellation, labels):
        result.append(f"{point.real}\t{point.imag}\t{label}\n")
    with open(output_file, "w") as result_file:
        result_file.truncate()
        result_file.writelines(result)


def beta2(D, wavelength):
    r"""
    Calculate :math:`\beta_2` from the dispersion coefficient and wavelength.

    :param D: Dispersion coefficient [ps/nm/km]
    :param wavelength: wavelength of optical carrier [nm]
    :returns: beta2 in [ps**2/km]
    """
    return -D * wavelength**2 / (2 * np.pi * 3e8) * 1e3


def wavelength(D, beta2):
    r"""
    Calculate the wavelength from :math:`\beta_2` and the dispersion coefficient.

    :param D: Dispersion coefficient [ps/nm/km]
    :param beta2: Group velocity dispersion [ps**2/km]
    :param: returns wavelength of optical carrier [nm]
    """
    return np.sqrt(-beta2 * 2 * np.pi * 3e8 / D / 1e3)


def estimate_SNR(x, y):
    """
    Calculate the (non-linear) SNR from the sent and received symbols.

    :params x: signal vector
    :params y: signal + noise vector
    """
    signal_power = torch.mean(torch.abs(x) ** 2)
    noise_power = torch.mean(torch.abs(x - y) ** 2)
    snr = signal_power / noise_power
    return 10 * torch.log10(snr)


def pow2db(y):
    """Calculate the power in decibel.

    :param y: power in linear units [W]
    :returns: power in decibel [dBW]

    """
    return 10 * torch.log10(y)


def pow2dbm(y):
    """Calculate the power in dBm.

    :param y: power in linear units [W]
    :returns: power in dBm [dBm]
    """
    return 10 * torch.log10(y) + 30


def db2pow(ydb):
    """Calculate the power in linear units.

    :param ydb: power in decibel [dB]
    :returns: power in linear units [W]
    """
    return 10 ** (ydb / 10)


def dbm2pow(ydbm):
    """Calculate the power in linear units.

    :param ydbm: power in dBm [dBm]
    :returns: power in linear units [W]
    """
    return 10 ** ((ydbm - 30) / 10)


def bits2int(bits):
    """Convert bit array to their integer representation MSB first.

    :param bits: array of bit arrays
    :returns: array of ints
    """
    return bits.dot(1 << np.arange(bits.shape[1])[::-1])


def ints2bits(ints, m):
    """Convert integers to their bit representations MSB first.

    :param ints: array of ints
    :param m: bit-width per int
    :returns: array of bit arrays
    """
    b = np.expand_dims(ints, 1)
    B = np.flip(
        np.unpackbits(b.view(np.uint8), axis=1, count=m, bitorder="little"), axis=1
    )[:, :m]
    return B


def bits2hex(bits):
    """Convert bit array to their hex representation MSB first.

    :param bits: array of bit arrays
    :returns: list of hex strings
    """
    vhex = np.vectorize(hex)
    ints = bits2int(bits)
    hexs = list(str(v)[2:].upper() for v in vhex(ints))
    return hexs


def hex2bits(hexs, m):
    """Convert list of hex strings to their bit representation MSB first.

    :param hexs: list of hex strings
    :param m: bit-width per hex
    :returns: array of bit arrays
    """
    ints = np.array([int("0x" + h, 0) for h in hexs])
    bits = ints2bits(ints, m)
    return bits
