"""Channels sub-module implemented within the PyTorch framework."""
import torch
import torchaudio
import math
import scipy.special
import numpy as np
import attr
import logging
from .. import utils
from .. import functional
from ..pulseshaping.torch import upsample, downsample, brickwall_filter

logger = logging.getLogger(__name__)


class ComplexAWGN(torch.nn.Module):
    """
    Complex AWGN channel defined with zero mean and variance N_0.

    This converts to a setting of stddev sqrt(N_0/2) for real and imaginary parts of the
    distribution.
    """

    def __init__(self, N_0=None):
        """Construct ComplexAWGN."""
        super(ComplexAWGN, self).__init__()
        self.N_0 = N_0

    def forward(self, x, N_0=None):
        """Apply Gaussian noise to a signal.

        :param x: input signal to apply noise to.
        :param N_0: noise variance
        :returns: input signal distorted with AWGN

        """
        if N_0 is None:
            if self.N_0 is None:
                raise Exception("N_0 needs to be defined somewhere")
            N_0 = self.N_0
        n = torch.zeros_like(x, device=x.device).normal_()
        # If only one value is supplied for all given samples
        if n.size() != N_0.size():
            N_0 = torch.full_like(n, N_0.item(), dtype=torch.float32)
        n = torch.mul(n, torch.sqrt(N_0))
        return x + n


class PhasenoiseWiener(torch.nn.Module):
    """
    Wiener phase noise channel.

    Forward call takes N0 for AWGN and sigma_phi for Wiener phase noise.

    :param start_phase_width: upper bound for initialization
                              with a uniform distribution.
    :param start_phase_init: initial phase value at the start of the
                             generation for each noise sequence.
    """

    def __init__(self, start_phase_width=2 * np.pi, start_phase_init=0):
        """Construct PhasenoiseWiener."""
        super(PhasenoiseWiener, self).__init__()
        self.awgn = ComplexAWGN()
        self.start_phase_init = start_phase_init
        self.start_phase_width = start_phase_width

    def forward(self, x, N0=None, sigma_phi=None):
        r"""Apply Wiener phase noise to a complex signal.

        :param x: ipnut signal to apply noise to
        :param N0: noise variance for Gaussian noise
        :param sigma_phi: standard deviation of Wiener phase noise process
        :returns: noise impaired signal
        """
        if N0 is not None:
            x = self.awgn(x, N0)
        x = self.apply(x, sigma_phi)
        logger.debug("x size: %s", x.size())
        return x

    def apply(self, x, sigma_phi=None):
        """
        Apply only Wiener phase noise.

        Parameters
        ----------
        x: array_like
           Input data
        """
        x = torch.squeeze(x)
        logger.debug("sigma_phi: %s", sigma_phi.item())
        sigma_phi_temp = sigma_phi

        # "Physical" Channel
        delta_phi = sigma_phi_temp * torch.randn(
            len(x), device=x.device, dtype=torch.float32
        )
        phi = torch.cumsum(delta_phi, 0) + (
            self.start_phase_init
            + torch.rand(1, device=x.device, dtype=torch.float32)
            * self.start_phase_width
        )
        x = x * torch.exp(1j * phi)
        logger.debug("x size: %s", x.size())
        return x

    def sample_noise(self, x, sigma_phi=None):
        """
        Sample one realization of phase noise.

        Parameters
        ----------
        x: array_like
           Input data
        """
        x = torch.squeeze(x)
        logger.debug("sigma_phi: %s", sigma_phi.item())
        sigma_phi_temp = sigma_phi

        # "Physical" Channel
        delta_phi = sigma_phi_temp * torch.randn(
            len(x), device=x.device, dtype=torch.float32
        )
        phi = torch.cumsum(delta_phi, 0) + (
            self.start_phase_init
            + torch.rand(1, device=x.device, dtype=torch.float32)
            * self.start_phase_width
        )
        logger.debug("x size: %s", x.size())
        return phi


class ResidualPhaseNoise(torch.nn.Module):
    """
    Residual phase noise implementation.

    This implements residual phase noise modeled with a zero mean phase noise process.
    """

    def __init__(self):
        """Construct ResidualPhaseNoise."""
        super(ResidualPhaseNoise, self).__init__()

    def forward(self, x, RPN):
        """
        Apply residual phase noise to signal.

        :param x: complex input signal
        :param RPN: phase noise variance per input symbol
        :returns: x with phase noise
        """
        n_phi = torch.zeros(x.size(), dtype=torch.float32, device=x.device).normal_()
        # If only one value is supplied for all given samples
        if n_phi.size() != RPN.size():
            RPN = torch.full_like(n_phi, RPN, dtype=float)
        n_phi = torch.mul(n_phi, torch.sqrt(RPN))
        return x * torch.exp(1j * n_phi)


@attr.define
class OpticalNoise(torch.nn.Module):
    """
    Zero mean additive Gaussian noise based on given OSNR defined over \
    certain wavelength.

    :param dt: Time step
    :param optical_carrier_wavelength: Wavelength of optical carrier in nm
    :param wavelength_bandwidth: Bandwidth in nm for which OSNR is given
    :param OSNR: Optical SNR in dB
    """

    dt: float
    optical_carrier_wavelength: float
    wavelength_bandwidth: float
    OSNR: float

    def __attrs_pre_init__(self):
        """Actual init function for attr classes."""
        super(OpticalNoise, self).__init__()

    def __attrs_post_init__(self):
        """Post-init function for attr classes."""
        # Conversion equation from https://www.rp-photonics.com/bandwidth.html
        self.frequency_bandwidth = (
            scipy.constants.c
            * self.wavelength_bandwidth
            / (self.optical_carrier_wavelength**2 * 1e-9)
        )

    def forward(self, signal):
        """Apply optical noise realization on signal.

        :param signal: input signal
        :returns: noise impaired signal
        """
        n0 = torch.randn(len(signal)).to(signal.device) + 1j * torch.randn(
            len(signal)
        ).to(signal.device)
        fvec = torch.fft.fftfreq(len(signal), self.dt)

        stft = torchaudio.transforms.Spectrogram(
            n_fft=len(signal), win_length=len(signal) // 8, power=2, onesided=False
        ).to(signal.device)
        stft_n0 = stft(n0)
        stft_signal = stft(signal)
        scl = (
            torch.sum(stft_signal[torch.abs(fvec) <= self.frequency_bandwidth / 2])
            / utils.db2pow(self.OSNR)
        ) / torch.sum(stft_n0[torch.abs(fvec) <= self.frequency_bandwidth / 2])

        n = n0 * torch.sqrt(scl)
        return signal + n


@attr.define
class EDFAAmpSinglePol(torch.nn.Module):
    """
    Gaussian noise model of an EDFA for single polarization fiberoptics.

    :param span_length: span_length [km]
    :param amp_noise: Consider amplification noise [bool]
    :param amp_gain: Which amplification gain model to use either 'alpha_equalization',
                     'equal_power'
    :param optical_carrier_frequency: Optical carrier frequency (for PSD) [GHz]
    :param bw: Occupied bandwidth (for PSD) [Hz]
    :param P_input_lin: Input power [W]
    :param padding: Number of samples to ignore in the beginning and end
    """

    span_length: int
    amp_gain: str
    alpha_db: torch.tensor
    amp_noise: bool
    noise_figure: torch.tensor
    optical_carrier_frequency: float
    bw: float
    P_input_lin: float
    padding: int
    alpha_lin: torch.tensor = attr.field(init=False)

    def __attrs_pre_init__(self):
        """Pre-init for attrs classes."""
        super(EDFAAmpSinglePol, self).__init__()

    def __attrs_post_init__(self):
        """Post-init for attrs classes0."""
        self.alpha_lin = self.alpha_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km

    def forward(self, signal, segment):
        r"""
        Amplify the signal according to the selected method after each span.

        Attention: the noise depends on the amplification. if :math:`\alpha = 0`,
        no noise is added.

        Parameters
        ----------
        signal : singal before the amplifier
        segment: segment number
        Returns
        -------
        signal : signal after the amplifier
        """
        if self.alpha_lin != 0 and segment == 2:
            if self.amp_gain == "alpha_equalization":
                gain_signal = torch.sqrt(torch.exp(self.alpha_lin * self.span_length))
            elif self.amp_gain == "equal_power":
                if self.padding > 0:
                    P_in = torch.mean(
                        torch.abs(signal[self.padding : -self.padding]) ** 2
                    )
                else:
                    P_in = torch.mean(torch.abs(signal) ** 2)
                gain_signal = torch.sqrt(self.P_input_lin / P_in)
            # calculate power spectrum density of the noise and add it to the
            # signal according to On Signal Constellations and Coding for
            # Long-Haul Fiber-Optical Systems (Christian Häger)
            signal *= gain_signal
            if self.amp_noise:
                gain_power = gain_signal**2
                F_n = 10 ** (self.noise_figure / 10)
                n_sp = F_n / 2 * ((1 / ((1 - 1 / gain_power))))
                psd = (
                    (gain_power - 1)
                    * scipy.constants.h
                    * self.optical_carrier_frequency
                    * 1e9  # Convert back to Hz
                    * n_sp
                )
                sigma = torch.sqrt(psd * self.bw / 2)
                b = sigma * (
                    torch.randn(len(signal)).to(signal.device)
                    + 1j * torch.randn(len(signal)).to(signal.device)
                )
                signal += b
        return signal

    def __call__(self, *args, **kwargs):
        """Execute forward() on calling of this class."""
        return self.forward(*args, **kwargs)


@attr.define
class EDFAAmpDualPol(torch.nn.Module):
    """
    Implement Gaussian noise EDFA model for dual polarization fiberoptical simulation.

    :param span_length: span_length
    :param amp_gain: Which amplification gain model to use either 'alpha_equalization',
                     'equal_power'
    :param alphaa_db: power loss coeficient [dB/km] for eigenstate a
    :param alphab_db: power loss coeficient [dB/km] for eigenstate b
    :param amp_noise: Consider amplification noise
    :param noise_figure: Amplifier noise figure in dB
    :param optical_carrier_frequency: Optical carrier frequency (for PSD) [GHz]
    :param bw: Bandwidth of baseband signal (for PSD)
    :param P_input_lin: Input power
    :param padding: Number of samples to ignore in the beginning and end
    """

    span_length: int
    amp_gain: str
    alphaa_db: torch.tensor
    alphab_db: torch.tensor
    amp_noise: bool
    noise_figure: torch.tensor
    optical_carrier_frequency: float
    bw: float
    P_input_lin: float
    padding: int
    alphaa_lin: torch.tensor = attr.field(init=False)
    alphab_lin: torch.tensor = attr.field(init=False)

    def __attrs_pre_init__(self):
        """Pre-init for attrs classes."""
        super(EDFAAmpDualPol, self).__init__()

    def __attrs_post_init__(self):
        """Post-init for attrs classes."""
        self.alphaa_lin = self.alphaa_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km
        self.alphab_lin = self.alphab_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km
        self.amp_type = "EDFA"

    def forward(self, u1, u2, segment):
        r"""
        Amplify the signal according to the selected method after each span.

        attention: the noise depends on the amplification. if :math:`\alpha = 0`,
        no noise is added.

        Parameters
        ----------
        signal : singal before the amplifier
        segment: segment number
        Returns
        -------
        signal : signal after the amplifier
        """
        if self.alphaa_lin != 0 and self.alphab_lin != 0 and segment == 2:
            if self.amp_gain == "alpha_equalization":
                gain_signal_1 = torch.exp(0.5 * self.alphaa_lin * self.span_length)
                gain_signal_2 = torch.exp(0.5 * self.alphab_lin * self.span_length)

            elif self.amp_gain == "equal_power":
                if self.padding > 0:
                    P_in = torch.mean(
                        torch.abs(u1[self.padding : -self.padding]) ** 2
                    ) + torch.mean(torch.abs(u2[self.padding : -self.padding]) ** 2)
                else:
                    P_in = torch.mean(torch.abs(u1) ** 2) + torch.mean(
                        torch.abs(u2) ** 2
                    )
                gain_signal_1 = torch.sqrt(self.P_input_lin / P_in)
                gain_signal_2 = gain_signal_1
            # calculate power spectrum density of the noise and add it to the
            # signal according to On Signal Constellations and Coding for
            # Long-Haul Fiber-Optical Systems (Christian Häger)
            u1 *= gain_signal_1
            u2 *= gain_signal_2
            logger.debug(
                "Pre noise addition: power u1 %s dBm",
                10 * torch.log10(torch.mean(torch.abs(u1) ** 2)).item() + 30,
            )
            if self.amp_noise:
                gain_power = gain_signal_1**2
                F_n = 10 ** (self.noise_figure / 10)  # amplifier noise figure
                hf = scipy.constants.h * self.optical_carrier_frequency * 1e9
                nsp = F_n / 2 * gain_power / (gain_power - 1)

                p_rho_ase = nsp * (gain_power - 1) * hf
                P_n_ase = 2 * p_rho_ase * self.bw
                logger.debug("bw: %s", self.bw)
                logger.debug("nsp: %s", nsp)
                logger.debug("P_n: %s", P_n_ase)
                logger.debug(
                    "hf: %s", scipy.constants.h * self.optical_carrier_frequency * 1e9
                )
                sigma = torch.sqrt(P_n_ase / 2)

                n1 = sigma * (
                    torch.randn(len(u1)).to(u1.device)
                    + 1j * torch.randn(len(u1)).to(u1.device)
                )
                n2 = sigma * (
                    torch.randn(len(u2)).to(u2.device)
                    + 1j * torch.randn(len(u2)).to(u2.device)
                )
                u1 += n1
                u2 += n2
                # print("Post noise addition: power u1",torch.mean(torch.abs(u1) ** 2))
                logger.debug(
                    "Noise power u1 %s dBm",
                    10 * torch.log10(torch.mean(torch.abs(n1) ** 2)).item() + 30,
                )
                logger.debug(
                    "Post noise addition: power u1 %s dBm",
                    10 * torch.log10(torch.mean(torch.abs(u1) ** 2)).item() + 30,
                )
        return u1, u2

    def __call__(self, *args, **kwargs):
        """Call forward() on calling of this class."""
        return self.forward(*args, **kwargs)


@attr.define
class RamanAmpDualPol(torch.nn.Module):
    """
    Implement a Gaussian model of the Raman dual polarization amp.

    :param amp_gain: Which amplification gain model to use either 'alpha_equalization',
                     'equal_power'
    :param alphaa_db: power loss coeficient [dB/km] for eigenstate a
    :param alphab_db: power loss coeficient [dB/km] for eigenstate b
    :param noise_figure: Amplifier noise figure in dB
    :param optical_carrier_frequency: Optical carrier frequency (for PSD) [GHz]
    :param bw: Bandwidth of baseband signal (for PSD)
    :param P_input_lin: Input power
    :param padding: Number of samples to ignore in the beginning and end
    """

    amp_gain: str
    alphaa_db: torch.tensor
    alphab_db: torch.tensor
    amp_noise: bool
    noise_figure: torch.tensor
    optical_carrier_frequency: float
    bw: float
    P_input_lin: float
    padding: int
    alphaa_lin: torch.tensor = attr.field(init=False)
    alphab_lin: torch.tensor = attr.field(init=False)

    def __attrs_pre_init__(self):
        """Pre-init of attrs classes."""
        super(RamanAmpDualPol, self).__init__()

    def __attrs_post_init__(self):
        """Post-init of attrs classes."""
        self.alphaa_lin = self.alphaa_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km
        self.alphab_lin = self.alphab_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km
        self.amp_type = "Raman"

    def forward(self, u1, u2, dz):
        r"""
        Amplify the signal according to the selected method after each span.

        Attention: the noise depends on the amplification. if :math:`\alpha = 0`,
        no noise is added
        Parameters
        ----------
        signal : signal before the amplifier
        dz : step-size
        Returns
        -------
        signal : signal after the amplifier
        """
        if self.alphaa_lin != 0 and self.alphab_lin != 0:
            if self.amp_gain == "alpha_equalization":
                gain_signal_1 = torch.exp(0.5 * self.alphaa_lin * dz)
                gain_signal_2 = torch.exp(0.5 * self.alphab_lin * dz)

            elif self.amp_gain == "equal_power":
                if self.padding > 0:
                    P_in = torch.mean(
                        torch.abs(u1[self.padding : -self.padding]) ** 2
                    ) + torch.mean(torch.abs(u2[self.padding : -self.padding]) ** 2)
                else:
                    P_in = torch.mean(torch.abs(u1) ** 2) + torch.mean(
                        torch.abs(u2) ** 2
                    )
                gain_signal_1 = torch.sqrt(self.P_input_lin / P_in)
                gain_signal_2 = gain_signal_1
            # calculate power spectrum density of the noise and add it to the
            # signal according to On Signal Constellations and Coding for
            # Long-Haul Fiber-Optical Systems (Christian Häger)
            u1 *= gain_signal_1
            u2 *= gain_signal_2
            logger.debug(
                "Pre noise addition: power u1 %s dBm",
                10 * torch.log10(torch.mean(torch.abs(u1) ** 2)).item() + 30,
            )
            if self.amp_noise:
                gain_power = gain_signal_1**2
                F_n = 10 ** (self.noise_figure / 10)  # amplifier noise figure
                n_sp = F_n / 2 * ((1 / ((1 - 1 / gain_power))))
                psd = (
                    (gain_power - 1)
                    * scipy.constants.h
                    * self.optical_carrier_frequency
                    * 1e9  # Convert back to Hz
                    * n_sp
                )
                # print("bw", self.bw)
                # print("n_sp", n_sp)
                # print("psd", psd)
                sigma = torch.sqrt(psd * self.bw / 2)

                n1 = sigma * (
                    torch.randn(len(u1)).to(u1.device)
                    + 1j * torch.randn(len(u1)).to(u1.device)
                )
                n2 = sigma * (
                    torch.randn(len(u2)).to(u2.device)
                    + 1j * torch.randn(len(u2)).to(u2.device)
                )
                u1 += n1
                u2 += n2
                # print("Post noise addition: power u1",torch.mean(torch.abs(u1) ** 2))
                logger.debug(
                    "Noise power u1 %s dBm",
                    10 * torch.log10(torch.mean(torch.abs(n1) ** 2)).item() + 30,
                )
                logger.debug(
                    "Post noise addition: power u1 %s dBm",
                    10 * torch.log10(torch.mean(torch.abs(u1) ** 2)).item() + 30,
                )
        return u1, u2

    def __call__(self, *args, **kwargs):
        """Execute `forward()` on call."""
        return self.forward(*args, **kwargs)


@attr.define
class SSFMPropagationSinglePol(torch.nn.Module):
    """Non-linear fibre propagation (with SSFM).

    This class is adapted from Code written by Dominik Rimpf published under the MIT
    license [https://gitlab.com/domrim/bachelorarbeit-code]

    This class solves the nonlinear Schrodinger equation for pulse propagation in an
    optical fiber using the split-step Fourier method.
    The actual implementation is the local error method Sec II.E of Optimization of the
    Split-Step Fourier Method in Modeling Optical-Fiber Communications Systems
    https://doi.org/10.1109/JLT.2003.808628


    :param dt: time step between samples (sample time)
    :param dz: propagation stepsize (delta z) of SSFM (segment length)
    :param alphadb: power loss coeficient [dB/km]
    :param beta2: dispersion polynomial coefficient
    :param gamma: nonlinearity coefficient
    :param length_span: Length of span between amplifiers
    :param num_span: Number of spans where num_span * length_span is full fiber channel
                     length
    :param delta_G: Goal local error
    :param maxiter: Number of step-size iterations

    """

    dt: torch.tensor
    dz: torch.tensor
    alphadb: torch.tensor
    betap: torch.tensor
    gamma: torch.tensor
    length_span: torch.tensor
    num_span: torch.tensor
    delta_G: torch.tensor = 1e-3
    maxiter: torch.tensor = 4
    alphalin: torch.tensor = attr.field(init=False)
    amp: object = None
    solver_method: str = "localerror"

    def __attrs_pre_init__(self):
        """Pre-init for attrs classes."""
        super(SSFMPropagationSinglePol, self).__init__()

    def __attrs_post_init__(self):
        """Post-init for attrs classes."""
        self.alphalin = self.alphadb * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km

    def get_operators(self, dz):
        """Calculate linear operators for given step length."""
        # Linear operator (frequency domain) (EDFA) for RAMAN just set alphadb to zero
        h_dz = torch.exp(self.h_basis * dz)
        h_dz_2 = torch.exp(self.h_basis * dz * 0.5)

        # Nonlinear operator (time domain)
        def nonlinear_operator(u):
            return torch.exp(-1j * self.gamma * torch.square(torch.absolute(u)) * dz)

        return (h_dz, h_dz_2, nonlinear_operator)

    def forward(self, u):  # noqa: C901
        """
        Compute the propagation through the configured fiber-optical channel.

        :param u0: input signal (array)
        :returns: array signal at the end of the fiber
        """
        w = (2 * torch.pi * torch.fft.fftfreq(u.shape[0], self.dt * 1e12)).to(
            self.gamma.device
        )  # THz

        self.h_basis = -self.alphalin / 2 - self.betap[0]
        wc = w
        for i in range(1, self.betap.shape[0]):
            self.h_basis = (
                self.h_basis - (1j * self.betap[i] / np.math.factorial(i)) * wc
            )
            wc = wc * w

        uz = u
        # For EDFA apply amplification at the end of each span
        # For RAMAN apply amplification at the beginning of each segment plus at the end
        # of the simulation
        # logger.info("y0 norm %s", torch.linalg.vector_norm(y).item())
        dz = self.dz
        for span in range(self.num_span):
            # logger.info("SSFM Span %s/%s", span + 1, self.num_span.item())
            z = 0
            iter = 0
            (h_2dz, h_dz, NOP2dz) = self.get_operators(2 * dz)
            (h_dz, h_dz_2, NOPdz) = self.get_operators(dz)
            new_dz = dz
            while z + 2 * new_dz <= self.length_span and new_dz != 0:
                if new_dz != dz:
                    dz = new_dz
                    (h_2dz, h_dz, NOP2dz) = self.get_operators(2 * dz)
                    (h_dz, h_dz_2, NOPdz) = self.get_operators(dz)

                uf = symmetrical_SSFM_step(uz, h_dz_2, NOPdz)
                uf = symmetrical_SSFM_step(uf, h_dz_2, NOPdz)
                # logger.info("uc norm %s", torch.linalg.vector_norm(uc).item())
                # logger.info("uf norm %s", torch.linalg.vector_norm(uf).item())
                good_sol_flag = True
                if self.solver_method == "localerror":
                    uc = symmetrical_SSFM_step(uz, h_dz, NOP2dz)
                    delta = torch.linalg.vector_norm(
                        uf - uc
                    ) / torch.linalg.vector_norm(uf)

                    if delta > 2 * self.delta_G:
                        new_dz = dz / 2
                        good_sol_flag = False
                        iter = iter + 1
                    elif self.delta_G <= delta <= 2 * self.delta_G:
                        new_dz = dz / 1.259921049894873
                    elif delta < 0.5 * self.delta_G:
                        new_dz = dz * 1.259921049894873

                if good_sol_flag or iter >= self.maxiter:
                    if self.solver_method == "localerror":
                        uz = (4 * uf - uc) / 3
                    elif self.solver_method == "fixedstep":
                        uz = uf
                    else:
                        uz = uf
                    z = z + 2 * dz
                    if iter >= self.maxiter:
                        logger.info(
                            "WARNING: Failed to converge to goal local error \
                            of %s in %s iterations",
                            self.delta_G,
                            self.maxiter,
                        )
                        new_dz = self.dz

                    iter = 0

                    if z + 2 * new_dz > self.length_span:
                        new_dz = (self.length_span - z) / 2

            if self.amp is not None:
                uz = self.amp(uz, 2)

        return uz

    def __call__(self, *args, **kwargs):
        """Execute `forward()` on call."""
        return self.forward(*args, **kwargs)


@attr.define
class SSFMPropagationDualPol(torch.nn.Module):
    """Coupled NLS fibre propagation (with SSFM).

    This class is adapted from Code written by Dominik Rimpf published under the MIT
    license [https://gitlab.com/domrim/bachelorarbeit-code]

    This class solves the coupled nonlinear Schrodinger equation for pulse propagation
    in an optical fiber using the split-step Fourier method.
    The implementation and name are based on
    https://photonics.umd.edu/software/ssprop/vector-version/

    :param dt: time step between samples (sample time)
    :param dz: propagation stepsize (delta z) of SSFM (segment length)
    :param alphaa_db: power loss coeficient [dB/km] for eigenstate a
    :param alphab_db: power loss coeficient [dB/km] for eigenstate b
    :param betapa: dispersion polynomial coefficient vector for eigenstate a
    :param betapb: dispersion polynomial coefficient vector for eigenstate b
    :param gamma: nonlinearity coefficient
    :param length_span: Length of span between amplifiers
    :param num_span: Number of spans where num_span * length_span is full fiber channel
                     length
    :param delta_G: Goal local error (optional) (default 1e-3)
    :param maxiter: Number of step-size iterations (optional) (default 4)
    :param psp: Principal eigenstate of the fiber specified as a 2-vector containing
                the angles Psi and Chi (optional) (default [0,0])
    :param solution_method: String that specifies which method to use when performing
                            the split-step calculations. Supported methods elliptical
                            and circular (optional) (default elliptical)

    """

    dt: torch.tensor
    dz: torch.tensor
    alphaa_db: torch.tensor
    alphab_db: torch.tensor
    betapa: torch.tensor
    betapb: torch.tensor
    gamma: torch.tensor
    length_span: torch.tensor
    num_span: torch.tensor
    delta_G: torch.tensor = 1e-3
    maxiter: torch.tensor = 4
    psp: torch.tensor = torch.tensor([0, 0])
    solution_method = "elliptical"
    alphaa_lin: torch.tensor = attr.field(init=False)
    alphab_lin: torch.tensor = attr.field(init=False)
    amp: object = None
    solver_method: str = "localerror"

    def __attrs_pre_init__(self):
        """Pre-init for attrs classes."""
        super(SSFMPropagationDualPol, self).__init__()

    def __attrs_post_init__(self):
        """Post-init for attrs classes."""
        self.alphaa_lin = self.alphaa_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km
        self.alphab_lin = self.alphab_db * (
            torch.log(torch.tensor(10)) / 10
        )  # dB/km -> 1/km
        self.Psi = self.psp[0]
        self.Chi = self.psp[1]

        if self.solution_method == "elliptical":
            self.TM11 = torch.cos(self.Psi) * torch.cos(self.Chi) + 1j * torch.sin(
                self.Psi
            ) * torch.sin(self.Chi)
            self.TM12 = torch.sin(self.Psi) * torch.cos(self.Chi) - 1j * torch.cos(
                self.Psi
            ) * torch.sin(self.Chi)
            self.TM21 = -torch.sin(self.Psi) * torch.cos(self.Chi) - 1j * torch.cos(
                self.Psi
            ) * torch.sin(self.Chi)
            self.TM22 = torch.cos(self.Psi) * torch.cos(self.Chi) - 1j * torch.sin(
                self.Psi
            ) * torch.sin(self.Chi)

            self.n1_basis = (
                (1j / 3) * self.gamma * (2 + torch.square(torch.cos(2 * self.Chi)))
            )
            self.n2_basis = (
                (1j / 3) * self.gamma * (2 + 2 * torch.square(torch.sin(2 * self.Chi)))
            )

        elif self.solution_method == "circular":
            sqrt2 = torch.sqrt(2)
            self.TM11 = 1 / sqrt2
            self.TM12 = -1j / sqrt2
            self.TM21 = self.TM12
            self.TM22 = self.TM11

            self.n1_basis = (2j / 3) * self.gamma
            self.n2_basis = (4j / 3) * self.gamma
        else:
            raise ValueError("Unknown solution method.")

    def get_operators(self, dz):
        """Obtain linear operators for given step length."""
        """
        Create operators based on step-size
        """
        if self.solution_method == "elliptical":
            # Linear operator (frequency domain) (EDFA)
            h11_dz = torch.exp(self.ha_basis * dz)
            h11_dz_2 = torch.exp(self.ha_basis * dz / 2)
            h22_dz = torch.exp(self.hb_basis * dz)
            h22_dz_2 = torch.exp(self.hb_basis * dz / 2)
            h12_dz = 0
            h12_dz_2 = 0
            h21_dz = 0
            h21_dz_2 = 0

        elif self.solution_method == "circular":
            self.solution_method = "elliptical"
            ha_dz, ha_dz_2, _, _, _, _, hb_dz, hb_dz_2 = self.get_operators(dz)
            self.solution_method = "circular"

            sp = 0.5 * (1 + torch.sin(2 * self.Chi))
            sn = 0.5 * (1 - torch.sin(2 * self.Chi))
            h11_dz = sp * ha_dz + sn * hb_dz
            h11_dz_2 = sp * ha_dz_2 + sn * hb_dz_2

            h12_dz = 0.5j * torch.exp(-2j * self.Psi) * torch.cos(2 * self.Chi)
            h12_dz_2 = h12_dz * (ha_dz_2 - hb_dz_2)
            h12_dz = h12_dz * (ha_dz - hb_dz)

            h21_dz = -0.5j * torch.exp(2j * self.Psi) * torch.cos(2 * self.Chi)
            h21_dz_2 = h21_dz * (ha_dz_2 - hb_dz_2)
            h21_dz = h21_dz * (ha_dz - hb_dz)

            h22_dz = sn * ha_dz + sp * hb_dz
            h22_dz_2 = sn * ha_dz_2 + sp * hb_dz_2

        else:
            raise ValueError("Unknown solution method.")

        # Nonlinear operator (time domain)
        def nonlinear_operator(u1, u2):
            u1abssqre = torch.square(torch.absolute(u1))
            u2abssqre = torch.square(torch.absolute(u2))
            return torch.exp(
                ((self.n1_basis * u1abssqre) + (self.n2_basis * u2abssqre)) * dz
            ), torch.exp(
                ((self.n1_basis * u2abssqre) + (self.n2_basis * u1abssqre)) * dz
            )

        return (
            h11_dz,
            h11_dz_2,
            h12_dz,
            h12_dz_2,
            h21_dz,
            h21_dz_2,
            h22_dz,
            h22_dz_2,
            nonlinear_operator,
        )

    def forward(self, ux, uy):
        """
        Compute the progpagation for the dual polarization signal through the \
        fiber optical channel.

        :param ux: input signal in x-polarization.
        :param uy: input signal in y-polarization.
        :returns: signal at the end of the fiber
        """
        w = (2 * torch.pi * torch.fft.fftfreq(ux.shape[0], self.dt * 1e12)).to(
            self.betapa.device
        )  # THz

        self.ha_basis = -self.alphaa_lin / 2 - self.betapa[0]
        self.hb_basis = -self.alphab_lin / 2 - self.betapb[0]
        wc = w
        for i in range(1, self.betapa.shape[0]):
            self.ha_basis = (
                self.ha_basis - (1j * self.betapa[i] / math.factorial(i)) * wc
            )
            self.hb_basis = (
                self.hb_basis - (1j * self.betapb[i] / math.factorial(i)) * wc
            )
            wc = wc * w

        u1 = self.TM11 * ux + self.TM12 * uy
        u2 = self.TM21 * ux + self.TM22 * uy
        # For EDFA apply amplification at the end of each span

        # logger.info("y0 norm %s", torch.linalg.vector_norm(y).item())
        dz = self.dz
        # logger.info("u1 norm %s", torch.linalg.vector_norm(u1).item())
        # logger.info("u2 norm %s", torch.linalg.vector_norm(u2).item())
        for span in range(self.num_span):
            # logger.info("SSFM Span %s/%s", span + 1, self.num_span.item())
            z = 0
            iter = 0
            (
                h11_2dz,
                h11_dz,
                h12_2dz,
                h12_dz,
                h21_2dz,
                h21_dz,
                h22_2dz,
                h22_dz,
                NOP2dz,
            ) = self.get_operators(2 * dz)
            (
                h11_dz,
                h11_dz_2,
                h12_dz,
                h12_dz_2,
                h21_dz,
                h21_dz_2,
                h22_dz,
                h22_dz_2,
                NOPdz,
            ) = self.get_operators(dz)
            new_dz = dz
            while z + 2 * new_dz <= self.length_span and new_dz != 0:
                if new_dz != dz:
                    dz = new_dz
                    (
                        h11_2dz,
                        h11_dz,
                        h12_2dz,
                        h12_dz,
                        h21_2dz,
                        h21_dz,
                        h22_2dz,
                        h22_dz,
                        NOP2dz,
                    ) = self.get_operators(2 * dz)
                    (
                        h11_dz,
                        h11_dz_2,
                        h12_dz,
                        h12_dz_2,
                        h21_dz,
                        h21_dz_2,
                        h22_dz,
                        h22_dz_2,
                        NOPdz,
                    ) = self.get_operators(dz)

                u1f, u2f = symmetrical_SSPROPV_step(
                    u1, u2, h11_dz_2, h12_dz_2, h21_dz_2, h22_dz_2, NOPdz
                )
                u1f, u2f = symmetrical_SSPROPV_step(
                    u1f, u2f, h11_dz_2, h12_dz_2, h21_dz_2, h22_dz_2, NOPdz
                )
                good_sol_flag = True
                # logger.info("u1c norm %s", torch.linalg.vector_norm(u1c).item())
                # logger.info("u2c norm %s", torch.linalg.vector_norm(u2c).item())
                # logger.info("u1f norm %s", torch.linalg.vector_norm(u1f).item())
                # logger.info("u2f norm %s", torch.linalg.vector_norm(u2f).item())
                if self.solution_method == "localerror":
                    u1c, u2c = symmetrical_SSPROPV_step(
                        u1, u2, h11_dz, h12_dz, h21_dz, h22_dz, NOP2dz
                    )
                    delta = (
                        torch.linalg.vector_norm(u1f - u1c)
                        + torch.linalg.vector_norm(u2f - u2c)
                    ) / (torch.linalg.vector_norm(u1f) + torch.linalg.vector_norm(u2f))
                    # logger.info("dz is %s, delta is %s", dz.item(), delta.item())

                    if delta > 2 * self.delta_G:
                        new_dz = dz / 2
                        good_sol_flag = False
                        iter = iter + 1
                    elif self.delta_G <= delta <= 2 * self.delta_G:
                        new_dz = dz / 1.259921049894873
                    elif delta < 0.5 * self.delta_G:
                        new_dz = dz * 1.259921049894873

                if good_sol_flag or iter >= self.maxiter:
                    if self.solution_method == "localerror":
                        u1 = (4 * u1f - u1c) / 3
                        u2 = (4 * u2f - u2c) / 3
                    elif self.solution_method == "fixedstep":
                        u1 = u1f
                        u2 = u2f
                    else:
                        u1 = u1f
                        u2 = u2f

                    z = z + 2 * dz

                    if self.amp is not None and self.amp.amp_type == "Raman":
                        u1, u2 = self.amp(u1, u2, 2 * dz)  # TODO Where to add noise?

                    if iter >= self.maxiter:
                        logger.info(
                            "WARNING: Failed to converge to goal local error \
                             of %s in %s iterations",
                            self.delta_G,
                            self.maxiter,
                        )
                        new_dz = self.dz

                    iter = 0

                    if z + 2 * new_dz > self.length_span:
                        new_dz = (self.length_span - z) / 2

                    # logger.info("z is %s,new_dz is %s", z.item(), new_dz.item())

            if self.amp is not None and self.amp.amp_type == "EDFA":
                u1, u2 = self.amp(u1, u2, 2)  # TODO Where to add noise?
            # logger.info("yz norm %s", torch.linalg.vector_norm(yz).item())
        ux = self.TM22 * u1 - self.TM12 * u2
        uy = -self.TM21 * u1 + self.TM11 * u2
        return ux, uy

    def __call__(self, *args, **kwargs):
        """Execute `forward()` on call."""
        return self.forward(*args, **kwargs)


def SSFM_step(signal, linear_op, nonlinear_op):
    """Calculate single polarization SSFM step."""
    f_signal = torch.fft.fft(signal) * linear_op
    t_signal = torch.fft.ifft(f_signal)
    t_signal = t_signal * nonlinear_op(t_signal)
    return t_signal


def symmetrical_SSFM_step(signal, half_linear_op, nonlinear_op):
    """Calculate symmetrical single polarization SSFM step."""
    f_signal = torch.fft.fft(signal) * half_linear_op
    t_signal = torch.fft.ifft(f_signal)
    t_signal = t_signal * nonlinear_op(t_signal)
    return torch.fft.ifft(torch.fft.fft(t_signal) * half_linear_op)


def symmetrical_SSPROPV_step(u1, u2, h11, h12, h21, h22, NOP):
    """Calculate symmetrical dual polarization SSFM step."""
    u1f = torch.fft.fft(u1)
    u2f = torch.fft.fft(u2)
    u1 = torch.fft.ifft(h11 * u1f + h12 * u2f)
    u2 = torch.fft.ifft(h21 * u1f + h22 * u2f)
    nu1, nu2 = NOP(u1, u2)
    u1 = nu1 * u1
    u2 = nu2 * u2
    u1f = torch.fft.fft(u1)
    u2f = torch.fft.fft(u2)
    u1 = torch.fft.ifft(h11 * u1f + h12 * u2f)
    u2 = torch.fft.ifft(h21 * u1f + h22 * u2f)
    return u1, u2


def SSFM_halfstep_end(signal, linear_op):
    """Calculate the final halfstep of the single polarization SSFM."""
    return torch.fft.ifft(torch.fft.fft(signal) * linear_op)


class WDMMux(torch.nn.Module):
    """Perform configurable muxing of baseband channels to simulate WDM."""

    def __init__(self, n_channels, spacing, samp_rate, sim_rate):
        """Initialize WDMMux."""
        super(WDMMux, self).__init__()

        self.n_up = int(sim_rate // samp_rate)
        if abs(samp_rate * self.n_up - sim_rate) > 0.1:
            raise ValueError(
                "Currently sim_rate must be an integer multiple of samp_rate"
            )

        self.freq_coeff = 2 * torch.pi * spacing / sim_rate
        self.n_channels = n_channels

    def forward(self, signals):
        """Modulate the signals on WDM channels in the digital baseband."""
        upsampled_signals = torch.zeros(
            (signals.shape[0], signals.shape[1] * self.n_up),
            device=signals.device,
            dtype=torch.complex64,
        )
        for idx, signal in enumerate(signals):
            upsampled_signals[idx] = upsample(
                self.n_up, signal, filter_length=10001, filter_gain=np.sqrt(self.n_up)
            )
            channel_idx = -(self.n_channels // 2) + idx
            upsampled_signals[idx] = upsampled_signals[idx] * torch.exp(
                1j
                * self.freq_coeff
                * channel_idx
                * torch.arange(upsampled_signals[idx].shape[0])
            )
        return torch.sum(upsampled_signals, dim=0)


class PolyPhaseChannelizer(torch.nn.Module):
    """Perform poly-phase channelization with variable filter coefficients."""

    def __init__(self, coeff, n_down):
        """Init PolyPhaseChannelizer."""
        super(PolyPhaseChannelizer, self).__init__()

        self.permutation_matrix = (
            torch.eye(n_down, dtype=coeff.dtype, device=coeff.device).flip(1).roll(1, 1)
        )
        self.kernels = torch.transpose(coeff.reshape(-1, n_down), -1, -2)
        self.n_down = n_down

    def forward(self, signal):
        """Apply poly-phase channelization on a wideband signal."""
        # Bring signal into correct shape for polyphase filtering
        num_periods = int(np.ceil(signal.shape[0] / self.n_down).item())
        target_len = num_periods * self.n_down
        pad_len = target_len - signal.shape[0]
        signal_cut = torch.nn.functional.pad(signal, (0, pad_len))
        signal_polyphase = torch.transpose(signal_cut.reshape(-1, self.n_down), -1, -2)

        # One additional signal point is added after convolution to guarantee
        # correct overlap of coefficients
        filtered_signals = torch.zeros(
            (self.n_down, num_periods + self.kernels.shape[1]),
            dtype=signal_cut.dtype,
            device=signal.device,
        )

        signal_rotated = self.permutation_matrix.mm(signal_polyphase)

        for idx in range(self.n_down):
            sig = signal_rotated[idx]
            kernel = self.kernels[idx]
            filt_sig = functional.torch.convolve_overlap_save(sig, kernel, "full")
            if idx == 0:
                filtered_signals[idx] = torch.nn.functional.pad(filt_sig, (0, 1))
            else:
                filtered_signals[idx] = torch.nn.functional.pad(filt_sig, (1, 0))
        # This corresponds to filter operation with mode "valid" -
        # remove last sample since it's value is invalid
        return (
            self.n_down
            * torch.fft.ifft(filtered_signals, dim=0).resolve_conj()[
                :,
                int(self.kernels.shape[1] // 2) : -int(self.kernels.shape[1] // 2) - 1,
            ]
        )


class WDMDemux(torch.nn.Module):
    """WDM Demuxing for a configurable number of channels and system parameters."""

    def __init__(
        self, n_channels, spacing, sym_rate, sim_rate, used_channels, method="classical"
    ):
        """Initialize WDMDemux."""
        super(WDMDemux, self).__init__()

        self.n_down = n_channels  # Use the Polyphase Filterbank approach
        self.n_channels = n_channels
        self.wdm_channels = used_channels
        self.sym_rate = sym_rate
        self.sim_rate = sim_rate
        self.spacing = spacing

        self.method = method

        if self.method == "polyphase":
            phase_filter_len = 1024
            cutoff = 1 / self.n_channels
            downsampling_prototype_filter = brickwall_filter(
                cutoff,
                filter_length=phase_filter_len * self.n_down,
                filter_gain=np.sqrt(self.n_channels),
                window="blackmann_harris",
            )

            self.channelizer = PolyPhaseChannelizer(
                downsampling_prototype_filter, n_channels
            )

    def forward(self, signal):
        """WDM Demux the wide-band signal."""
        if self.method == "classical":
            return self.classical_demux(signal)[self.wdm_channels]
        elif self.method == "polyphase":
            return self.polyphase_demux(signal)[self.wdm_channels]
        else:
            raise Exception(f"Demux method {self.method} unknown!")

    def polyphase_demux(self, signal):
        """Peform WDM Demuxing with the Polyphase Channelizer method."""
        return self.channelizer(signal)

    def classical_demux(self, signal):
        """Perform WDM Demuxing for each channel individually."""
        filter_len = 10001

        results = []
        for n in torch.arange(self.n_channels):
            signal_shifted = signal * torch.exp(
                -1j
                * 2
                * torch.pi
                * n
                * self.spacing
                / self.sim_rate
                * torch.arange(signal.shape[0])
            )
            signal_down = downsample(
                self.n_down, signal_shifted, filter_len, filter_gain=self.n_down
            ).unsqueeze(0)
            results.append(signal_down)
        results = torch.cat(results, dim=0)
        return results
