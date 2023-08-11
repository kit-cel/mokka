"""Module implementing pulseshaping with PyTorch."""

import torch
import attr
import math

from .. import functional

torch.pi = math.pi


@attr.define
class PulseShaping(torch.nn.Module):
    """
    PulseShaping filter class.

    Several classical shapes, a filter function \
    and a matched filter function are provided.
    """

    _impulse_response: torch.tensor
    learnable: bool = False
    impulse_response_conj: torch.tensor = attr.field(init=False)

    def __attrs_post_init__(self):
        """Perform construction of other parts."""
        super(PulseShaping, self).__init__()
        if self.learnable:
            self.impulse_response = torch.nn.Parameter(self._impulse_response)
            self.impulse_response_conj = torch.nn.Parameter(
                torch.resolve_conj(self._impulse_response.conj())
            )
        else:
            self.register_buffer("impulse_response", self._impulse_response)
            self.register_buffer(
                "impulse_response_conj",
                torch.resolve_conj(self._impulse_response.conj()),
            )
        self.ir_norm = torch.linalg.norm(self.impulse_response)

    def forward(self, y, n_up):
        """Perform pulse shaping.

        :param y: transmit signal
        :param n_up: upsampling factor
        :returns: shaped and upsampled transmit signal at sample rate
        """
        y_up = torch.zeros(y.shape[0] * n_up, dtype=torch.complex64, device=y.device)
        y_up[::n_up] = y
        y_shaped = functional.torch.convolve_overlap_save(
            y_up, self.impulse_response, mode="full"
        )
        return y_shaped

    def matched(self, r, n_down):
        """Perform matched filtering.

        This function assumes perfect timing sync.

        :param r: received complex signal
        :param n_down: downsampling factor
        :returns: filtered and downsampled signal at symbol rate
        """
        y_filt = functional.torch.convolve(r, self.impulse_response_conj / n_down)
        offset = self.impulse_response_conj.shape[0] - 1
        y = y_filt[::n_down][int(offset / n_down) : -int(offset / n_down)]
        return y

    def normalize_filter(self):
        """Normalize filter taps saved in this object."""
        # print('saved ir_norm',self.ir_norm)
        # print('current ir_norm',torch.linalg.norm(self.impulse_response))
        self.impulse_response = torch.nn.Parameter(
            self.ir_norm
            * self.impulse_response
            / torch.linalg.norm(self.impulse_response)
        )
        self.impulse_response_conj = torch.nn.Parameter(
            self.ir_norm
            * self.impulse_response_conj
            / torch.linalg.norm(self.impulse_response_conj)
        )
        # print('current ir_norm',torch.linalg.norm(self.impulse_response))

    @classmethod
    def get_rc_ir(cls, syms, r, n_up, learnable=False):
        """Determine normed coefficients of an RC filter.

        Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
        At poles, l'Hospital was used

        :param syms: "normed" length of ir. ir-length will be 2*syms+1
        :param r: roll-off factor [Float]
        :param n_up: upsampling factor [Int]

        :returns: tuple containing time-index-array and impulse response in an array

        """
        with torch.no_grad():
            # initialize output length and sample time
            T_symbol = 1.0
            T_sample = 1.0 / n_up
            # length of one sample is the symbol-duration divided
            # by the oversampling factor (=1/sampling rate)
            T_ir = 2 * syms * T_symbol
            # Duration of the impulse response is positive and negative
            # normed symbols added multplied by Symbol Duration
            ir = torch.zeros(int(T_ir / T_sample) + 1)
            # samples of impulse response is definied by duration
            # of the ir divided by the sample time plus one for the 0th sample

            # time indices and sampled time
            k_steps = torch.arange(
                -T_ir / T_sample / 2, T_ir / T_sample / 2 + 1, dtype=int
            )
            t_steps = k_steps * T_sample

            for k in k_steps:
                if t_steps[k] == 0:
                    ir[k] = 1.0 / T_symbol

                elif r != 0 and torch.abs(t_steps[k]) == T_symbol / (2.0 * r):
                    ir[k] = r / (2.0 * T_symbol) * torch.sin(torch.pi / (2.0 * r))

                else:
                    ir[k] = (
                        torch.sin(torch.pi * t_steps[k] / T_symbol)
                        / torch.pi
                        / t_steps[k]
                        * torch.cos(r * torch.pi * t_steps[k] / T_symbol)
                        / (1.0 - (2.0 * r * t_steps[k] / T_symbol) ** 2)
                    )

            # Norming on Energy = 1
            ir /= torch.linalg.norm(ir) * torch.sqrt(T_sample)

        return cls(ir, learnable)

    @classmethod
    def get_rrc_ir(cls, syms, r, n_up, learnable=False):
        """Determine normed coefficients of an RRC filter.

        This function is adapted from Sourcecode written by Dominik Rimpf and was
        published under the MIT license [https://gitlab.com/domrim/bachelorarbeit-code]

        Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
        At poles, l'Hospital was used

        :param syms: "normed" length of ir. ir-length will be 2*syms+1
        :param r: roll-off factor [Float]
        :param n_up: upsampling factor [Int]

        :returns: tuple containing time-index-array and impulse response in an array

        """
        with torch.no_grad():
            # initialize output length and sample time
            T_symbol = 1.0
            T_sample = 1 / n_up
            T_ir = 2 * syms * T_symbol
            # Duration of the impulse response is positive and negative
            # normed symbols added multplied by Symbol Duration
            ir = torch.zeros(int(T_ir / T_sample) + 1)
            # samples of impulse response is definied by duration of
            # the ir divided by the sample time plus one for the 0th sample

            # time indices and sampled time
            k_steps = torch.arange(
                -T_ir / T_sample / 2, T_ir / T_sample / 2 + 1, dtype=torch.int
            )
            t_steps = k_steps * T_sample

            for k in k_steps.int():
                if t_steps[k] == 0:
                    ir[k] = (torch.pi + 4.0 * r - torch.pi * r) / (torch.pi * T_symbol)

                elif r != 0 and torch.abs(t_steps[k]) == T_symbol / (4.0 * r):
                    ir[k] = (
                        r
                        * (
                            -2.0
                            * torch.cos(
                                torch.pi * torch.tensor(1.0 + r) / torch.tensor(4.0 * r)
                            )
                            + torch.pi
                            * torch.sin(
                                torch.pi * torch.tensor(1.0 + r) / torch.tensor(4.0 * r)
                            )
                        )
                        / (torch.pi * T_symbol)
                    )

                else:
                    ir[k] = (
                        4.0
                        * r
                        * t_steps[k]
                        / T_symbol
                        * torch.cos(torch.pi * (1.0 + r) * t_steps[k] / T_symbol)
                        + torch.sin(torch.pi * (1.0 - r) * t_steps[k] / T_symbol)
                    ) / (
                        (1.0 - (4.0 * r * t_steps[k] / T_symbol) ** 2)
                        * torch.pi
                        * t_steps[k]
                    )

            # Norming on Energy = 1
            ir /= torch.linalg.norm(ir) * torch.sqrt(torch.tensor(T_sample))
            ir = ir.type(torch.complex64)
        return cls(ir, learnable)


class MidriseQuantizer(torch.nn.Module):
    """Implement a Quantizer which translates a continuos signal into 2**m levels."""

    def __init__(self, bit, max_amplitude=1.0):
        """Construct `MidriseQuuantizer`."""
        super(MidriseQuantizer, self).__init__()

        self.delta = max_amplitude / (2 ** (bit - 1))
        self.max_amplitude = max_amplitude

    def forward(self, y):
        """Perform quantization."""
        y_quant = (
            torch.sign(y)
            * self.delta
            * (
                torch.floor(
                    torch.clamp(torch.abs(y), 0, self.max_amplitude - self.delta / 2)
                    / self.delta
                )
                + 0.5
            )
        )
        y_quant = y_quant + y - y.detach()  # Retain the gradients
        return y_quant


class SoftMidriseQuantizer(torch.nn.Module):
    """
    Use tanh() with temperature to approximate Midrise quantization.

    For temperature -> 0 the SoftMidriseQuantizer approximates the
    MidriseQuantizer.
    """

    def __init__(self, bit, max_amplitude=1.0, temperature=0.01):
        """Initialize SoftMidriseQuantizer."""
        super(SoftMidriseQuantizer, self).__init__()

        self.delta = max_amplitude / ((2 ** (bit - 1)))
        self.levels = 2**bit
        self.max_amplitude = max_amplitude
        self.temperature = temperature

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Apply quantization to signal.

        :param x: real signal to quantize.
        """
        y = torch.zeros_like(x, device=x.device)
        y = torch.sum(
            self.delta
            / 2
            * self.tanh(
                (
                    x[:, None]
                    + self.delta
                    * (
                        (self.levels // 2 - 1)
                        - torch.arange(self.levels - 1).view(
                            *(1,) * len(x.size()) + (-1,)
                        )
                    )
                )
                / self.temperature
            ),
            dim=-1,
        )
        return y


def cosine_sum_window(a_0, length=501):
    """Get cosine sum window coefficients."""
    w_n = a_0 - (1 - a_0) * torch.cos(
        (2 * torch.pi * torch.arange(length, device=a_0.device)) / length
    )
    return w_n


def hann_window(length=501):
    """Get Hann window coefficients."""
    return cosine_sum_window(torch.tensor(0.5), length)


def hamming_window(length=501):
    """Get Hamming window coefficients."""
    return cosine_sum_window(torch.tensor(25 / 46), length)


def blackman_harris_window(length=501):
    """Get Blackman-harris window coefficients."""
    a_0 = 0.35875
    a_1 = 0.48829
    a_2 = 0.14128
    a_3 = 0.01168

    n = torch.arange(length) / length
    w_n = (
        a_0
        - a_1 * torch.cos(2 * torch.pi * n)
        + a_2 * torch.cos(4 * torch.pi * n)
        - a_3 * torch.cos(6 * torch.pi * n)
    )
    return w_n


def brickwall_filter(
    cutoff, filter_length, filter_gain, window="hann", dtype=torch.complex64
):
    """Approximate brickwall filter in fourier domain with configurable window."""
    coeff = cutoff * torch.sinc(
        cutoff
        * torch.linspace(
            -(filter_length - 1) / 2,
            (filter_length - 1) / 2,
            filter_length,
            dtype=dtype,
        )
    )
    coeff = filter_gain * coeff / torch.sum(coeff)
    if window == "hann":
        window = hann_window(coeff.shape[0])
    elif window == "hamming":
        window = hamming_window(coeff.shape[0])
    elif window == "blackmann_harris":
        window = blackman_harris_window(coeff.shape[0])
    elif window == "rect":
        window = 1
    else:
        raise Exception(f"Window function {window} unknown!")
    coeff = coeff * window
    return coeff


def upsample(n_up, signal, filter_length=501, filter_gain=1):
    """Perform upsampling on signal."""
    signal = signal.unsqueeze(-1)
    padding = torch.zeros(
        (*signal.size()[:-1], n_up - 1),
        dtype=signal.dtype,
        layout=signal.layout,
        device=signal.device,
    )
    signal = torch.concatenate((signal, padding), dim=-1).view(*signal.size()[:-2], -1)

    # Fourierapprox of upsampling filter
    cutoff = 1 / n_up
    coeff = brickwall_filter(
        cutoff,
        filter_length,
        filter_gain,
        dtype=signal.dtype,
        window="blackmann_harris",
    )
    signal_up = functional.torch.convolve_overlap_save(signal, coeff, "same")
    return signal_up


def downsample(n_down, signal, filter_length=501, filter_gain=1):
    """Perform downsampling on signal."""
    coeff = brickwall_filter(
        1 / n_down,
        filter_length,
        filter_gain,
        dtype=signal.dtype,
        window="blackmann_harris",
    )
    signal = functional.torch.convolve_overlap_save(signal, coeff, "full")
    signal = signal[
        int(filter_length // 2) : -int(
            torch.ceil(torch.tensor(filter_length) / 2).item()
        )
    ][::n_down]
    return signal
