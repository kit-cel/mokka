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
        y_shaped = functional.torch.convolve(y_up, self.impulse_response)
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
                -T_ir / T_sample / 2, T_ir / T_sample / 2 + 1, dtype=int
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
                    torch.clamp(torch.abs(y), 0, self.max_amplitude) / self.delta
                )
                + 0.5
            )
        )
        y_quant = y_quant + y - y.detach()  # Retain the gradients
        return y_quant
