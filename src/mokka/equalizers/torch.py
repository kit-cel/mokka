"""Equalizer blocks after channel before decoder/demodulator - in PyTorch."""

import torch
import numpy as np

from ..functional.torch import convolve_overlap_save


class CD_compensation(torch.nn.Module):
    """
    Class implementing chromatic dispersion compensation in the frequency domain.

    :param dt: sample time [s]
    :param beta2: dispersion coefficient of the optical fiber [ps^2/km]
    :param channel_length: total length of the channel [km]
    """

    def __init__(self, dt, beta2, channel_length):
        """Construct `CD_compensation`."""
        super(CD_compensation, self).__init__()
        self.dt = dt
        self.beta2 = beta2  # ps**2/km
        self.channel_length = channel_length

    def forward(self, y, center_freq=0):
        """
        Peform chromatic dispersion compensation in the frequency domain.

        :param y: received complex single polarization signal y
        :returns: equalized signal
        """
        nt = y.size()[0]
        dw = (2 * np.pi * ((torch.fft.fftfreq(nt, self.dt * 1e12)) + center_freq)).to(
            y.device
        )  # *(1/4)

        linear_operator = torch.exp(1j * self.beta2 / 2 * dw**2 * self.channel_length)
        Y = torch.fft.fft(y)
        Y = Y * linear_operator
        return torch.fft.ifft(Y)


class Butterfly2x2(torch.nn.Module):
    """
    Class implementing the 2x2 complex butterfly filter structure.

    This typically used in dual-polarization fiber-optical communications
    for equalization.
    """

    def __init__(self, taps=None, num_taps=None, trainable=False):
        """
        Initialize Butterfly2x2 Filter.

        :param taps:      filter taps for initialization.
        :param num_taps:  number of taps if no filter taps are provided for
                          initialization
        :param trainable: if the taps should be included in the parameters() attribute
                          to be trainable with automatic differentiation.
        """
        super(Butterfly2x2, self).__init__()
        if taps is None:
            if num_taps is None:
                raise ValueError("Either taps or num_taps must be set")
            filter_taps = torch.zeros((4, num_taps), dtype=torch.complex64) #0.1* torch.ones((4, num_taps), dtype=torch.complex64) #torch.zeros((4, num_taps), dtype=torch.complex64)
            #filter_taps = 1.0
            filter_taps[::, num_taps // 2] = 1.0 #1.0
            self.num_taps = num_taps
        else:
            assert taps.dim() == 2
            assert taps.size()[0] == 4
            assert taps.dtype == torch.complex64
            filter_taps = taps
            self.num_taps = taps.size()[1]
        # We store h_xx, h_xy, h_yy, h_yx
        if trainable == True:
            self.register_parameter("taps", torch.nn.Parameter(filter_taps))
        elif trainable == False:
            self.taps = filter_taps
        else:
            raise ValueError("trainable must be set to either True or False.")
        
    def forward(self, y, mode="valid"):
        """
        Perform 2x2 convolution with taps in self.taps.

        :params y: Signal to convolve
        :params mode: convolution mode (either valid, same, or full)
        :returns: convolved signal
        """
        assert y.dim() == 2
        assert y.size()[0] == 2
        assert y.dtype == torch.complex64

        result_x = convolve_overlap_save(
            y[0, :], self.taps[0, :], mode=mode
        ) + convolve_overlap_save(y[1, :], self.taps[1, :], mode=mode)
        result_y = convolve_overlap_save(
            y[1, :], self.taps[2, :], mode=mode
        ) + convolve_overlap_save(y[0, :], self.taps[3, :], mode=mode)

        return torch.cat((result_x.unsqueeze(0), result_y.unsqueeze(0)), dim=0)
    
    def forward_abssquared(self, y, mode="valid"):
        """
        Perform 2x2 convolution with absolut-squared of taps in self.taps.

        :params y: Signal to convolve
        :params mode: convolution mode (either valid, same, or full)
        :returns: convolved signal
        """
        assert y.dim() == 2
        assert y.size()[0] == 2
        #assert y.dtype == torch.complex64

        h_abs_sq = self.taps.real**2 + self.taps.imag**2

        result_x = convolve_overlap_save(
            y[0, :], h_abs_sq[0, :], mode=mode
        ).real + convolve_overlap_save(y[1, :], h_abs_sq[1, :], mode=mode).real
        result_y = convolve_overlap_save(
            y[1, :], h_abs_sq[2, :], mode=mode
        ).real + convolve_overlap_save(y[0, :], h_abs_sq[3, :], mode=mode).real

        return torch.cat((result_x.unsqueeze(0), result_y.unsqueeze(0)), dim=0)


class Butterfly4x4(torch.nn.Module):
    """
    Class implementing the 4x4 real-valued butterfly filter.

    This structure typically used in dual-polarization fiber-optical communications.
    It is a simplification of the 2x2 complex-valued butterfly filter.
    """

    def __init__(self, taps=None, num_taps=None, trainable=False):
        """
        Initialize the Butterfly4x4 class.

        :param taps: Initial filter taps
        :param num_taps: if taps is empty the number of taps should be set here
        :param trainable: if filter taps should be available in parameters() for
                          training with automatic differentiation
        """
        if taps is None:
            if num_taps is None:
                raise ValueError("Either taps or num_taps must be set")
            filter_taps = torch.ones((16, num_taps), dtype=torch.float)
        else:
            assert taps.dim() == 2
            assert taps.size()[0] == 16
            assert taps.dtype == torch.complex64
            filter_taps = taps
        # We store
        # h_xrxr, h_xrxi, h_xryr, h_xryi,
        # h_xixr, h_xixi, h_xiyr, h_xiyi,
        # h_yrxr, h_yrxi, h_yryr, h_yryi,
        # h_yixr, h_yixi, h_yiyr, h_yiyi
        self.taps = filter_taps

        def forward(self, y, mode="valid"):
            assert y.dim() == 2
            assert y.size()[1] == 2
            assert y.dtype == torch.complex64

            x_r = y[:, 0].real
            x_i = y[:, 0].imag
            y_r = y[:, 1].real
            y_i = y[:, 1].imag

            result = tuple(
                convolve_overlap_save(x_r, self.filter_taps[0 + row * 4, :], mode=mode)
                + convolve_overlap_save(
                    x_i, self.filter_taps[1 + row * 4, :], mode=mode
                )
                + convolve_overlap_save(
                    y_r, self.filter_taps[2 + row * 4, :], mode=mode
                )
                + convolve_overlap_save(y_i, self.filter_taps[3 + row * 4], mode=mode)
                for row in range(4)
            )

            return torch.cat(
                (
                    torch.complex(result[0], result[1]),
                    torch.complex(result[2], result[3]),
                )
            )
