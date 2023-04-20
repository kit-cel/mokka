"""Equalizer blocks after channel before decoder/demodulator - in PyTorch."""
import torch
import numpy as np


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

    def forward(self, y):
        """
        Peform chromatic dispersion compensation in the frequency domain.

        :param y: received complex single polarization signal y
        :returns: equalized signal
        """
        nt = y.size()[0]
        dw = (2 * np.pi * torch.fft.fftfreq(nt, self.dt * 1e12)).to(y.device)  # *(1/4)

        linear_operator = torch.exp(
            -1j * self.beta2 / 2 * dw**2 * self.channel_length
        )
        Y = torch.fft.fft(y)
        Y = Y / linear_operator
        return torch.fft.ifft(Y)
