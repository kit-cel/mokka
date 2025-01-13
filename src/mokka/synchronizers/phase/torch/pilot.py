"""Module implementing pilot phase synchronizers."""
import torch
from ....functional.torch import convolve
from ....functional.torch import unwrap_torch


class PilotInserter(torch.nn.Module):
    """
    Insert a pilot symbol at a regular interval.

    Either a QPSK symbol or another pre-defined symbol.
    """

    def __init__(self, block_length=64, pilot_symbol=None, **kwargs):
        """
        Initialize :py:class:`PilotInserter`.

        :param block_length: Blocklength of signal plus one pilot symbol
        :param pilot_symbol: Pilot symbol or pilot sequence for consecutive blocks
        """
        super(PilotInserter, self).__init__(**kwargs)
        self.block_length = block_length
        if pilot_symbol is None:
            self.pilot_symbol = 1 / (torch.sqrt(torch.tensor(2))) * (1 + 1j)
        else:
            self.pilot_symbol = pilot_symbol

    def forward(self, y):
        """
        Insert pilot symbols into transmit signal y.

        :param y: Transmit symbols
        """
        y = torch.squeeze(y)
        if len(y.size()) < 2:
            y = torch.reshape(torch.unsqueeze(y, -1), (-1, self.block_length - 1))
        else:
            assert y.size()[1] == self.block_length - 1

        if not self.pilot_symbol.size() or self.pilot_symbol.size()[0] < y.size()[0]:
            if not self.pilot_symbol.size():
                pilot_size = 1
            else:
                pilot_size = self.pilot_symbol.size()[0]
            pilots = torch.repeat_interleave(
                self.pilot_symbol,
                torch.tensor((-1 * (-y.size()[0] // pilot_size),)),
            )[: y.size()[0]].to(y.device)
        else:
            pilots = self.pilot_symbol.to(y.device)
        pilots = torch.unsqueeze(pilots, -1)
        signal = torch.concat((pilots, y), dim=-1).flatten()
        return signal


class PilotPhaseCompensation(torch.nn.Module):
    """
    Phase compensation based on inserted pilot symbols.

    This block performs the phase compensation based on pilot symbols inserted by
    :py:class:`PilotInserter`.
    """

    def __init__(
        self, block_length=64, pilot_symbol=None, window_size=10 * 64, **kwargs
    ):
        """
        Initialize :py:class:`PilotPhaseCompensation`.

        :param block_length: Block length of the transmit signal plus pilots
        :param pilot_symbol: Either a single pilot symbol or a sequence which is
          inserted to consecutive transmit symbol blocks
        :param window_size: Window size to use for moving average of the phase
          estimation.
        """
        super(PilotPhaseCompensation, self).__init__(**kwargs)
        self.block_length = block_length
        self.window_size = window_size
        if pilot_symbol is None:
            self.pilot_symbol = 1 / (torch.sqrt(torch.tensor(2))) * (1 + 1j)
        else:
            self.pilot_symbol = pilot_symbol

    def forward(self, y):
        """
        Apply the phase compensation to received signal `y`.

        :param y: Received signal
        """
        if len(y.size()) < 2:
            y = torch.reshape(torch.unsqueeze(y, 1), (-1, self.block_length))
        else:
            assert y.size()[1] == self.block_length
        if not self.pilot_symbol.size() or self.pilot_symbol.size()[0] < y.size()[0]:
            if not self.pilot_symbol.size():
                pilot_size = 1
            else:
                pilot_size = self.pilot_symbol.size()[0]
            pilots = torch.repeat_interleave(
                self.pilot_symbol,
                torch.tensor(
                    -1 * (-y.size()[0] // pilot_size),
                ),
            )[: y.size()[0]]
        else:
            pilots = self.pilot_symbol
        pilots = torch.unsqueeze(pilots, -1).to(y.device)

        received_pilots = torch.unsqueeze(y[:, 0], -1)
        phase_est = (
            torch.angle(received_pilots * torch.conj(pilots))
            .type(torch.float32)
            .flatten()
        )
        phase_est = unwrap_torch(phase_est)

        phase_comp = torch.zeros_like(y, dtype=torch.float32)
        phase_comp[:, 0] = phase_est
        phase_comp = phase_comp.flatten()

        filter_kernel = torch.ones((self.window_size,), dtype=torch.complex64).to(
            y.device
        )

        phase_comp_val = torch.zeros_like(phase_comp, dtype=torch.float32)
        phase_comp_val[:: self.block_length] = 1.0
        phase_comp_norm = convolve(
            phase_comp_val.type(torch.float32),
            filter_kernel.type(torch.float32),
            mode="same",
        )

        phase_est = (
            convolve(phase_comp.type(torch.complex64), filter_kernel, mode="same")
            / phase_comp_norm
        )
        y = y.flatten() * torch.exp(-1j * phase_est)
        y = torch.reshape(y, (-1, self.block_length))[:, 1:].flatten()

        return y
