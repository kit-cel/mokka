"""Submodule implementing methods for the blind phase search."""

import torch
import numpy as np
from ....functional.torch import unwrap


class BPS(torch.nn.Module):
    """
    The blind phase search algorithm (BPS).

    This class implements the differentiable and non-differentiable versions of the BPS.
    """

    def __init__(
        self,
        Mtestangles,
        symbols,
        N,
        diff,
        temperature_per_epoch,
        no_sectors,
        avg_filter_type="tri",
        trainable=False,
    ):
        """Construct the BPS class."""
        super(BPS, self).__init__()
        (
            self.Mtestangles,
            self.symbols,
            self.N,
            self.diff,
            self._temperature,
            self.no_sectors,
            self.avg_filter_type,
        ) = (
            Mtestangles,
            symbols,
            N,
            diff,
            temperature_per_epoch,
            no_sectors,
            avg_filter_type,
        )

        if self.avg_filter_type == "rect":
            avg_filter = torch.ones(
                2 * self.N + 1, dtype=torch.float32, device=symbols.device
            )
        elif self.avg_filter_type == "tri":
            avg_filter = torch.cat(
                [
                    torch.arange(
                        1, self.N + 2, dtype=torch.float32, device=symbols.device
                    ),
                    torch.arange(
                        self.N, 0, step=-1, dtype=torch.float32, device=symbols.device
                    ),
                ]
            )
        elif self.avg_filter_type == "nonlinear":
            self.nonlin_filter = torch.nn.Sequential(
                torch.nn.Linear(2 * self.N + 1, 1, device=symbols.device),
                torch.nn.ELU(),
                # torch.nn.Linear(2 * self.N + 1, 1, device=symbols.device),
            )
        else:
            raise ValueError("Unknown averaging filter")

        self.trainable = trainable
        if self.trainable:
            self.sigmoid = torch.nn.Sigmoid()
            self._temperature = torch.nn.Parameter(temperature_per_epoch)
            if self.avg_filter_type != "nonlinear":
                self.avg_filter = torch.nn.Parameter(avg_filter)

            # self.avg_filter = avg_filter  # TODO
        else:
            self.register_buffer("avg_filter", avg_filter)

    @property
    def temperature_per_epoch(self):
        """
        Get temperature setting for differentiable BPS.

        :returns: temperature value.

        """
        if self.trainable:
            return self.sigmoid(self._temperature)
        return self._temperature

    @property
    def normalized_filter(self):
        """
        Return normalized filter.

        :returns: normalized filter taps

        """
        return self.avg_filter / torch.sum(self.avg_filter)

    @temperature_per_epoch.setter
    def temperature_per_epoch(self, new_temperature_per_epoch):
        if self.trainable:
            raise AttributeError(
                "Temperature parameter in a trainable BPS cannot be updated manually"
            )
        self._temperature = new_temperature_per_epoch

    def forward(self, x, *args):
        """
        Perform blind phase search.

        :param x: received complex symbols
        :returns:

        """
        # Ignore other arguments given to other parts of the channel
        bpsx = self.bps_torch(x)
        return torch.squeeze(bpsx[0]), torch.squeeze(bpsx[1])

    def set_constellation(self, constellation):
        """
        Set constellation points in blind phase equalizer.

        :param constellation: array of (complex) constellation points
        """
        self.symbols = constellation
        return self

    def train(self, mode):
        """Enable training.

        :param mode: if true enable training and differentiable mode.
        :returns: the BPS in trainable mode

        """
        self.diff = mode
        return super(BPS, self).train(mode)

    def bps_torch(
        self,
        x,
    ):
        """
        Perform a blind phase search according to [1].

        Parameters
        ----------
        x           : array_like
            input signal (single polarisation)
        Returns
        -------
        Eout    : array_like
            signal with compensated phase
        ph      : array_like
            unwrapped angle from phase recovery


        References
        ----------
        [1] Timo Pfau et al, Hardware-Efficient Coherent Digital Receiver \
            Concept With Feedforward Carrier Recovery for M-QAM Constellations, \
            Journal of Lightwave Technology 27, pp 989-999 (2009)
        """
        softmin = torch.nn.Softmin(dim=1)

        # Use this line
        if self.no_sectors == 4:
            if self.Mtestangles == 2:
                angles = torch.atleast_2d(
                    torch.tensor(
                        (-np.pi / 6, np.pi / 6),
                        device=x.device,
                    )
                )
            else:
                angles = torch.atleast_2d(
                    torch.linspace(
                        -np.pi / 4,
                        np.pi / 4 - (np.pi / 4 + np.pi / 4) / self.Mtestangles,
                        self.Mtestangles,
                        device=x.device,
                    )
                )
        else:
            angles = torch.atleast_2d(
                torch.linspace(
                    -np.pi,
                    np.pi - (np.pi + np.pi) / self.Mtestangles,
                    self.Mtestangles,
                    device=x.device,
                )
            )

        Ew = torch.atleast_2d(x)
        if self.diff is True:
            for i in range(Ew.shape[0]):
                mvg = self._bps_idx_torch(Ew[i], angles)
                if self.no_sectors == 4:
                    smvg = softmin(mvg / self.temperature_per_epoch)
                    sangles = 4 * torch.squeeze(angles)
                    ph = (
                        unwrap(
                            torch.angle(
                                smvg @ torch.cos(sangles)
                                + 1j * (smvg @ torch.sin(sangles))
                            ),
                            np.pi,
                            -1,
                            2 * np.pi,
                        )
                        / 4
                    )
                else:
                    smvg = softmin(mvg / self.temperature_per_epoch)
                    sangles = torch.squeeze(angles)
                    ph = unwrap(
                        torch.angle(
                            smvg @ torch.cos(sangles) + 1j * (smvg @ torch.sin(sangles))
                        ),
                        np.pi,
                        -1,
                        2 * np.pi,
                    )
            return torch.squeeze(Ew * torch.exp(1j * ph)), ph
        else:
            ph = torch.empty(Ew.shape[0], Ew.shape[1], device=x.device)
            for i in range(Ew.shape[0]):
                idx = self._bps_idx_torch(Ew[i], angles)
                ph[i] = self.select_angles_torch(angles.clone(), idx.int())

            if self.no_sectors == 4:
                ph[0, :] = unwrap(4 * ph[0, :], np.pi, -1, 2 * np.pi) / 4
            else:
                ph[0, :] = unwrap(ph[0, :], np.pi, -1, 2 * np.pi)
            return Ew * torch.exp(1j * ph), ph

    def _bps_idx_torch(self, x, angles):
        """Blind phase search core algorithm using PyTorch."""
        EE = torch.unsqueeze(x, 1) * torch.exp(1j * angles)
        idx = torch.zeros(x.shape[0], dtype=torch.int, device=x.device)
        dist = torch.min(torch.abs(torch.unsqueeze(EE, 2) - self.symbols) ** 2, dim=2)[
            0
        ]
        # print("dist shape", dist.shape)
        # unfold = torch.nn.Unfold(0, 2 * self.N + 1, 3)
        dist_pad = torch.cat(
            (
                torch.zeros([self.N, dist.shape[1]], device=dist.device),
                dist,
                torch.zeros([self.N, dist.shape[1]], device=dist.device),
            ),
            0,
        )
        if self.avg_filter_type == "nonlinear":
            unfolded_dist = dist_pad.unfold(0, 2 * self.N + 1, 1)
            # print("unfolded dist shape", unfolded_dist.shape)

            mvg = self.nonlin_filter(unfolded_dist).squeeze()
            # print("nonlin_mvg shape", nonlin_mvg.shape)
        else:
            mvg = (
                torch.conv1d(
                    input=dist.transpose(0, 1).view(-1, 1, x.shape[0]),
                    weight=self.normalized_filter.view(1, 1, -1),
                    padding="same",
                )
                .squeeze()
                .transpose(0, 1)
            )

        # print("mvg shape", mvg.shape)
        # print("mvg shape", torch.sum(mvg, 1).shape)
        # input()
        # normalize mvg values
        mvg /= torch.unsqueeze(torch.sum(mvg, 1), 1)
        if self.diff is True:
            return mvg
        else:
            idx = torch.argmin(mvg, dim=1)
            return idx

    def select_angles_torch(self, angles, idx):
        """
        Perform selection on angles.

        :param angles: vector of angles
        :param idx: vector of indices
        :returns: selected angles
        """
        L = angles.shape[0]
        anglesn = torch.zeros(L, dtype=angles.dtype)
        if angles.shape[0] > 1:
            # omp parallel for
            for i in range(L):
                anglesn[i] = angles[i, idx[i]]
            return anglesn
        else:
            L = idx.shape[0]
            anglesn = torch.zeros(L, dtype=angles.dtype)
            anglesn = angles[0, idx.long()]
            return anglesn
