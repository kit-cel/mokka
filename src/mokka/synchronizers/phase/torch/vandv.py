"""This module contains implementations of the Viterbi & Viterbi phase synchronizers."""

import torch
from ....functional.torch import unwrap, convolve


class ViterbiViterbi(torch.nn.Module):
    """
    Implements the classical Viterbi & Viterbi phase sync.

    If partition is set to a positive integer the algorithm
    is applied with partitioning based on the amplitudes.
    """

    def __init__(
        self,
        window_length=50,
        symmetry=4,
        partition=0,
        partition_window=50,
        partition_activation="ReLU",
        partition_max_amp=1.5,
    ):
        """
        Initialize Viterbi & Viterbi synchronizer.

        :param window_length: length of the window for averaging
        :param symmetry: rotational symmetry of the constellation
        :param partition: if partitioning should be used
        :param partition_window: length of the window for partitioning
        :param partition_activation: activation function for partitioning
        :param partition_max_amp: maximum amplitude for partitioning
        """
        super(ViterbiViterbi, self).__init__()
        self.register_buffer("partition", torch.tensor(partition))
        if self.partition.item():
            # Split the partition into multiple non-overlapping sections
            bounds = torch.linspace(
                0, partition_max_amp, steps=self.partition.item() + 1
            )
            self.selection_weights = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.tensor((bounds[w], bounds[w + 1])))
                    for w in range(partition)
                ]
            )
            self.selection_scales = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float))
                    for _ in range(partition)
                ]
            )
            if partition_activation == "heaviside":
                self.partition_activation = lambda x: torch.heaviside(
                    x,
                    torch.tensor(
                        [
                            0.0,
                        ]
                    ),
                )
            else:
                self.partition_activation = getattr(torch.nn, partition_activation)()
        self.register_buffer(
            "window_length", torch.tensor(window_length, dtype=torch.int)
        )
        self.register_buffer(
            "partition_window", torch.tensor(partition_window, dtype=torch.int)
        )
        self.register_buffer("symmetry", torch.tensor(symmetry))

    def calc_partition(self, y):
        """
        Calculate partitioning for a given input sequence of symbols.

        :param y: Input sequence
        """
        time_axis = torch.arange(0, len(y), 1, device=y.device)
        if not self.partition:
            return (y, time_axis)
        y_cpe_2 = torch.zeros_like(y, device=y.device)
        for r in range(self.partition.item()):
            scale = self.selection_scales[r]
            y_cpe_2 = y_cpe_2 + (
                self.partition_activation(
                    scale * (self.selection_weights[r][1] - torch.abs(y))
                )
                * self.partition_activation(
                    scale * (torch.abs(y) - self.selection_weights[r][0])
                )
                * y
                / torch.abs(y)
            )
        time_axis = torch.nonzero(torch.abs(y_cpe_2) > 1e-5)
        return (y_cpe_2, time_axis)

    def forward(self, x):
        """
        Perform phase synchronization on complex input sequence.

        :param x: complex input sequence
        :returns: synchronized symbols sequence
        """
        x = torch.squeeze(x)
        x = x * torch.exp(torch.tensor(1j) * torch.pi / self.symmetry.item())
        y_cpe_2, time_axis_2 = self.calc_partition(x)
        y_sym = torch.pow(y_cpe_2, self.symmetry.item())
        time_axis_2 = torch.nonzero(torch.abs(y_sym) > 1e-5)
        y_sym_pre = y_sym.clone()
        y_sym[time_axis_2] = y_sym[time_axis_2] / torch.abs(
            y_sym[time_axis_2]
        )  # Normalize non-zero symbols
        #        y_sym[time_axis_2] = torch.exp(1j * torch.angle(y_sym[time_axis_2]))
        if torch.any(torch.isnan(y_sym)):
            nan_idx = torch.nonzero(torch.isnan(y_sym))
            print("y_sym: ", y_sym[nan_idx])
            print("y_cpe_2: ", y_cpe_2[nan_idx])
            print("y_sym_pre: ", y_sym_pre[nan_idx])
        y_sym_pad = torch.concat(
            (
                torch.zeros(self.window_length.item() // 2, device=x.device),
                y_sym,
                torch.zeros(self.window_length.item() // 2, device=x.device),
            )
        )
        y_sym_sum = torch.cumsum(y_sym_pad, 0)

        angle = torch.angle(
            y_sym_sum[self.window_length.item() :]
            - y_sym_sum[: -self.window_length.item()]
        )

        phase_est = (1.0 / self.symmetry.item()) * unwrap(angle)

        if self.partition.item():
            phase_comp = torch.zeros_like(x, dtype=torch.float32)
            phase_comp[time_axis_2] = phase_est[time_axis_2]
            phase_comp = phase_comp.flatten()
            filter_kernel = torch.ones(
                (self.partition_window.item(),),
                dtype=torch.complex64,
                device=phase_comp.device,
            )
            phase_comp_val = torch.zeros_like(phase_comp, dtype=torch.float32)
            phase_comp_val[time_axis_2] = 1.0
            phase_comp_norm = convolve(
                phase_comp_val.type(torch.float32),
                filter_kernel.type(torch.float32),
                mode="same",
            )
            phase_est = (
                convolve(phase_comp.type(torch.complex64), filter_kernel, mode="same")
                / phase_comp_norm
            )

        rx_corrected = (
            x
            * torch.exp(-1j * phase_est)
            * torch.exp(torch.tensor(-1j) * torch.pi / self.symmetry.item())
        )
        return rx_corrected
