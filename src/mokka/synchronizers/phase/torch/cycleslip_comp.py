"""Module containing methods for cycle slip compensation."""
import torch


def calculate_squared_distance_symbols(rx_symbols, tx_symbols):
    """Calculate the squared distance between complex symbols."""
    return torch.abs(rx_symbols - tx_symbols) ** 2


class GenieCycleSlipComp(torch.nn.Module):
    """
    Genie-aided cycle slip compensation with median filtering.

    This class performs genie-aided cycle slip compensation while
    applying median filtering on the result. This allows to compensate
    cycle slips while leaving the white noise undisturbed.

    Otherwise one might end up overcorrecting white noise and outpeform
    the Shannon limit.
    """

    def __init__(self, num_sectors=4, median_len=10):
        """
        Initialize the genie-aided cycle slip compensation.

        :param num_sectors: Number of sectors where cycle slip can occur
        :param median_len: Length of the median filter to apply on the measure
        """
        super(GenieCycleSlipComp, self).__init__()
        self.num_sectors = torch.as_tensor(num_sectors)
        self.softmin = torch.nn.Softmin(dim=1)
        self.median_len = torch.as_tensor(median_len)

    def forward(self, rx_symbols, tx_symbols, measure_fcn):
        """
        Perform the cycle slip compensation.

        :param rx_symbols: Received symbols with cycle slips
        :param tx_symbols: Sent symbols without noise and cycle slips
        :param measure_fcn: Function handle which takes received symbols
                       and sent symbols and returns a loss.
        :returns: Compensated symbols
        """
        N = rx_symbols.shape[0]

        rotations = torch.exp(
            1j
            * torch.tensor(
                torch.arange(self.num_sectors) * 2 * torch.pi / self.num_sectors
            )
        )

        measure = torch.zeros((N, self.num_sectors), device=rx_symbols.device)
        for sector in torch.arange(self.num_sectors):
            rx_symbols_rotated = rx_symbols * rotations[sector]
            measure[:, sector] = measure_fcn(rx_symbols_rotated, tx_symbols)

        # Normalization of measure

        if True:
            ml2_norm = measure / torch.unsqueeze(torch.sum(measure, 1), 1)
            softdecision = torch.clamp(self.softmin(ml2_norm / 1e-2), min=1e-12)

            filtered_softdecision = torch.zeros(
                (N - 2 * self.median_len, self.num_sectors), device=rx_symbols.device
            )
            for sector in torch.arange(self.num_sectors, device=rx_symbols.device):
                filtered_softdecision[:, sector] = torch.median(
                    softdecision[:, sector].unfold(0, 2 * self.median_len + 1, 1), 1
                )[0]

            filtered_softdecision_norm = filtered_softdecision / torch.unsqueeze(
                torch.sum(filtered_softdecision, 1), 1
            )

            phase_rotation = torch.angle(
                torch.sum(filtered_softdecision_norm * rotations.unsqueeze(0), axis=1)
            )
            return rx_symbols[self.median_len : -self.median_len] * torch.exp(
                1j * phase_rotation
            )
        else:
            _, idx = torch.min(measure, dim=1)
            phase_rotation = torch.angle(torch.gather(rotations, dim=0, index=idx))
            return rx_symbols * torch.exp(1j * phase_rotation)
