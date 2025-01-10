"""Equalizer blocks after channel before decoder/demodulator - in PyTorch."""

import torch
import numpy as np

from ..functional.torch import convolve_overlap_save, convolve


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


class LinearFilter(torch.nn.Module):
    """Class implementing a SISO linear filter.

    Optionally with trainable filter_taps.
    """

    def __init__(
        self,
        filter_taps: torch.Tensor,
        trainable: bool = False,
        timedomain: bool = True,
    ):
        """
        Initialize LinearFilter with filter taps and define if it is trainable.

        :param filter_taps: Filter taps for the linear filter
        :param trainable: Set filter taps as :py:class:`torch.nn.Parameter`
        :param timedomain: Perform convolution in time domain
        """
        super(LinearFilter, self).__init__()
        if trainable:
            self.register_parameter("taps", torch.nn.Parameter(filter_taps))
        else:
            self.register_buffer("taps", filter_taps)

        if timedomain:
            self.convolve = convolve
        else:
            self.convolve = convolve_overlap_save

    def forward(self, y: torch.Tensor, mode: str = "valid"):
        """
        Perform filtering of signal.

        :param y: input signal
        """
        return self.convolve(y, self.taps, mode=mode)


class Butterfly2x2(torch.nn.Module):
    """
    Class implementing the 2x2 complex butterfly filter structure.

    This typically used in dual-polarization fiber-optical communications
    for equalization.
    """

    def __init__(self, taps=None, num_taps=None, trainable=False, timedomain=True):
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
            filter_taps = torch.zeros((4, num_taps), dtype=torch.complex64)
            filter_taps[0, num_taps // 2] = 1.0
            filter_taps[2, num_taps // 2] = 1.0
            self.num_taps = num_taps
        else:
            assert taps.dim() == 2
            assert taps.size()[0] == 4
            assert taps.dtype == torch.complex64
            filter_taps = taps
            self.num_taps = taps.size()[1]
        # We store h_xx, h_xy, h_yy, h_yx
        if trainable:
            self.register_parameter("taps", torch.nn.Parameter(filter_taps))
        elif trainable:
            self.register_buffer("taps", filter_taps)
        else:
            raise ValueError("trainable must be set to either True or False.")

        if timedomain:
            self.convolve = convolve
        else:
            self.convolve = convolve_overlap_save

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

        result_x = self.convolve(y[0, :], self.taps[0, :], mode=mode) + self.convolve(
            y[1, :], self.taps[1, :], mode=mode
        )
        result_y = self.convolve(y[1, :], self.taps[2, :], mode=mode) + self.convolve(
            y[0, :], self.taps[3, :], mode=mode
        )

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
        # assert y.dtype == torch.complex64

        h_abs_sq = self.taps.real**2 + self.taps.imag**2

        result_x = (
            self.convolve(y[0, :], h_abs_sq[0, :], mode=mode).real
            + self.convolve(y[1, :], h_abs_sq[1, :], mode=mode).real
        )
        result_y = (
            self.convolve(y[1, :], h_abs_sq[2, :], mode=mode).real
            + self.convolve(y[0, :], h_abs_sq[3, :], mode=mode).real
        )

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


def correct_start_polarization(
    signal,
    pilot_signal,
    correct_static_phase=False,
    correct_polarization=True,
    return_phase_pol=False,
):
    """
    Correlate the signal with a known pilot_signal.

    Return the signal aligned with the first sample of the pilot_signal.
    This method correlates both polarization with the pilot_signal to check
    for flipped polarization, e.g. after an equalizers.

    :param signal: Dual polarization complex input signal
    :param pilot_signal: Dual polarization complex pilot signal to search for
    :param correct_static_phase: Estimate phase shift from maximum correlation value
                                 and apply phase correction to the returned aligned
                                 signal
    :returns: Signal aligned with the first element in the pilot_signal
    """
    cross_corr = torch.stack(
        (
            convolve_overlap_save(
                signal[0, :],
                torch.flip(pilot_signal[0, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
            convolve_overlap_save(
                signal[1, :],
                torch.flip(pilot_signal[1, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
            convolve_overlap_save(
                signal[1, :],
                torch.flip(pilot_signal[0, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
            convolve_overlap_save(
                signal[0, :],
                torch.flip(pilot_signal[1, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
        ),
        dim=0,
    )
    max_values, time_offsets = torch.max(torch.abs(cross_corr), dim=1, keepdim=True)
    # Check if the two maximum values in regular polarization are flipped compare
    # the sum of the maximum values

    if not correct_polarization or (
        max_values[0] + max_values[1] > max_values[2] + max_values[3]
    ):
        if correct_static_phase:
            static_phase_shift = torch.angle(
                torch.gather(
                    cross_corr[(0, 1), :], dim=1, index=time_offsets[(0, 1), :]
                )
            )
        num_samples = signal.shape[1] - torch.maximum(time_offsets[0], time_offsets[1])
        aligned_signal = torch.stack(
            (
                signal[0, time_offsets[0] : time_offsets[0] + num_samples],
                signal[1, time_offsets[1] : time_offsets[1] + num_samples],
            ),
        )
        flip_pol = False
    else:
        # Flip polarizations
        if correct_static_phase:
            static_phase_shift = torch.angle(
                torch.gather(
                    cross_corr[(2, 3), :], dim=1, index=time_offsets[(2, 3), :]
                )
            )
        num_samples = signal.shape[1] - torch.maximum(time_offsets[2], time_offsets[3])
        aligned_signal = torch.stack(
            (
                signal[1, time_offsets[2] : time_offsets[2] + num_samples],
                signal[0, time_offsets[3] : time_offsets[3] + num_samples],
            ),
        )
        flip_pol = True
    if correct_static_phase:
        aligned_signal = aligned_signal * torch.exp(
            torch.tensor(-1j) * static_phase_shift
        )
    if not return_phase_pol:
        return aligned_signal
    else:
        return aligned_signal, static_phase_shift, flip_pol


def correct_start(signal, pilot_signal, correct_static_phase=False):
    """
    Correlate the signal with a known pilot_signal.

    Return the signal aligned with the first sample of the pilot_signal.
    Optionally correct static_phase_offset using the maximum of the correlation

    :param signal: Single polarization complex input signal
    :param pilot_signal: Single polarization complex pilot signal to search for
    :param correct_static_phase: Estimate phase shift from maximum correlation value
                                 and apply phase correction to the returned aligned
                                 signal
    :returns: Signal aligned with the first element in the pilot_signal
    """
    cross_corr = convolve_overlap_save(
        signal,
        torch.flip(pilot_signal[:].conj().resolve_conj(), dims=(0,)),
        "valid",
    )
    max_value, time_offset = torch.max(torch.abs(cross_corr), dim=0, keepdim=True)
    # Check if the two maximum values in regular polarization are flipped compare
    # the sum of the maximum values

    if correct_static_phase:
        static_phase_shift = torch.angle(max_value)
    aligned_signal = signal[time_offset:]
    if correct_static_phase:
        aligned_signal = aligned_signal * torch.exp(
            torch.tensor(-1j) * static_phase_shift
        )
    return aligned_signal


def toeplitz(c, r):
    """Adapted from https://stackoverflow.com/a/68899386 with modifications."""
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape, dtype=c.dtype).nonzero().T
    return vals[j - i].reshape(*shape)


def impulse_response_to_toeplitz(h):
    """Take an impulse response and convert into matrix form."""
    padding = torch.zeros((h.shape[0],), dtype=h.dtype)
    first_row = torch.cat((h[0].unsqueeze(0), padding))
    first_col = torch.cat((h, padding))
    H = toeplitz(first_col, first_row)
    return H


def MU_MMSE_inv(H, N0):
    """
    Find inverse of multiuser compound H matrix based on multiuser MMSE algorithm.

    :param H: multiuser compound H matrix with batch indices t,m,r
              and individual matrices of equal size RxS
    """
    # Sum over Tx antennas
    H_tot = torch.zeros(
        (H.shape[2] * H.shape[3], H.shape[2] * H.shape[3]), dtype=H.dtype
    )
    for tx_mat in H:
        # Sum over individual users
        for user_mat in tx_mat:
            # print("user_mat: ", user_mat.flatten(0, 1).shape)
            # Sum over transmit antenna per user
            # user_mat is this case is a column-matrix concatenating all
            # matrices of a particular user & antenna in row direction
            # We end up with a matrix of shape R N_R x S
            # We now have H_{t,m}
            H_tot += torch.matmul(
                user_mat.flatten(0, 1), user_mat.flatten(0, 1).conj().resolve_conj().T
            )
    # After the sum we need to invert this matrix
    H_tot += N0 * torch.eye(user_mat.shape[1] * user_mat.shape[0], dtype=H.dtype)
    H_tot_inv = H_tot.inverse()
    return H_tot_inv


def MU_MMSE(H, N0, tx_antenna=0, user=0):
    """
    Compute equalizer taps for given tx_antenna and user.

    Specify a compound H matrix, AWGN noise power a tx_antenna index and a user index
    and obtain the corresponding linear equalizer matrix.

    :param H: multiuser compound H matrix with batch indices t,m,r
              and individual matrices of equal size RxS
    """
    H_tot_inv = MU_MMSE_inv(H, N0)
    f_select = torch.matmul(H_tot_inv, H[tx_antenna, user].flatten(0, 1))
    return f_select


def SU_MMSE(H: torch.Tensor, N0: float):
    """
    Compute equalizer taps according to the single user MMSE algorithm.

    :param H: Channel impulse matrix
    :param N0: AWGN noise power
    """
    f = torch.matmul(
        (
            torch.matmul(H, H.T.conj().resolve_conj())
            + N0 * torch.eye(H.shape[0], dtype=H.dtype)
        ).inverse(),
        H,
    )
    return f


def unit_vector(tau: int, S: int):
    """
    One-hot vector to select symbols at delay tau.

    :param tau: Delay to generate unit_vector for
    :param S: Length of the vector
    :returns: unit vector with impulse at delay tau
    """
    e = torch.zeros((S,), dtype=torch.complex64)
    e[tau] = 1.0
    return e


def create_transmit_matrix(s_k: torch.Tensor, L: int):
    """
    Create transmit matrix with symbols s_k and channel length L.

    :param s_k: Vector of transmit symbols
    :param L: Maximum channel length
    """
    cols = s_k.shape[0] - (2 * L - 1)
    rows = 2 * L
    S_k = torch.as_strided(s_k[: cols * rows], (cols, rows), (1, 1)).flip(dims=(1,)).T
    return S_k


def pad_transmit_vector(s: torch.Tensor):
    """
    Add padding to the transmit vector.

    :param s: Transmit vector to be padded by 100 zeros on each side.
    """
    return torch.nn.functional.pad(s, (100, 100), "constant", 0)


def h2H(h: torch.Tensor):
    """
    Create compound H matrix from MIMO impulse responses.

    This compound H matrix consists of the expanded matrices in time domain using
    the Toeplitz form, then the matrices are stored accordingly in the compound matrix
    tensor.
    """
    H_compound = torch.zeros(
        (h.shape[0], h.shape[1], h.shape[2], h.shape[3] + 1, h.shape[3] * 2),
        dtype=h.dtype,
    )
    for txid, tx_ir in enumerate(h):
        for uid, user_ir in enumerate(tx_ir):
            for rxid, recv_ir in enumerate(user_ir):
                H = impulse_response_to_toeplitz(recv_ir).T
                H_compound[txid, uid, rxid, :, :] = H.clone()
    return H_compound


def h2f(h, N0):
    """
    Construct compound H and calculate F with MMSE/ZF.

    (depending on the setting of N0) and return the filter taps
    for f

    :param h: impulse response in time domain h_{t,m,r} for each user/transmit/receive
      combination
    :param N0: white noise at receiver
    """
    H_compound = torch.zeros(
        (h.shape[0], h.shape[1], h.shape[2], h.shape[3] + 1, h.shape[3] * 2),
        dtype=h.dtype,
    )
    for txid, tx_ir in enumerate(h):
        for uid, user_ir in enumerate(tx_ir):
            for rxid, recv_ir in enumerate(user_ir):
                H = impulse_response_to_toeplitz(recv_ir).T
                H_compound[txid, uid, rxid, :, :] = H.clone()

    H_tot_inv = MU_MMSE_inv(H_compound, N0)

    F = torch.zeros(
        (h.shape[0], h.shape[1], h.shape[2] * (h.shape[3] + 1)), dtype=h.dtype
    )
    F_split = torch.zeros(
        (h.shape[0], h.shape[1], h.shape[2], h.shape[3] + 1), dtype=h.dtype
    )
    for txid in range(h.shape[0]):
        for uid in range(h.shape[1]):
            F[txid, uid, :] = (
                torch.matmul(H_tot_inv, H_compound[txid, uid].flatten(0, 1))
                @ unit_vector(h.shape[3], 2 * h.shape[3])
                .to(torch.complex64)
                .unsqueeze(1)
            ).squeeze(-1)
            for rxid in range(h.shape[2]):
                f = F[
                    txid,
                    uid,
                    rxid * (h.shape[3] + 1) : (rxid + 1) * (h.shape[3] + 1),
                ]
                F_split[txid, uid, rxid, :] = f.view(1, 1, 1, -1).clone()
    return F_split


def find_start_offset(signal: torch.Tensor, pilot_signal: torch.Tensor):
    """
    Find the start offset using a pilot_signal with correlation.

    :param signal: received signal to find pilot sequence in
    :param pilot_signal: pilot sequence to search for
    """
    cross_corr = torch.stack(
        (
            convolve_overlap_save(
                signal[0, :],
                torch.flip(pilot_signal[0, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
            convolve_overlap_save(
                signal[1, :],
                torch.flip(pilot_signal[1, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
            convolve_overlap_save(
                signal[1, :],
                torch.flip(pilot_signal[0, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
            convolve_overlap_save(
                signal[0, :],
                torch.flip(pilot_signal[1, :].conj().resolve_conj(), dims=(0,)),
                "valid",
            ),
        ),
        dim=0,
    )
    max_values, time_offsets = torch.max(torch.abs(cross_corr), dim=1, keepdim=True)
    return time_offsets
