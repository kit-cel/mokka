"""PyTorch implementations of adaptive equalizers."""

from ..torch import Butterfly2x2
from ..torch import correct_start_polarization, correct_start
from ...functional.torch import convolve_overlap_save
from ..torch import h2f
import torch
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)


class CMA(torch.nn.Module):
    """Class to perform CMA equalization."""

    butterfly_filter: Butterfly2x2

    def __init__(
        self,
        R,
        sps,
        lr,
        butterfly_filter=None,
        filter_length=31,
        block_size=1,
        no_singularity=False,
        singularity_length=3000,
    ):
        """
        Initialize :py:class:`CMA`.

        :param R: constant modulus radius
        :param sps: samples per symbol
        :param lr: learning rate
        :param butterfly_filter: Optional
          :py:class:`mokka.equalizers.torch.Butterfly2x2` object
        :param filter_length: butterfly filter length
          (if object is not given)
        :param block_size: Number of symbols to process before
          updating the equalizer taps
        :param no_singularity: Initialize the x- and y-polarization to avoid
          singularity by decoding
          the signal from the same polarization twice.
        :param singularity_length: Delay for initialization with the
          no_singularity approach
        """
        super(CMA, self).__init__()
        self.register_buffer("R", torch.as_tensor(R))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("block_size", torch.as_tensor(block_size))
        if butterfly_filter is not None:
            self.butterfly_filter = butterfly_filter
            self.register_buffer(
                "filter_length", torch.as_tensor(butterfly_filter.taps.shape[1])
            )
        else:
            self.butterfly_filter = Butterfly2x2(num_taps=filter_length)
            self.butterfly_filter.taps[0, filter_length // 2] = 1.0
            self.butterfly_filter.taps[2, filter_length // 2] = 1.0
            self.register_buffer("filter_length", torch.as_tensor(filter_length))
        # Do some clever initalization, first only equalize x-pol and then enable y-pol
        self.register_buffer("no_singularity", torch.as_tensor(no_singularity))
        self.register_buffer("singularity_length", torch.as_tensor(singularity_length))
        if self.no_singularity:
            self.butterfly_filter.taps[2, :] = 0
            self.butterfly_filter.taps[3, :] = 0

    def reset(self):
        """Reset equalizer and butterfly filters."""
        self.butterfly_filter = Butterfly2x2(num_taps=self.filter_length.item())
        self.butterfly_filter.taps[0, self.filter_length.item() // 2] = 1.0
        self.butterfly_filter.taps[2, self.filter_length.item() // 2] = 1.0
        if self.no_singularity:
            self.butterfly_filter.taps[2, :] = 0
            self.butterfly_filter.taps[3, :] = 0

    def forward(self, y):
        """
        Perform CMA equalization on input signal y.

        :param y: input signal y
        """
        # Implement CMA "by hand"
        # Basically step through the signal advancing always +sps symbols
        # and filtering 2*filter_len samples which will give one output
        # sample with mode "valid"

        equalizer_length = self.butterfly_filter.taps.size()[1]
        num_samp = y.shape[1]
        e = torch.zeros(
            (2, (num_samp - equalizer_length) // self.sps), dtype=torch.float32
        )
        # logger.debug("CMA loop num: %s", reset_number)
        R = self.R  # 1.0
        lr = self.lr  # 1e-3
        out = torch.zeros(
            2, (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )
        # We try to put the symbol of interest in the center tap of the equalizer
        eq_offset = (equalizer_length - 1) // 2
        for i, k in enumerate(
            range(eq_offset, num_samp - 1 - eq_offset * 2, self.sps * self.block_size)
        ):
            if i % 20000 == 0 and i != 0:
                lr = lr / 2.0
            if i == self.singularity_length and self.no_singularity:
                self.butterfly_filter.taps[3, :] = -1 * torch.flip(
                    self.butterfly_filter.taps[1, :].conj().resolve_conj(), dims=(0,)
                )
                self.butterfly_filter.taps[2, :] = torch.flip(
                    self.butterfly_filter.taps[0, :].conj().resolve_conj(), dims=(0,)
                )
            in_index = torch.arange(k - equalizer_length, k)
            out_tmp = self.butterfly_filter(y[:, in_index], "valid")
            out[:, i] = out_tmp.squeeze()
            e[:, i] = R - torch.pow(torch.abs(out[:, i]), 2)
            self.butterfly_filter.taps = self.butterfly_filter.taps + (
                2
                * lr
                * torch.flip(
                    torch.stack(
                        (
                            e[0, i]
                            * out[0, i]
                            * y[0, in_index].conj().resolve_conj(),  # hxx <- e_x x_0 x*
                            e[0, i]
                            * out[0, i]
                            * y[1, in_index].conj().resolve_conj(),  # hxy <- e_x x_0 y*
                            e[1, i]
                            * out[1, i]
                            * y[1, in_index].conj().resolve_conj(),  # hyy <- e_y y_0 y*
                            e[1, i]
                            * out[1, i]
                            * y[0, in_index].conj().resolve_conj(),  # hyx <- e_y y_0 x*
                        ),
                        dim=0,
                    ),
                    dims=(1,),
                )
            )
        self.out_e = e.detach().clone()
        return out

    def get_error_signal(self):
        """Return error signal used to adapt equalizer taps."""
        return self.out_e


def ELBO_DP(
    y,
    q,
    sps,
    constellation_symbols,
    butterfly_filter,
    p_constellation=None,
    IQ_separate=False,
):
    """
    Calculate dual-pol. ELBO loss for arbitrary complex constellations.

    Instead of splitting into in-phase and quadrature we can just
    the whole thing.
    This implements the dual-polarization case.
    """
    # Input is a sequence y of length N
    N = y.shape[1]
    # Now we have two polarizations in the first dimension
    # We assume the same transmit constellation for both, calculating
    # q needs to be shaped 2 x N x M  -> for each observation on each polarization we
    # have M q-values and we have M constellation symbols
    L = butterfly_filter.taps.shape[1]
    L_offset = (L - 1) // 2
    if p_constellation is None:
        p_constellation = (
            torch.ones_like(constellation_symbols) / constellation_symbols.shape[0]
        )

    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2
    E_Q_x = torch.zeros(2, N, device=q.device, dtype=torch.complex64)
    E_Q_x_abssq = torch.zeros(2, N, device=q.device, dtype=torch.float32)
    if IQ_separate:
        num_lev = constellation_symbols.shape[0]
        E_Q_x[:, ::sps] = torch.complex(
            torch.sum(
                q[:, :, :num_lev] * constellation_symbols.unsqueeze(0).unsqueeze(0),
                dim=-1,
            ),
            torch.sum(
                q[:, :, num_lev:] * constellation_symbols.unsqueeze(0).unsqueeze(0),
                dim=-1,
            ),
        )
        E_Q_x_abssq[:, ::sps] = torch.add(  # Precompute E_Q{|x|^2}
            torch.sum(
                q[:, :, :num_lev]
                * (constellation_symbols**2).unsqueeze(0).unsqueeze(0),
                dim=-1,
            ),
            torch.sum(
                q[:, :, num_lev:]
                * (constellation_symbols**2).unsqueeze(0).unsqueeze(0),
                dim=-1,
            ),
        )
        p_constellation = p_constellation.repeat(2)
    else:
        E_Q_x[:, ::sps] = torch.sum(
            q * constellation_symbols.unsqueeze(0).unsqueeze(0), axis=-1
        )
        E_Q_x_abssq[:, ::sps] = torch.sum(
            q
            * (constellation_symbols.real**2 + constellation_symbols.imag**2)
            .unsqueeze(0)
            .unsqueeze(0),
            axis=-1,
        )

    # Term A - sum all the things, but spare the first dimension,
    # since the two polarizations are sorta independent
    bias = 1e-14
    A = torch.sum(
        q[:, L_offset:-L_offset, :]
        * torch.log(
            (q[:, L_offset:-L_offset, :] / p_constellation.unsqueeze(0).unsqueeze(0))
            + bias
        ),
        dim=(1, 2),
    )

    # Precompute h \ast E_Q{x}
    # Due to definition that we assume the symbol is at the center tap
    # we remove (filter_length - 1)//2 at start and end
    h_conv_E_Q_x = butterfly_filter(E_Q_x, mode="valid")
    # Limit the length of y to the "computable space" because y depends
    # on more past values than given
    # We try to generate the received symbol sequence
    # with the estimated symbol sequence
    C = torch.sum(
        y[:, L_offset:-L_offset].real ** 2 + y[:, L_offset:-L_offset].imag ** 2, axis=1
    )
    C -= 2 * torch.sum(y[:, L_offset:-L_offset].conj() * h_conv_E_Q_x, axis=1).real
    C += torch.sum(
        h_conv_E_Q_x.real**2
        + h_conv_E_Q_x.imag**2
        + butterfly_filter.forward_abssquared(
            (E_Q_x_abssq - torch.abs(E_Q_x) ** 2).to(torch.float32), mode="valid"
        ),
        axis=1,
    )

    # We compute B without constants
    # B_tilde = -N * torch.log(C)
    # bias1 = 1e-8
    loss = torch.sum(A) + (N - L + 1) * torch.sum(torch.log(C + bias))  # / sps
    var = C / (N - L + 1)  # * sps #N
    return loss, var


class VAE_LE_DP(torch.nn.Module):
    """
    Adaptive Equalizer based on the variational autoencoder principle with a linear equalizer.

    This code is based on the work presented in [1].

    [1] V. Lauinger, F. Buchali, and L. Schmalen, ‘Blind equalization and channel estimation in coherent optical communications using variational autoencoders’,
    IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2529–2539, Sep. 2022, doi: 10.1109/JSAC.2022.3191346.
    """  # noqa

    def __init__(
        self,
        num_taps_forward,
        num_taps_backward,
        demapper,
        sps,
        block_size=200,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        var_from_estimate=False,
        num_block_train=None,
    ):
        """
        Initialize :py:class:`VAE_LE_DP`.

        This VAE equalizer is implemented with a butterfly linear equalizer in the
        forward path and a butterfly linear equalizer in the backward pass.
        Therefore, it is limited to correct impairments of linear channels.

        :param num_taps_forward: number of equalizer taps
        :param num_taps_backward: number of channel taps
        :param demapper: mokka demapper object to perform complex symbol demapping
        :param sps: samples per symbol
        :param block_size: number of symbols per block - defines the update rate
          of the equalizer
        :param lr: learning rate for the adam algorithm
        :param requires_q:  return q-values in forward call
        :param IQ_separate: process I and Q separately - requires a demapper
          which performs demapping on real values
          and a bit-mapping which is equal on I and Q.
        :param var_from_estimate: Update the variance in the demapper from
          the SNR estimate of the output
        :param num_block_train: Number of blocks to train the equalizer before
          switching to non-training equalization mode (for static channels only)
        """
        super(VAE_LE_DP, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_forward))
        self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_backward))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("var_from_estimate", torch.as_tensor(var_from_estimate))
        self.butterfly_forward = Butterfly2x2(
            num_taps=num_taps_forward, trainable=True, timedomain=True
        )
        self.butterfly_backward = Butterfly2x2(
            num_taps=num_taps_backward, trainable=True, timedomain=True
        )
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})

        self.optimizer_var = torch.optim.Adam(
            [self.demapper.noise_sigma],
            lr=0.5,  # 0.5e-2,
        )
        self.num_block_train = num_block_train

    def reset(self):
        """Reset :py:class:`VAE_LE_DP` object."""
        self.lr = self.start_lr.clone()
        self.butterfly_forward = Butterfly2x2(
            num_taps=self.num_taps_forward.item(),
            trainable=True,
            timedomain=True,
        ).to(self.butterfly_forward.taps.device)
        self.butterfly_backward = Butterfly2x2(
            num_taps=self.num_taps_backward.item(),
            trainable=True,
            timedomain=True,
        ).to(self.butterfly_forward.taps.device)
        self.optimizer = torch.optim.SGD(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})

    @torch.enable_grad()
    def forward(self, y):
        """
        Peform equalization of input signal y.

        :param y: Complex input signal
        """
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size
        # (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out
        # the first and last block

        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        # Back-off one block-size + filter_overlap from end to avoid overrunning
        for i, k in enumerate(
            range(
                index_padding,
                num_samps - index_padding - self.sps * self.block_size,
                self.sps * self.block_size,
            )
        ):
            # if i % (20000//self.block_size) == 0 and i != 0:
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add
            # (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index], "valid")

            # We downsample so we will have
            # floor(((sps * block_size - num_taps + 1) / sps)
            # = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            if self.IQ_separate:
                q_hat = torch.cat(
                    (
                        torch.cat(
                            (
                                self.demapper(y_symb[0, :].real).unsqueeze(0),
                                self.demapper(y_symb[0, :].imag).unsqueeze(0),
                            ),
                            axis=-1,
                        ),
                        torch.cat(
                            (
                                self.demapper(y_symb[1, :].real).unsqueeze(0),
                                self.demapper(y_symb[1, :].imag).unsqueeze(0),
                            ),
                            axis=-1,
                        ),
                    ),
                    axis=0,
                )
            else:
                q_hat = torch.cat(
                    (
                        self.demapper(y_symb[0, :]).unsqueeze(0),
                        self.demapper(y_symb[1, :]).unsqueeze(0),
                    )
                )
            # We calculate the loss with less symbols, since the forward operation
            # with "valid" is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                (self.butterfly_forward.num_taps - 1)
                // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            ]
            loss, var = ELBO_DP(
                # loss, var = ELBO_DP(
                y[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.butterfly_backward,
                p_constellation=self.demapper.p_symbols,
                IQ_separate=self.IQ_separate,
            )

            # logger.info("Iteration: %s/%s, VAE loss: %s", i+1,
            # ((num_samps - index_padding - self.sps * self.block_size)
            # // (self.sps * self.block_size)).item(), loss.item())

            if self.num_block_train is None or (self.num_block_train > i):
                # print("noise_sigma: ", self.demapper.noise_sigma)
                loss.backward()
                self.optimizer.step()
                # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            if self.var_from_estimate:
                self.demapper.noise_sigma = torch.clamp(
                    torch.sqrt(torch.mean(var.detach().clone()) / 2),
                    min=torch.tensor(0.05, requires_grad=False, device=q_hat.device),
                    max=2 * self.demapper.noise_sigma.detach().clone(),
                )

            output_symbols = y_symb[
                :, : self.block_size
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])

            output_q = q_hat[:, : self.block_size, :]
            out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps
        #  // 2 :])

        if self.requires_q:
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(torch.cat(out, axis=1), torch.cat(out_q, axis=1), var, loss)
        return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        """
        Update learning rate of VAE equalizer.

        :param new_lr: new value of learning rate to be set
        """
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_var):
        """
        Update variance of demapper.

        :param new_var: new value of variance to be set
        """
        self.demapper.noise_sigma = new_var


def update_adaptive(y_hat_sym, pilot_seq, regression_seq, idx, length, sps):
    """
    Calculate update signal to be used to update adaptive equalizer taps.

    :param y_hat_sym: Estimated receive symbol sequence
    :param pilot_seq: Known pilot symbol sequence
    :param regression_seq: Regression sequence to be applied to the equalizer taps
    :param idx: symbol index to use for calculation of the error signal
    :param length: Not used in this function, just for API compatibility reasons
    :param sps: Not used in this function, just for API compatibility reasons
    """
    e_k = pilot_seq[idx] - y_hat_sym[idx]
    # idx_up = idx * sps

    # print("Using regression sequence at indices: ", idx_up, " to ", idx_up + length)
    result = e_k * torch.flip(regression_seq.conj().resolve_conj(), dims=(0,))
    return result, e_k


class PilotAEQ_DP(torch.nn.Module):
    """
    Perform pilot-based adaptive equalization.

    This class performs equalization on a dual polarization signal with a known dual
    polarization pilot sequence. The equalization is performed either with the LMS
    method, ZF method or a novel LMSZF method which combines the regression vectors
    of LMS and ZF to improve stability and channel estimation properties.
    """

    def __init__(
        self,
        sps,
        lr,
        pilot_sequence,
        pilot_sequence_up,
        butterfly_filter=None,
        filter_length=31,
        method="LMS",
        block_size=1,
        adaptive_lr=False,
        adaptive_scale=0.1,
        preeq_method=None,
        preeq_offset=3000,
        preeq_lradjust=1.0,
        lmszf_weight=0.5,
    ):
        """
        Initialize :py:class:`PilotAEQ_DP`.

        :param sps: samples per symbol
        :param lr: learning rate to update adaptive equalizer taps
        :param pilot_sequence: Known dual polarization pilot sequence
        :param pilot_sequence_up: Upsampled dual polarization pilot sequence
        :param butterfly_filter: :py:class:`mokka.equalizers.torch.Butterfly2x2` object
        :param filter_length: If a butterfly_filter argument is not provided the filter
          length to initialize the butterfly filter.
        :param method: adaptive update method for the equalizer filter taps
        :param block_size: number of symbols to process before each update step
        :param adaptive_lr: Adapt learning rate during simulation
        :param preeq_method: Use a different method to perform a first-stage
          equalization
        :param preeq_offset: Length of first-stage equalization
        :param preeq_lradjust: Change learning rate by this factor for first-stage
          equalization
        :param lmszf_weight: if LMSZF is used as equalization method the weight between
          ZF and LMS update algorithms.
        """
        super(PilotAEQ_DP, self).__init__()
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("pilot_sequence", torch.as_tensor(pilot_sequence))
        self.register_buffer("pilot_sequence_up", torch.as_tensor(pilot_sequence_up))
        if butterfly_filter is not None:
            self.butterfly_filter = butterfly_filter
            self.register_buffer(
                "filter_length", torch.as_tensor(butterfly_filter.taps.shape[1])
            )
        else:
            self.butterfly_filter = Butterfly2x2(num_taps=filter_length)
            self.butterfly_filter.taps[0, filter_length // 2] = 1.0
            self.butterfly_filter.taps[2, filter_length // 2] = 1.0
            self.register_buffer("filter_length", torch.as_tensor(filter_length))
        self.method = method
        self.update = update_adaptive
        self.block_size = block_size
        self.adaptive_lr = adaptive_lr
        self.adaptive_scale = adaptive_scale
        self.preeq_method = preeq_method
        self.preeq_offset = preeq_offset
        self.preeq_lradjust = preeq_lradjust
        self.lmszf_weight = torch.as_tensor(lmszf_weight)

    def reset(self):
        """Reset :py:class:`PilotAEQ_DP` object."""
        self.butterfly_filter = Butterfly2x2(num_taps=self.filter_length.item())
        self.butterfly_filter.taps[0, self.filter_length.item() // 2] = 1.0
        self.butterfly_filter.taps[2, self.filter_length.item() // 2] = 1.0

    def forward(self, y):
        """
        Equalize input signal y.

        :param y: Complex receive signal y
        """
        # y_cut is perfectly aligned with pilot_sequence_up (after cross
        # correlation & using peak)
        # The adaptive filter should be able to correct polarization flip on its own
        y_cut = correct_start_polarization(
            y, self.pilot_sequence_up[:, : y.shape[1]], correct_polarization=False
        )

        equalizer_length = self.butterfly_filter.taps.size()[1]
        eq_offset = ((equalizer_length - 1) // 2) // self.sps
        num_samp = y_cut.shape[1]
        u = torch.zeros(
            (
                4,
                (self.pilot_sequence.shape[1] - 2 * eq_offset),  # // self.sps,
                equalizer_length,
            ),
            dtype=torch.complex64,
        )
        filter_taps = torch.zeros(
            (
                4,
                (self.pilot_sequence.shape[1] - 2 * eq_offset),  # // self.sps,
                equalizer_length,
            ),
            dtype=torch.complex64,
        )
        e = torch.zeros(
            (4, (num_samp - equalizer_length) // self.sps),
            dtype=torch.complex64,
        )
        peak_distortion = torch.zeros(
            (4, (num_samp - equalizer_length) // self.sps),
            dtype=torch.complex64,
        )
        lr = self.lr  # 1e-3
        out = torch.zeros(
            2, (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )

        if self.preeq_method is None:
            eq_method = self.method
        else:
            eq_method = self.preeq_method

        if eq_method in ("LMS"):
            regression_seq = y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
        elif eq_method in ("ZF", "ZFadv"):
            regression_seq = self.pilot_sequence_up.clone().conj().resolve_conj()
        elif eq_method in ("LMSZF"):
            regression_seq = (
                torch.sqrt(torch.as_tensor(self.lmszf_weight))
                * y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
                + torch.sqrt(1.0 - torch.as_tensor(self.lmszf_weight))
                * self.pilot_sequence_up.clone().conj().resolve_conj()
            )
        for i, k in enumerate(range(equalizer_length, num_samp - 1, self.sps)):
            # i counts the actual loop number
            # k starts at equalizer_length
            # the first symbol we produce in the pilot sequence is at index
            # (equalizer_length-1)//2
            # we need the symbols around the center symbol for equalization
            in_index = torch.arange(k - equalizer_length, k)
            out_tmp = self.butterfly_filter(y_cut[:, in_index], "valid")
            out[:, i] = out_tmp.squeeze()
            # Out will contain samples starting with
            # ((filter_length-1)//2)//sps
            if i + 2 * eq_offset + 1 < self.pilot_sequence.shape[1]:
                if self.preeq_method is not None and i == self.preeq_offset:
                    eq_method = self.method
                    if self.method == "LMS":
                        regression_seq = y_cut.clone()
                    elif self.method in ("ZF", "ZFadv"):
                        regression_seq = (
                            self.pilot_sequence_up.clone().conj().resolve_conj()
                        )
                    elif self.method == "LMSZF":
                        regression_seq = (
                            torch.sqrt(torch.as_tensor(self.lmszf_weight))
                            * y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
                            + torch.sqrt(1.0 - torch.as_tensor(self.lmszf_weight))
                            * self.pilot_sequence_up.clone().conj().resolve_conj()
                        )
                if i == self.preeq_offset:
                    lr = lr * self.preeq_lradjust

                if eq_method == "ZFadv":
                    # Update regression seq by calculating h from f and
                    # estimating \hat{y}
                    # We can use the same function as in the forward pass
                    f = torch.stack(
                        (
                            torch.stack(
                                (
                                    self.butterfly_filter.taps[0, :],
                                    self.butterfly_filter.taps[1, :],
                                )
                            ),
                            torch.stack(
                                (
                                    self.butterfly_filter.taps[3, :],
                                    self.butterfly_filter.taps[2, :],
                                )
                            ),
                        )
                    )
                    h = h2f(f.unsqueeze(0), 0.0).squeeze()
                    h_filt = Butterfly2x2(
                        torch.stack((h[0, 0, :], h[0, 1, :], h[1, 1, :], h[1, 0, :]))
                        .conj()
                        .resolve_conj()
                    )

                    # We need equalizer_length samples in regression_seq
                    h_filt_seq = torch.nn.functional.pad(
                        self.pilot_sequence_up,
                        (h_filt.taps.shape[1] // 2 - 1, h_filt.taps.shape[1] // 2 - 1),
                    )
                    update_seq = h_filt(
                        h_filt_seq[
                            :,
                            i * self.sps : i * self.sps
                            + equalizer_length
                            + h_filt.taps.shape[1]
                            - 1,
                        ]
                    )
                    regression_seq[
                        :, i * self.sps : i * self.sps + equalizer_length
                    ] = update_seq

                u[0, i, :], e00 = self.update(
                    out[0, :],
                    self.pilot_sequence[0, eq_offset:],
                    regression_seq[0, in_index],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[1, i, :], e01 = self.update(
                    out[0, :],
                    self.pilot_sequence[0, eq_offset:],
                    regression_seq[1, in_index],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[2, i, :], e11 = self.update(
                    out[1, :],
                    self.pilot_sequence[1, eq_offset:],
                    regression_seq[1, in_index],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[3, i, :], e10 = self.update(
                    out[1, :],
                    self.pilot_sequence[1, eq_offset:],
                    regression_seq[0, in_index],
                    i,
                    equalizer_length,
                    self.sps,
                )
                if self.adaptive_lr:
                    # For LMS according to Rupp 2011 this stepsize ensures the
                    # stability/robustness
                    lr = (
                        self.adaptive_scale
                        * 2
                        / torch.sum(
                            (
                                torch.abs(
                                    regression_seq[
                                        :,
                                        i * self.sps : i * self.sps + equalizer_length,
                                    ]
                                )
                                ** 2
                            ),
                            dim=1,
                        )
                    )
                    lr = torch.tensor([lr[0], lr[1], lr[1], lr[0]]).unsqueeze(1)
                    # logger.info("lr: %s", lr.numpy())
                if i > 0 and i % self.block_size == 0:
                    self.butterfly_filter.taps = self.butterfly_filter.taps + (
                        2
                        * lr
                        * torch.mean(u[:, i - self.block_size : i, :], dim=1).squeeze()
                    )
                filter_taps[:, i, :] = self.butterfly_filter.taps.clone()
                e[:, i] = torch.stack((e00, e01, e11, e10))
                peak_distortion[:, i] = torch.mean(
                    e[:, i].unsqueeze(1)
                    * self.pilot_sequence[(0, 0, 1, 1), i : i + 2 * eq_offset]
                    .conj()
                    .resolve_conj(),
                    dim=1,
                )
        # Add some padding in the start
        out = torch.cat((torch.zeros((2, 100), dtype=torch.complex64), out), dim=1)
        self.u = u
        self.e = e
        self.peak_distortion = peak_distortion
        self.filter_taps = filter_taps
        return out


class PilotAEQ_SP(torch.nn.Module):
    """
    Perform pilot-based adaptive equalization (QPSK).

    This class performs the adaptive equalization for a single polarization.
    """

    def __init__(
        self,
        sps,
        lr,
        pilot_sequence,
        pilot_sequence_up,
        filter_length=31,
        method="LMS",
    ):
        """
        Initialize :py:class:`PilotAEQ_SP`.

        Pilot-based adaptive equalizer for the single polarization case.

        :param sps: samples per symbol
        :param lr: learning rate of adaptive equalizer update
        :param pilot_sequence: known transmit pilot sequence
        :param pilot_sequence_up: upsampled known transmit pilot sequence
        """
        super(PilotAEQ_SP, self).__init__()
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("pilot_sequence", torch.as_tensor(pilot_sequence))
        self.register_buffer("pilot_sequence_up", torch.as_tensor(pilot_sequence_up))
        self.register_buffer(
            "taps", torch.zeros((filter_length,), dtype=torch.complex64)
        )
        self.taps[filter_length // 2] = 1.0
        self.update = update_adaptive
        self.method = method

    def reset(self):
        """Reset :py:class:`PilotAEQ_SP`."""
        self.taps.zero_()
        self.taps[self.taps.size()[0] // 2] = 1.0

    def forward(self, y):
        """
        Equalize a single polarization signal.

        :param y: complex single polarization received signal
        """
        y_cut = correct_start(y, self.pilot_sequence_up)

        equalizer_length = self.taps.size()[0]
        eq_offset = ((equalizer_length - 1) // 2) // self.sps
        num_samp = y_cut.shape[0]
        u = torch.zeros(
            ((num_samp - equalizer_length) // self.sps, equalizer_length),
            dtype=torch.complex64,
        )
        e = torch.zeros(
            ((num_samp - equalizer_length) // self.sps),
            dtype=torch.complex64,
        )
        peak_distortion = torch.zeros(
            ((num_samp - equalizer_length) // self.sps),
            dtype=torch.complex64,
        )
        lr = self.lr  # 1e-3
        out = torch.zeros(
            (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )
        if self.method in ("LMS", "LMS_ZF"):
            regression_seq = y_cut
        elif self.method in ("ZF"):
            regression_seq = self.pilot_sequence_up

        for i, k in enumerate(range(equalizer_length, num_samp - 1, self.sps)):
            in_index = torch.arange(k - equalizer_length, k)
            out_tmp = convolve_overlap_save(y_cut[in_index], self.taps, "valid")
            out[i] = out_tmp.squeeze()
            # Out will contain samples starting with
            # ((filter_length-1)//2)//sps
            if i * self.sps + 2 * eq_offset < self.pilot_sequence.shape[0]:
                u[i, :], e[i] = self.update(
                    out,
                    self.pilot_sequence[eq_offset:],
                    regression_seq,
                    i,
                    equalizer_length,
                    self.sps,
                )
            self.taps = self.taps + (2 * lr * u[i, :].squeeze())
            peak_distortion[i] = torch.mean(
                e[i] * self.pilot_sequence[i : i + 2 * eq_offset].conj().resolve_conj()
            )
        # Add some padding in the start
        self.u = u.clone()
        self.e = e.clone()
        self.peak_distortion = peak_distortion.clone()
        out = torch.cat((torch.zeros((100,), dtype=torch.complex64), out), dim=0)
        return out


class AEQ_SP(torch.nn.Module):
    """Class to perform CMA equalization for a single polarization signal."""

    def __init__(
        self,
        R,
        sps,
        lr,
        taps=None,
        filter_length=31,
        block_size=1,
        no_singularity=False,
    ):
        """
        Initialize :py:class:`AEQ_SP`.

        :param R: average radio/kurtosis of constellation
        :param sps: samples per symbol
        :param lr: learning rate of adaptive update algorithm
        :param taps: Initial equalizer taps
        :param filter_length: length of equalizer if not provided as object
        :param block_size: number of symbols per update step
        :param no_singularity: Not used for the single polarization case
        """
        super(AEQ_SP, self).__init__()
        self.register_buffer("R", torch.as_tensor(R))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer(
            "taps",
            torch.zeros(
                filter_length,
                dtype=torch.complex64,
            ),
        )
        self.taps[self.taps.shape[0] // 2] = 1.0
        # Do some clever initalization, first only equalize x-pol and then enable y-pol

    def reset(self):
        """Reset :py:class:`AEQ_SP`."""
        self.taps.zero_()
        self.taps[self.taps.shape[0] // 2] = 1.0

    def forward(self, y):
        """Perform adaptive equalization."""
        # Implement CMA "by hand"
        # Basically step through the signal advancing always +sps symbols
        # and filtering 2*filter_len samples which will give one output sample with
        # mode "valid"

        equalizer_length = self.taps.shape[0]
        num_samp = y.shape[0]
        e = torch.zeros(
            ((num_samp - equalizer_length) // self.sps), dtype=torch.float32
        )
        # logger.debug("CMA loop num: %s", reset_number)
        R = self.R  # 1.0
        lr = self.lr  # 1e-3
        out = torch.zeros(
            (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )
        # We try to put the symbol of interest in the center tap of the equalizer
        eq_offset = (equalizer_length - 1) // 2
        for i, k in enumerate(
            range(eq_offset, num_samp - 1 - eq_offset * 2, self.sps * self.block_size)
        ):
            if i % 20000 == 0 and i != 0:
                lr = lr / 2.0
            in_index = torch.arange(k - equalizer_length, k)
            out_tmp = convolve_overlap_save(y[in_index], self.taps, "valid")
            out[i] = out_tmp.squeeze()
            e[i] = R - torch.pow(torch.abs(out[i]), 2)
            self.taps = self.taps + 2 * lr * torch.flip(
                e[i] * out[i] * y[in_index].conj().resolve_conj(),  # hxx <- e_x x_0 x*
                dims=(0,),
            )
        self.out_e = e.detach().clone()
        return out

    def get_error_signal(self):
        """Extract the error signal."""
        return self.out_e
