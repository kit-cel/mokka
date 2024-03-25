"""
PyTorch implementations of adaptive equalizers.

"""
from ..torch import Butterfly2x2
from ..torch import correct_start_polarization
from ...functional.torch import convolve_overlap_save
import torch


class CMA(torch.nn.Module):
    """
    Perform CMA equalization
    """

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
    ):
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
        if self.no_singularity:
            self.butterfly_filter.taps[2, :] = 0
            self.butterfly_filter.taps[3, :] = 0

    def reset(self):
        self.butterfly_filter = Butterfly2x2(num_taps=self.filter_length.item())
        self.butterfly_filter.taps[0, self.filter_length.item() // 2] = 1.0
        self.butterfly_filter.taps[2, self.filter_length.item() // 2] = 1.0
        if self.no_singularity:
            self.butterfly_filter.taps[2, :] = 0
            self.butterfly_filter.taps[3, :] = 0

    def forward(self, y):
        # Implement CMA "by hand"
        # Basically step through the signal advancing always +sps symbols
        # and filtering 2*filter_len samples which will give one output sample with mode "valid"

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
        eq_offset = (
            equalizer_length - 1
        ) // 2  # We try to put the symbol of interest in the center tap of the equalizer
        for i, k in enumerate(
            range(eq_offset, num_samp - 1 - eq_offset * 2, self.sps * self.block_size)
        ):
            if i % 20000 == 0 and i != 0:
                lr = lr / 2.0
            if i == 3000 and self.no_singularity:
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
        return self.out_e


##############################################################################################
########################### Variational Autoencoer based Equalizer ###########################
##############################################################################################


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
    # q needs to be shaped 2 x N x M  -> for each observation on each polarization we have M q-values
    # we have M constellation symbols
    L = butterfly_filter.taps.shape[1]
    L_offset = (L - 1) // 2
    if p_constellation is None:
        p_constellation = (
            torch.ones_like(constellation_symbols) / constellation_symbols.shape[0]
        )

    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2
    E_Q_x = torch.zeros(2, N, device=q.device, dtype=torch.complex64)
    E_Q_x_abssq = torch.zeros(2, N, device=q.device, dtype=torch.float32)
    if IQ_separate == True:
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

    # Term A - sum all the things, but spare the first dimension, since the two polarizations
    # are sorta independent
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
    h_conv_E_Q_x = butterfly_filter(
        E_Q_x, mode="valid"
    )  # Due to definition that we assume the symbol is at the center tap we remove (filter_length - 1)//2 at start and end
    # Limit the length of y to the "computable space" because y depends on more past values than given
    # We try to generate the received symbol sequence with the estimated symbol sequence
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
    loss = torch.sum(A) + torch.sum((N - L + 1) / sps * torch.log(C + 1e-8))
    var = C / (N - L + 1) * sps
    return loss, var


##############################################################################################


class VAE_LE_DP(torch.nn.Module):
    """
    Adaptive Equalizer based on the variational autoencoder principle with a linear equalizer.

    This code is based on the work presented in [1].

    [1] V. Lauinger, F. Buchali, and L. Schmalen, ‘Blind equalization and channel estimation in coherent optical communications using variational autoencoders’, IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2529–2539, Sep. 2022, doi: 10.1109/JSAC.2022.3191346.

    """

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
    ):
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

    def reset(self):
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
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})

    @torch.enable_grad()
    def forward(self, y):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        for i, k in enumerate(
            range(
                index_padding,
                num_samps
                - index_padding
                - self.sps
                * self.block_size,  # Back-off one block-size + filter_overlap from end to avoid overrunning
                self.sps * self.block_size,
            )
        ):
            # if i % (20000//self.block_size) == 0 and i != 0:
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index], "valid")

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            if self.IQ_separate == True:
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
            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
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

            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            if self.var_from_estimate == True:
                self.demapper.noise_sigma = torch.clamp(
                    torch.mean(var.detach().clone()),
                    min=torch.tensor(0.05, requires_grad=False, device=q_hat.device),
                    max=2
                    * self.demapper.noise_sigma.detach().clone(),  # torch.sqrt(var).detach()), min=0.1
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
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])
        if self.requires_q == True:
            return torch.cat(out, axis=1), torch.cat(out_q, axis=1)
        else:
            return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


def update_ZF(y_hat_sym, pilot_seq, pilot_seq_up, _, idx, length, sps):
    e_k = pilot_seq[idx] - y_hat_sym[idx]
    idx_up = idx * sps

    result = e_k * torch.flip(
        pilot_seq_up[idx_up : idx_up + length].conj().resolve_conj(), dims=(0,)
    )
    return result


def update_LMS(y_hat_sym, pilot_seq, _, y_samp, idx, length, sps):
    e_k = pilot_seq[idx] - y_hat_sym[idx]
    idx_up = idx * sps

    return e_k * torch.flip(
        y_samp[idx_up : idx_up + length].conj().resolve_conj(), dims=(0,)
    )


class PilotAEQ_DP(torch.nn.Module):
    """
    Perform pilot-based adaptive equalization (QPSK)
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
    ):
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
        if method == "LMS":
            self.update = update_LMS
        elif method == "ZF":
            self.update = update_ZF
        elif method == "LMS_ZF":
            self.update = update_LMS

    def reset(self):
        self.butterfly_filter = Butterfly2x2(num_taps=self.filter_length.item())
        self.butterfly_filter.taps[0, self.filter_length.item() // 2] = 1.0
        self.butterfly_filter.taps[2, self.filter_length.item() // 2] = 1.0

    def forward(self, y):
        y_cut = correct_start_polarization(y, self.pilot_sequence_up)

        equalizer_length = self.butterfly_filter.taps.size()[1]
        eq_offset = ((equalizer_length - 1) // 2) // self.sps
        num_samp = y_cut.shape[1]
        u = torch.zeros(
            (4, (num_samp - equalizer_length) // self.sps, equalizer_length),
            dtype=torch.complex64,
        )
        lr = self.lr  # 1e-3
        out = torch.zeros(
            2, (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )
        for i, k in enumerate(range(equalizer_length, num_samp - 1, self.sps)):
            in_index = torch.arange(k - equalizer_length, k)
            out_tmp = self.butterfly_filter(y_cut[:, in_index], "valid")
            out[:, i] = out_tmp.squeeze()
            # Out will contain samples starting with
            # ((filter_length-1)//2)//sps
            if i * self.sps + 2 * eq_offset < self.pilot_sequence.shape[1]:
                if i == 3000 and self.method == "LMS_ZF":
                    self.update = update_ZF
                u[0, i, :] = self.update(
                    out[0, :],
                    self.pilot_sequence[0, eq_offset:],
                    self.pilot_sequence_up[0, :],
                    y_cut[0, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[1, i, :] = self.update(
                    out[0, :],
                    self.pilot_sequence[0, eq_offset:],
                    self.pilot_sequence_up[1, :],
                    y_cut[1, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[2, i, :] = self.update(
                    out[1, :],
                    self.pilot_sequence[1, eq_offset:],
                    self.pilot_sequence_up[1, :],
                    y_cut[1, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[3, i, :] = self.update(
                    out[1, :],
                    self.pilot_sequence[1, eq_offset:],
                    self.pilot_sequence_up[0, :],
                    y_cut[0, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                self.butterfly_filter.taps = self.butterfly_filter.taps + (
                    2 * lr * u[:, i, :].squeeze()
                )
        # Add some padding in the start
        out = torch.cat((torch.zeros((2, 100), dtype=torch.complex64), out), dim=1)
        return out


class PilotAEQ_SP(torch.nn.Module):
    """
    Perform pilot-based adaptive equalization (QPSK)
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
        super(PilotAEQ_SP, self).__init__()
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("pilot_sequence", torch.as_tensor(pilot_sequence))
        self.register_buffer("pilot_sequence_up", torch.as_tensor(pilot_sequence_up))
        self.register_buffer(
            "taps", torch.zeros((filter_length,), dtype=torch.complex64)
        )
        self.taps[filter_length // 2] = 1.0
        if method == "ZF":
            self.update = update_LMS
        else:
            self.update = update_ZF

    def reset(self):
        self.taps.zero_()
        self.taps[self.taps.size()[0] // 2] = 1.0

    def forward(self, y):
        y_cut = correct_start_polarization(y, self.pilot_sequence_up)

        equalizer_length = self.taps.size()[0]
        eq_offset = ((equalizer_length - 1) // 2) // self.sps
        num_samp = y_cut.shape[1]
        u = torch.zeros(
            ((num_samp - equalizer_length) // self.sps, equalizer_length),
            dtype=torch.complex64,
        )
        lr = self.lr  # 1e-3
        out = torch.zeros(
            (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )
        for i, k in enumerate(range(equalizer_length, num_samp - 1, self.sps)):
            in_index = torch.arange(k - equalizer_length, k)
            out_tmp = convolve_overlap_save(y_cut[:, in_index], self.taps, "valid")
            out[:, i] = out_tmp.squeeze()
            # Out will contain samples starting with
            # ((filter_length-1)//2)//sps
            if i * self.sps + 2 * eq_offset < self.pilot_sequence.shape[1]:
                u[i, :] = self.update(
                    out[0, :],
                    self.pilot_sequence[0, eq_offset:],
                    self.pilot_sequence_up[0, :],
                    y_cut[0, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
            self.taps = self.taps + (2 * lr * u[i, :].squeeze())
        # Add some padding in the start
        out = torch.cat((torch.zeros((2, 100), dtype=torch.complex64), out), dim=1)
        return out


class AEQ_SP(torch.nn.Module):
    """
    Perform CMA equalization
    """

    taps: torch.Tensor

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
        self.taps.zero_()
        self.taps[self.taps.shape[0] // 2] = 1.0

    def forward(self, y):
        # Implement CMA "by hand"
        # Basically step through the signal advancing always +sps symbols
        # and filtering 2*filter_len samples which will give one output sample with mode "valid"

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
        eq_offset = (
            equalizer_length - 1
        ) // 2  # We try to put the symbol of interest in the center tap of the equalizer
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
        return self.out_e
