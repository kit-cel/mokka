"""PyTorch implementations of adaptive equalizers."""

from ..torch import Butterfly2x2, ButterflyNxN
from ..torch import correct_start_polarization, correct_start
from ...functional.torch import convolve_overlap_save
from ..torch import h2f
from mokka.synchronizers.phase.torch import BPS, vandv
from mokka.functional.torch import unwrap
import torch
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

import scipy.io

from collections import namedtuple


class CMA(torch.nn.Module):
    """Class to perform CMA equalization."""

    butterfly_filter: ButterflyNxN #Butterfly2x2

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
        num_channels=2,
        mode='valid'
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
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        self.register_buffer("mode", torch.as_tensor(mode))
        if butterfly_filter is not None:
            self.butterfly_filter = butterfly_filter
            self.register_buffer(
                "filter_length", torch.as_tensor(butterfly_filter.taps.shape[1])
            )
        else:
            self.butterfly_filter = ButterflyNxN(
                num_taps=filter_length,
                num_channels=num_channels,
                trainable=True,
                timedomain=True,
                mode=mode,
            )
            self.register_buffer("filter_length", torch.as_tensor(filter_length))
        # Do some clever initalization, first only equalize x-pol and then enable y-pol
        self.register_buffer("no_singularity", torch.as_tensor(no_singularity))
        self.register_buffer("singularity_length", torch.as_tensor(singularity_length))
        if self.no_singularity: ### FIXME: taps.shape = num_channels, num_channels, num_taps
            self.butterfly_filter.taps[2, :] = 0
            self.butterfly_filter.taps[3, :] = 0

    def reset(self):
        """Reset equalizer and butterfly filters."""
        self.butterfly_filter = ButterflyNxN(
                num_taps=self.filter_length,
                num_channels=self.num_channels,
                trainable=True,
                timedomain=True,
                mode=self.mode,
            )
        if self.no_singularity:### FIXME: taps.shape = num_channels, num_channels, num_taps
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

        equalizer_length = self.butterfly_filter.taps.size()[-1]
        num_samp = y.shape[1]
        e = torch.zeros(
            (2, (num_samp - equalizer_length) // self.sps), dtype=torch.float32
        )
        # logger.debug("CMA loop num: %s", reset_number)
        R = self.R  # 1.0
        lr = self.lr  # 1e-3
        out = torch.zeros(
            self.num_channels, (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
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
            update = torch.empty_like(self.butterfly_filter.taps)
            for i in range(self.num_channels):
                for o in range(self.num_channels):
                    update[i,o,:] = 0   ### FIXME: taps.shape = num_channels, num_channels, num_taps

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
    
##############################################################################################
##############################################################################################


class CMloss_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels=None,
        taps = None,
        num_taps=None,
        demapper=None,
        sps=2,
        block_size=200,
        cpe_window_length = 50,
        angles_per_quadrant = 16,
        lr=0.5e-2,
        requires_q=False,
        use_corr_avoid=False,
        alpha_corr=1e-4,
        corr_window=50,
        IQ_separate=False,
        device="cpu",
        mode="valid",
    ):
        super(CMloss_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        if num_taps != None:
            self.register_buffer("num_taps", torch.as_tensor(num_taps))
        self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
        self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("use_corr_avoid", torch.as_tensor(use_corr_avoid))
        self.register_buffer("alpha_corr", torch.as_tensor(alpha_corr))
        self.register_buffer("corr_window", torch.as_tensor(corr_window))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.mode = mode
        temp_sq = torch.abs(demapper.constellation)**2
        self.kurtosis = torch.sum(demapper.p_symbols* temp_sq**2/temp_sq )     #/pow_mean --> consider pow_mean as 1

         ########## Equalizer ############
        if taps is None:
            if num_taps is None:
                raise ValueError("Please provide either a valid num_taps_EQ or taps_EQ.")
            else:
                self.butterfly_forward = ButterflyNxN(
                    num_taps=num_taps,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_forward = ButterflyNxN(
                taps=taps,
                num_channels=num_channels,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )

        self.loss_func = torch.nn.MSELoss()
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )

        # self.optimizer_var = torch.optim.Adam(
        #     [self.demapper.noise_sigma],
        #     lr=0.5,  # 0.5e-2,
        # )

        self.cpe = BPS(
            self.angles_per_quadrant,
            demapper.constellation,
            self.cpe_window_length,
            diff=False,
            temperature_per_epoch=1e-3,
            no_sectors=4,
            avg_filter_type="tri", #"rect",
            trainable=False,
        )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )


    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = ButterflyNxN(
            num_taps=self.num_taps.item(),
            num_channels=self.num_channels,
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
            mode=self.mode,
        )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )

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
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index])

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]


            y_abs_sq = torch.abs(y_symb)**2
            loss = self.loss_func(
                # loss, var = ELBO_DP(
                y_abs_sq,
                torch.full_like(y_abs_sq, self.kurtosis, requires_grad = False)
            )

            if self.use_corr_avoid == True:
                #corr_loss = torch.zeros_like(loss)
                for kk in range(y.shape[0]):             # calculate correlation of each channel to all others
                    ind_corr = torch.arange(y.shape[0])
                    ind_corr = ind_corr[ind_corr!=kk]
                    loss += self.alpha_corr*torch.sum(torch.abs(
                        torch.conv1d(
                            y_symb[kk,:].unsqueeze(0),
                            y_symb[ind_corr,y_symb.shape[-1]-self.corr_window//2:y_symb.shape[-1]+self.corr_window//2].unsqueeze(1),
                            padding='valid',
                        )
                    )**2)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            output_symbols = y_symb[
                :, : self.block_size
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])

            # output_q = q_hat[:, : self.block_size, :]
            # out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        out_const = torch.cat(out, axis=1).detach().clone()
        out_q = out_const * (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])

        for cc in range(self.num_channels):
            out_q[cc,:] = self.cpe(out_q[cc,:])[0]
        if self.IQ_separate == True:
            num_level = self.demapper.constellation.shape[-1]
            q_hat = torch.zeros(
                out_q.shape[0],
                out_q.shape[1],
                2 * num_level,
                device=out_q.device,
                dtype=torch.float32,
            )
            for out_chan in range(out_q.shape[0]):
                q_hat[out_chan, :, :num_level] = self.demapper(
                    out_q[out_chan, :].real
                ).unsqueeze(0)
                q_hat[out_chan, :, num_level:] = self.demapper(
                    out_q[out_chan, :].imag
                ).unsqueeze(0)
        else:
            q_hat = torch.zeros(
                out_q.shape[0],
                out_q.shape[1],
                self.demapper.constellation.shape[-1],
                device=out_q.device,
                dtype=torch.float32,
            )
            for out_chan in range(out_q.shape[0]):
                q_hat[out_chan, :, :] = self.demapper(
                    out_q[out_chan, :]
                ).unsqueeze(0)


        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"])
            return eq_out(out_q, q_hat, loss) #eq_out(torch.cat(out, axis=1), torch.cat(out_q, axis=1), loss)
        return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################


class RDloss_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    FIXME: not yet implemented... still CMloss
    """

    def __init__(
        self,
        num_channels,
        num_taps,
        demapper,
        sps,
        block_size=200,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        device="cpu",
        mode="valid",
    ):
        super(RDloss_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        self.register_buffer("num_taps", torch.as_tensor(num_taps))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.mode = mode
        temp_sq = torch.abs(demapper.constellation)**2
        self.kurtosis = torch.sum(demapper.p_symbols* temp_sq**2/temp_sq )     #/pow_mean --> consider pow_mean as 1

        self.loss_func = torch.nn.MSELoss()
        self.butterfly_forward = ButterflyNxN(
            num_taps=num_taps,
            num_channels=num_channels,
            trainable=True,
            timedomain=True,
            device=device,
            mode=mode,
        )
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )

        self.optimizer_var = torch.optim.Adam(
            [self.demapper.noise_sigma],
            lr=0.5,  # 0.5e-2,
        )

    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = ButterflyNxN(
            num_taps=self.num_taps.item(),
            num_channels=self.num_channels,
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
            mode=self.mode,
        )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )

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
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index])

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                (self.butterfly_forward.num_taps - 1)
                // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            ]
            y_abs_sq = torch.abs(y_symb)**2
            loss = self.loss_func(
                # loss, var = ELBO_DP(
                torch.abs(y_symb)**2,
                torch.full_like(y_abs_sq, self.kurtosis, requires_grad = False)
            )

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

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
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"])
            return eq_out(torch.cat(out, axis=1), torch.cat(out_q, axis=1), loss)
        return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

##############################################################################################
##############################################################################################


class MSEloss_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels,
        num_taps,
        demapper,
        sps,
        block_size=200,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        use_cpe=False,
        cpe_window_length=None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(MSEloss_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        self.register_buffer("num_taps", torch.as_tensor(num_taps))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        if cpe_window_length != None:
            self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))

        self.register_buffer("N_phi_ave", torch.as_tensor(50))      # 50 at code of IPQ
        self.mode = mode
        
        # f = scipy.io.loadmat("waveforms/data_from_IPQ/processed_data_90GBd_LO17_16qam_gray_VPPx_270mV_VPPy_210mV_scale25mV_C4CF4_10dBm_1.mat") #weights.mat")
        # weigh = f.get('wk_cma')
        # taps = torch.zeros(8,8,201, dtype=torch.complex64)
        # taps[:,:,1:] = torch.from_numpy(weigh.reshape(8,8,200)).permute(1,0,2)

        self.loss_func = torch.nn.MSELoss()
        self.butterfly_forward = ButterflyNxN(
            # taps=taps,
            num_taps=num_taps,
            num_channels=num_channels,
            trainable=True,
            timedomain=True,
            device=device,
            mode=mode,
        )
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )

        self.optimizer_var = torch.optim.Adam(
            [self.demapper.noise_sigma],
            lr=0.5,  # 0.5e-2,
        )

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                #cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=1e-3,
                    no_sectors=4,
                    avg_filter_type="tri", #"rect",
                    trainable=False,
                )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )

    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = ButterflyNxN(
            num_taps=self.num_taps.item(),
            num_channels=self.num_channels,
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
            mode=self.mode,
        )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )

    def forward(self, y, x):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        ref_out = []

        out = []
        out_q = []
        phi_plot = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        ref_up = torch.zeros_like(y)
        ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]

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
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index])

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                (self.butterfly_forward.num_taps - 1)
                // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            ]
            #ref_index = in_index[ (self.butterfly_forward.num_taps - 1) : ] + (self.butterfly_forward.num_taps + 1)*self.sps + 86
            temp_ref = ref_up[:,y_index]
            ref = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 

            
            if self.use_cpe == True:
                N_phi_ave = self.N_phi_ave
                phi_off = (torch.angle(y_symb) - torch.angle(ref)).detach() # unwrap necessary?
                y_phi_sum = torch.zeros_like(y_symb)[:,:-N_phi_ave]
                for ii in range(N_phi_ave):
                    y_phi_sum += torch.exp(1j*phi_off[:,ii:-(N_phi_ave-ii)])    # sum in the complex domain
                phi_ma = torch.angle(y_phi_sum)
                ref_loss = ref[:,N_phi_ave//2:-(N_phi_ave-N_phi_ave//2)] *torch.exp(1j*phi_ma) 
                loss = self.loss_func( y_symb[:,N_phi_ave//2:-(N_phi_ave-N_phi_ave//2)].real, ref_loss.real ) + self.loss_func( y_symb[:,N_phi_ave//2:-(N_phi_ave-N_phi_ave//2)].imag, ref_loss.imag)
                # for cc in range(self.num_channels):
                #     ref[cc,:] *= torch.exp(1j*self.cpe(y_symb[cc,:])[1])    # rotate reference to make update independet of phase noise
                # loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)
                phi_plot.append(phi_ma)
            else:
                loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            output_symbols = y_symb[
                :, : self.block_size
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])

#########################
            ref_tmp = ref[
                :, : self.block_size
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            ref_out.append(
                ref_tmp
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])
##############################

            output_q = q_hat[:, : self.block_size, :]
            out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        out_const = torch.cat(out, axis=1).detach().clone()
        out_const *= (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"]) #, "phi_ref"])
            return eq_out(out_const, q_hat, loss) #, torch.cat(phi_plot, axis=1))
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################
 ##############################################################################################
##############################################################################################


class MSEflex_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels,
        num_taps,
        demapper,
        sps,
        block_size=200,
        symb_per_step = 10,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        use_cpe=False,
        cpe_window_length = None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(MSEflex_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("symb_per_step", torch.as_tensor(symb_per_step))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        self.register_buffer("num_taps", torch.as_tensor(num_taps))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        if cpe_window_length != None:
            self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))

        self.register_buffer("N_phi_ave", torch.as_tensor(32))      # 50 at code of IPQ
        self.mode = mode
        
        # f = scipy.io.loadmat("waveforms/data_from_IPQ/weights.mat")
        # weigh = f.get('wk_cma')
        # taps = torch.zeros(8,8,151, dtype=torch.complex64)
        # taps[:,:,1:] = torch.from_numpy(weigh.reshape(8,8,150)).permute(1,0,2)

        self.pad_func = torch.nn.ConstantPad1d((num_taps - 1) // 2, 0)
        self.loss_func = torch.nn.MSELoss()
        self.butterfly_forward = ButterflyNxN(
            # taps=taps,
            num_taps=num_taps,
            num_channels=num_channels,
            trainable=True,
            timedomain=True,
            device=device,
            mode=mode,
        )
        taps_ma = torch.zeros(self.num_channels,self.num_channels,self.N_phi_ave,dtype=torch.complex64, device=device)
        for ma_idx in range(self.num_channels):
            taps_ma[ma_idx,ma_idx,:] = torch.ones(self.N_phi_ave,dtype=torch.complex64, device=device)
        self.moving_ave = ButterflyNxN(
            taps=taps_ma,
            #num_taps=num_taps,
            num_channels=num_channels,
            trainable=False,
            timedomain=True,
            device=device,
            mode="valid",
        )
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )

        self.optimizer_var = torch.optim.Adam(
            [self.demapper.noise_sigma],
            lr=0.5,  # 0.5e-2,
        )

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                #cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=1e-3,
                    no_sectors=4,
                    avg_filter_type="tri", #"rect",
                    trainable=False,
                )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )

    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = ButterflyNxN(
            num_taps=self.num_taps.item(),
            num_channels=self.num_channels,
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
            mode=self.mode,
        )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )

    def forward(self, y, x):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        assert self.N_phi_ave < self.butterfly_forward.num_taps     # TODO: implement for both cases

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        phi_plot = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        ref_up = torch.zeros_like(y)
        ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]

        mod_symb_per_step_2 = self.symb_per_step%2
        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        temp_eqIn = torch.zeros(self.N_phi_ave, y.shape[0], self.sps * self.block_size + 2*index_padding, dtype=y.dtype, device=y.device)     # shift register with dim: N_phi_ave, num. of channels, block_size*sps
        # temp_phi = torch.zeros(self.N_phi_ave, dtype=y.dtype, device=y.device)
        temp_ref = torch.zeros(self.N_phi_ave, y.shape[0], self.block_size, dtype=y.dtype, device=y.device)
        
        y_pad = self.pad_func(y)
        ref_pad = self.pad_func(x)
        # # init ref matrix
        # temp_eqIn[self.N_phi_ave//2,:,:] = y[:, 0 : self.sps * self.block_size + 2*index_padding]
        # for jj in range(0,self.N_phi_ave//2):
        #     temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y[:, jj+1 : jj+1 + self.sps * self.block_size + 2*index_padding]  # later in time
        #     temp_eqIn[self.N_phi_ave//2 + jj,:,jj:] = y[:, : -jj + self.sps * self.block_size + 2*index_padding]  # earlier in time

        for i, k in enumerate(
            range(
                index_padding,
                num_samps
                - index_padding
                - self.sps
                * self.block_size,  # Back-off one block-size + filter_overlap from end to avoid overrunning
                self.sps * self.symb_per_step,
            )
        ):
            # if i % (20000//self.block_size) == 0 and i != 0:
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k, # - index_padding,   # padding already included in y_pad
                k + self.sps * self.block_size + 2*index_padding,   # times two since padding already included in y_pad
            )
            ref_index = torch.arange(
                i * self.symb_per_step + (index_padding+1)//2 + index_padding,  
                i * self.symb_per_step + (index_padding+1)//2 + index_padding + self.block_size,   
            )

            for jj in range(0,self.N_phi_ave//2):
                temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y_pad[:, self.sps*jj+1+in_index]  # later in time
                temp_eqIn[self.N_phi_ave//2 + jj,:,:] = y_pad[:, in_index - self.sps*jj]  # earlier in time
                temp_ref[self.N_phi_ave//2 - jj-1,:,:] = ref_pad[:, jj+1+ref_index]  # later in time
                temp_ref[self.N_phi_ave//2 + jj,:, :] = ref_pad[:, ref_index - jj] # earlier in time

            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(temp_eqIn)[ : , : , 0 :: self.sps ]

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                self.N_phi_ave//2, :, :  #0 :: self.sps
            ].squeeze()  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            # y_index = in_index[
            #     (self.butterfly_forward.num_taps - 1)
            #     // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            # ]
            # temp_symb = temp_symb.roll(shifts=(1), dims=(0))
            # #ref_index = in_index[ (self.butterfly_forward.num_taps - 1) : ] + (self.butterfly_forward.num_taps + 1)*self.sps + 86
            # temp_ref = ref_up[:,y_index]
            # temp_symb[0,:,:] = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 
            # ref = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 
            ref = temp_ref[self.N_phi_ave//2, :, :]
            
            if self.use_cpe == True:
                # N_phi_ave = self.N_phi_ave
                phi_off = (torch.angle(y_hat) - torch.angle(temp_ref)).detach() # unwrap necessary?
                # y_phi_sum = torch.zeros_like(y_symb)[:,:-N_phi_ave]
                # self.moving_ave(torch.exp(1j*phi_off[16,:,:]))
                # for ii in range(N_phi_ave):
                #     y_phi_sum += torch.exp(1j*phi_off[:,ii:-(N_phi_ave-ii)])    # sum in the complex domain
                phi_ma = torch.angle(torch.sum(torch.exp(1j*phi_off) , dim=0) )
                ref_loss = ref *torch.exp(1j*phi_ma) 
                loss = self.loss_func( y_symb.real, ref_loss.real ) + self.loss_func( y_symb.imag, ref_loss.imag)
                # for cc in range(self.num_channels):
                #     ref[cc,:] *= torch.exp(1j*self.cpe(y_symb[cc,:])[1])    # rotate reference to make update independet of phase noise
                # loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)
                phi_plot.append(phi_ma)
            else:
                loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            # what about odd symbols_per_step?
            if i==0:
                ind_out = torch.arange( 0, (self.block_size+self.symb_per_step)//2 ) 
            else:
                ind_out = torch.arange( (self.block_size-self.symb_per_step)//2, (self.block_size+self.symb_per_step)//2 ) + mod_symb_per_step_2
            
            output_symbols = y_symb[
                :, ind_out
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])

            output_q = q_hat[:, ind_out, :]
            out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        ###################

        out_const = torch.cat(out, axis=1).detach().clone()
        out_const *= (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"]) #, "phi_ref"])
            return eq_out(out_const, q_hat, loss) #, torch.cat(phi_plot, axis=1))
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################
 ##############################################################################################
##############################################################################################

class MSEflex_phiMA_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels,
        num_taps,
        demapper,
        sps,
        block_size=200,
        symb_per_step = 10,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        use_cpe=False,
        device="cpu",
        mode="valid",
    ):
        super(MSEflex_phiMA_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("symb_per_step", torch.as_tensor(symb_per_step))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        self.register_buffer("num_taps", torch.as_tensor(num_taps))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))

        self.register_buffer("N_phi_ave", torch.as_tensor(20))      # 50 at code of IPQ
        self.mode = mode
        
        # f = scipy.io.loadmat("waveforms/data_from_IPQ/weights.mat")
        # weigh = f.get('wk_cma')
        # taps = torch.zeros(8,8,151, dtype=torch.complex64)
        # taps[:,:,1:] = torch.from_numpy(weigh.reshape(8,8,150)).permute(1,0,2)

        self.pad_func = torch.nn.ConstantPad1d((num_taps - 1) // 2, 0)
        self.loss_func = torch.nn.MSELoss()
        self.butterfly_forward = ButterflyNxN(
            # taps=taps,
            num_taps=num_taps,
            num_channels=num_channels,
            trainable=True,
            timedomain=True,
            device=device,
            mode=mode,
        )
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )

        self.optimizer_var = torch.optim.Adam(
            [self.demapper.noise_sigma],
            lr=0.5,  # 0.5e-2,
        )

        cpe_window_length = 50
        self.cpe = BPS(
            16,
            demapper.constellation,
            cpe_window_length,
            diff=False,
            temperature_per_epoch=1e-3,
            no_sectors=4,
            avg_filter_type="rect",
            trainable=False,
        )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )

    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = ButterflyNxN(
            num_taps=self.num_taps.item(),
            num_channels=self.num_channels,
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
            mode=self.mode,
        )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )

    def forward(self, y, x):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        assert self.N_phi_ave < self.butterfly_forward.num_taps     # TODO: implement for both cases

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        phi_plot = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        ref_up = torch.zeros_like(y)
        ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]

        mod_symb_per_step_2 = self.symb_per_step%2
        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        temp_eqIn = torch.zeros(self.N_phi_ave, y.shape[0], self.sps * self.block_size + 2*index_padding, dtype=y.dtype, device=y.device)     # shift register with dim: N_phi_ave, num. of channels, block_size*sps
        temp_ref = torch.zeros(self.N_phi_ave, y.shape[0], self.block_size, dtype=y.dtype, device=y.device)
        
        y_pad = self.pad_func(y)
        ref_pad = self.pad_func(x)

        for i, k in enumerate(
            range(
                index_padding,
                num_samps
                - index_padding
                - self.sps
                * self.block_size,  # Back-off one block-size + filter_overlap from end to avoid overrunning
                self.sps * self.symb_per_step,
            )
        ):
            in_index = torch.arange(
                k, # - index_padding,   # padding already included in y_pad
                k + self.sps * self.block_size + 2*index_padding,   # times two since padding already included in y_pad
            )
            ref_index = torch.arange(
                i * self.symb_per_step + (index_padding+1)//2 + index_padding,  
                i * self.symb_per_step + (index_padding+1)//2 + index_padding + self.block_size,   
            )

            for jj in range(0,self.N_phi_ave//2):
                temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y_pad[:, self.sps*jj+1+in_index]  # later in time
                temp_eqIn[self.N_phi_ave//2 + jj,:,:] = y_pad[:, in_index - self.sps*jj]  # earlier in time
                temp_ref[self.N_phi_ave//2 - jj-1,:,:] = ref_pad[:, jj+1+ref_index]  # later in time
                temp_ref[self.N_phi_ave//2 + jj,:, :] = ref_pad[:, ref_index - jj] # earlier in time

            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(temp_eqIn)[ : , : , 0 :: self.sps ]

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                self.N_phi_ave//2, :, :  #0 :: self.sps
            ].squeeze()  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            ref = temp_ref[self.N_phi_ave//2, :, :]
            if self.use_cpe == True:
                phi_off = (torch.angle(y_hat) - torch.angle(temp_ref)).detach() # unwrap necessary?
                phi_ma = torch.angle(torch.sum(torch.exp(1j*phi_off) , dim=0) )
                ref_loss = ref *torch.exp(1j*phi_ma) 
                loss = self.loss_func( y_symb.real, ref_loss.real ) + self.loss_func( y_symb.imag, ref_loss.imag)
                phi_plot.append(phi_ma)
            else:
                loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            # what about odd symbols_per_step?
            if i==0:
                ind_out = torch.arange( 0, (self.block_size+self.symb_per_step)//2 ) 
            else:
                ind_out = torch.arange( (self.block_size-self.symb_per_step)//2, (self.block_size+self.symb_per_step)//2 ) + mod_symb_per_step_2
            
            output_symbols = y_symb[
                :, ind_out
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])

            output_q = q_hat[:, ind_out, :]
            out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        ###################

        out_const = torch.cat(out, axis=1).detach().clone()
        out_const *= (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"]) #, "phi_ref"])
            return eq_out(out_const, q_hat, loss) #, torch.cat(phi_plot, axis=1))
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################
 ##############################################################################################
##############################################################################################


class LMS_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels,
        num_taps,
        demapper,
        sps,
        block_size=200,
        symb_per_step = 10,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        use_cpe=False,
        cpe_window_length=None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(LMS_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("symb_per_step", torch.as_tensor(symb_per_step))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        self.register_buffer("num_taps", torch.as_tensor(num_taps))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        if cpe_window_length != None:
            self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))

        self.register_buffer("N_phi_ave", torch.as_tensor(32))      # 50 at code of IPQ
        self.mode = mode
        
        # f = scipy.io.loadmat("waveforms/data_from_IPQ/weights.mat")
        # weigh = f.get('wk_cma')
        # taps = torch.zeros(8,8,151, dtype=torch.complex64)
        # taps[:,:,1:] = torch.from_numpy(weigh.reshape(8,8,150)).permute(1,0,2)

        self.pad_func = torch.nn.ConstantPad1d((num_taps - 1) // 2, 0)
        self.loss_func = torch.nn.MSELoss()
        self.butterfly_forward = ButterflyNxN(
            # taps=taps,
            num_taps=num_taps,
            num_channels=num_channels,
            trainable=False,
            timedomain=True,
            device=device,
            mode=mode,
        )
        self.demapper = demapper

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                #cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=1e-3,
                    no_sectors=4,
                    avg_filter_type="rect",
                    trainable=False,
                )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )

    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = ButterflyNxN(
            num_taps=self.num_taps.item(),
            num_channels=self.num_channels,
            trainable=False,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
            mode=self.mode,
        )

    def forward(self, y, x):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        assert self.N_phi_ave < self.butterfly_forward.num_taps     # TODO: implement for both cases

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        phi_plot = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        mod_symb_per_step_2 = self.symb_per_step%2
        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        temp_eqIn = torch.zeros(self.N_phi_ave, y.shape[0], self.butterfly_forward.num_taps, dtype=y.dtype, device=y.device)     # shift register with dim: N_phi_ave, num. of channels, block_size*sps
        # temp_phi = torch.zeros(self.N_phi_ave, dtype=y.dtype, device=y.device)
        temp_ref = torch.zeros(self.N_phi_ave, y.shape[0], 1, dtype=y.dtype, device=y.device)
        
        num_out_samps = (num_samps - 2*index_padding - self.butterfly_forward.num_taps)//self.sps +1
        y_hat_out = torch.zeros(y.shape[0], num_out_samps, dtype=y.dtype, device=y.device)
        err_out = torch.zeros(y.shape[0], num_out_samps, dtype=y.dtype, device=y.device)

        y_pad = self.pad_func(y)
        ref_pad = self.pad_func(x)
        # # init ref matrix
        # temp_eqIn[self.N_phi_ave//2,:,:] = y[:, 0 : self.sps * self.block_size + 2*index_padding]
        # for jj in range(0,self.N_phi_ave//2):
        #     temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y[:, jj+1 : jj+1 + self.sps * self.block_size + 2*index_padding]  # later in time
        #     temp_eqIn[self.N_phi_ave//2 + jj,:,jj:] = y[:, : -jj + self.sps * self.block_size + 2*index_padding]  # earlier in time

        for i, k in enumerate(
            range(
                index_padding,
                num_samps
                - index_padding
                - self.butterfly_forward.num_taps,  # Back-off one block-size + filter_overlap from end to avoid overrunning
                self.sps,
            )
        ):
            # if i % (20000//self.block_size) == 0 and i != 0:
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k, # - index_padding,   # padding already included in y_pad
                k + self.butterfly_forward.num_taps   # times two since padding already included in y_pad
            )
            ref_index = torch.arange(
                i + (index_padding+1)//2 + index_padding  - self.N_phi_ave//2,
                i + (index_padding+1)//2 + index_padding  + self.N_phi_ave//2
            )

            temp_ref = (ref_pad[:,ref_index].permute(1,0)).to(y.dtype)#.unsqueeze(-1)


            for jj in range(0,self.N_phi_ave//2):
                temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y_pad[:, self.sps*jj+1+in_index]  # later in time
                temp_eqIn[self.N_phi_ave//2 + jj,:,:] = y_pad[:, in_index - self.sps*jj]  # earlier in time
                #temp_ref[self.N_phi_ave//2 - jj-1,:,:] = ref_pad[:, jj+1+ref_index]  # later in time
                #temp_ref[self.N_phi_ave//2 + jj,:, :] = ref_pad[:, ref_index - jj] # earlier in time

            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(temp_eqIn)

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb = y_hat[
                self.N_phi_ave//2, : #0 :: self.sps
            ].squeeze()  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]
            ref = temp_ref[self.N_phi_ave//2, :].squeeze()
            
            with torch.set_grad_enabled(False):
                #N_phi_ave = self.N_phi_ave
                phi_off = (torch.angle(y_hat) - torch.angle(temp_ref)).detach() # unwrap necessary?
                #y_phi_sum = torch.zeros_like(y_symb)[:,:-N_phi_ave]
                # for ii in range(N_phi_ave):
                #     y_phi_sum += torch.exp(1j*phi_off[:,ii:-(N_phi_ave-ii)])    # sum in the complex domain
                phi_ma = torch.angle(torch.sum(torch.exp(1j*phi_off) , dim=0) )
                y_hat_out[:,i] = y_symb*torch.exp(-1j*phi_ma) 
                err_out[:,i] = (ref - y_hat_out[:,i])*torch.exp(1j*phi_ma) # should be num_channels x 1
                for in_ch in range(y.shape[0]):
                    self.butterfly_forward.taps[:,in_ch,:] += (2*self.lr * err_out[:,i].unsqueeze(1)  * temp_eqIn[self.N_phi_ave//2,in_ch,:])

        out_const = y_hat_out.detach().clone()
        out_const *= (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
        if self.IQ_separate == True:
            num_level = self.demapper.constellation.shape[-1]
            q_hat = torch.zeros(
                out_const.shape[0],
                out_const.shape[1],
                2 * num_level,
                device=out_const.device,
                dtype=torch.float32,
            )
            for out_chan in range(out_const.shape[0]):
                q_hat[out_chan, :, :num_level] = self.demapper(
                    out_const[out_chan, :].real
                ).unsqueeze(0)
                q_hat[out_chan, :, num_level:] = self.demapper(
                    out_const[out_chan, :].imag
                ).unsqueeze(0)
        else:
            q_hat = torch.zeros(
                out_const.shape[0],
                out_const.shape[1],
                self.demapper.constellation.shape[-1],
                device=out_const.device,
                dtype=torch.float32,
            )
            for out_chan in range(out_const.shape[0]):
                q_hat[out_chan, :, :] = self.demapper(
                    out_const[out_chan, :]
                ).unsqueeze(0)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"]) #, "phi_ref"])
            return eq_out(out_const, q_hat, err_out) #, torch.cat(phi_plot, axis=1))
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################

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


##############################################################################################


def ELBO_NxN(
    y,
    q,
    sps,
    constellation_symbols,
    butterfly_filter,
    phi=None,
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
    num_channels = y.shape[0]
    # Now we have two polarizations in the first dimension
    # We assume the same transmit constellation for both, calculating
    # q needs to be shaped num_channels x N x M  -> for each observation on each polarization we have M q-values
    # we have M constellation symbols
    L = butterfly_filter.taps.shape[-1]
    L_offset = (L - 1) // 2
    if p_constellation is None:
        p_constellation = (
            torch.ones_like(constellation_symbols) / constellation_symbols.shape[0]
        )

    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2
    E_Q_x = torch.zeros_like(
        y
    )  # torch.zeros(2, N, device=q.device, dtype=torch.complex64)
    E_Q_x_abssq = torch.zeros(
        num_channels, N, device=y.device, dtype=torch.float32
    )  # torch.zeros(2, N, device=q.device, dtype=torch.float32)
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
        ) * torch.exp(-1j*phi)
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
        E_Q_x
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
            (E_Q_x_abssq - torch.abs(E_Q_x) ** 2)#.to(torch.float32)
        ),
        axis=1,
    )

    # We compute B without constants
    # B_tilde = -N * torch.log(C)
    # bias1 = 1e-8
    loss = torch.sum(A) + (N - L + 1) * torch.sum(torch.log(C + bias))  # / sps
    var = C / (N - L + 1)  # * sps #N
    return loss, var


##############################################################################################
##############################################################################################


def ELBO_DP_IQ(y, q, sps, amp_levels, h_est, p_amps=None):
    """
    Calculate dual-pol. ELBO loss for I/Q-symmetric constellations by splitting the calculation in two real-valued paths for I/Q separately.

    Instead of calculation in the complex domain, it is split in two real-valued vectors for separate calculation of I/Q.
    This implements the dual-polarization case.
    """
    # Input is a sequence y of length N
    N = y.shape[1]
    h = h_est
    pol = 2  # dual-polarization
    # Now we have two polarizations in the first dimension
    # We assume the same transmit constellation for both, calculating
    # q needs to be shaped 2 x N x M  -> for each observation on each polarization we have M q-values
    # we have M constellation symbols
    L = h.shape[-1]
    L_offset = (L - 1) // 2
    if p_amps is None:
        p_amps = torch.ones_like(amp_levels) / amp_levels.shape[0]

    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2
    E_Q_x = torch.zeros(2, 2, N, device=q.device, dtype=torch.float32)
    Var = torch.zeros(2, N, device=q.device, dtype=torch.float32)
    num_lev = amp_levels.shape[0]
    E_Q_x[:, 0, ::sps] = torch.sum(
        q[:, :, :num_lev] * amp_levels.unsqueeze(0).unsqueeze(0), dim=-1
    )  # .permute(0,2,1)
    E_Q_x[:, 1, ::sps] = torch.sum(
        q[:, :, num_lev:] * amp_levels.unsqueeze(0).unsqueeze(0), dim=-1
    )  # .permute(0,2,1)
    Var[:, ::sps] = torch.add(  # Precompute E_Q{|x|^2}
        torch.sum(
            q[:, :, :num_lev] * (amp_levels**2).unsqueeze(0).unsqueeze(0), dim=-1
        ),
        torch.sum(
            q[:, :, num_lev:] * (amp_levels**2).unsqueeze(0).unsqueeze(0), dim=-1
        ),
    )
    Var[:, ::sps] -= torch.sum(E_Q_x[:, :, ::sps] ** 2, dim=1)
    p_amps = p_amps.repeat(2)

    h_absq = torch.sum(h**2, dim=2)

    D_real = torch.zeros(2, N - 2 * L_offset, device=q.device, dtype=torch.float32)
    D_imag = torch.zeros(2, N - 2 * L_offset, device=q.device, dtype=torch.float32)
    E = torch.zeros(2, device=q.device, dtype=torch.float32)
    idx = torch.arange(2 * L_offset, N)
    nm = idx.shape[0]

    for j in range(2 * L_offset + 1):  # h[chi,nu,c,k]
        D_real += (
            h[:, 0, 0:1, j].expand(-1, nm) * E_Q_x[0, 0:1, idx - j].expand(pol, -1)
            - h[:, 0, 1:2, j].expand(-1, nm) * E_Q_x[0, 1:2, idx - j].expand(pol, -1)
            + h[:, 1, 0:1, j].expand(-1, nm) * E_Q_x[1, 0:1, idx - j].expand(pol, -1)
            - h[:, 1, 1:2, j].expand(-1, nm) * E_Q_x[1, 1:2, idx - j].expand(pol, -1)
        )
        D_imag += (
            h[:, 0, 1:2, j].expand(-1, nm) * E_Q_x[0, 0:1, idx - j].expand(pol, -1)
            + h[:, 0, 0:1, j].expand(-1, nm) * E_Q_x[0, 1:2, idx - j].expand(pol, -1)
            + h[:, 1, 1:2, j].expand(-1, nm) * E_Q_x[1, 0:1, idx - j].expand(pol, -1)
            + h[:, 1, 0:1, j].expand(-1, nm) * E_Q_x[1, 1:2, idx - j].expand(pol, -1)
        )
        Var_sum = torch.sum(Var[:, idx - j], dim=-1)
        E += h_absq[:, 0, j] * Var_sum[0] + h_absq[:, 1, j] * Var_sum[1]

    # Term A - sum all the things, but spare the first dimension, since the two polarizations
    # are sorta independent
    bias = 1e-14
    A = torch.sum(
        q[:, L_offset:-L_offset, :]
        * torch.log(
            (q[:, L_offset:-L_offset, :] / p_amps.unsqueeze(0).unsqueeze(0)) + bias
        ),
        dim=(1, 2),
    )  # Limit the length of y to the "computable space" because y depends on more past values than given
    # We try to generate the received symbol sequence with the estimated symbol sequence
    C = torch.sum(
        y[:, L_offset:-L_offset].real ** 2 + y[:, L_offset:-L_offset].imag ** 2, axis=1
    )
    C += (
        -2
        * torch.sum(
            y[:, L_offset:-L_offset].real * D_real
            + y[:, L_offset:-L_offset].imag * D_imag,
            dim=1,
        )
        + torch.sum(D_real**2 + D_imag**2, dim=1)
        + E
    )

    # We compute B without constants
    # B_tilde = -N * torch.log(C)
    loss = torch.sum(A) + (N - L + 1) * torch.sum(torch.log(C + 1e-8))  # / sps
    var = C / (N - L + 1)  # * sps
    return loss, var


##############################################################################################


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
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(torch.cat(out, axis=1), torch.cat(out_q, axis=1), var, loss)
        return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_var):
        self.demapper.noise_sigma = new_var


##############################################################################################


class VAE_LE_DP_IQ(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization as in:
    V. Lauinger, F. Buchali and L. Schmalen, "Blind equalization and channel estimation in coherent optical communications using variational autoencoders," IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2529-2539, Sep. 2022.
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
        var_from_estimate=False,
        device="cpu",
    ):
        super(VAE_LE_DP_IQ, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_forward))
        self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_backward))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("var_from_estimate", torch.as_tensor(var_from_estimate))
        self.butterfly_forward = Butterfly2x2(
            num_taps=num_taps_forward, trainable=True, timedomain=True, device=device
        )
        pol = 2  # dual-polarization
        self.h_est = torch.zeros(
            [pol, pol, 2, num_taps_backward]
        )  # initialize estimated impulse response
        (
            self.h_est[0, 0, 0, num_taps_backward // 2 + 1],
            self.h_est[1, 1, 0, num_taps_backward // 2 + 1],
        ) = (1, 1)
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )
        self.optimizer.add_param_group({"params": self.h_est})

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
            device=self.butterfly_forward.taps.device,
        )
        pol = 2  # dual-polarization
        self.h_est = torch.zeros(
            [pol, pol, 2, self.num_taps_backward]
        )  # initialize estimated impulse response
        (
            self.h_est[0, 0, 0, self.num_taps_backward // 2 + 1],
            self.h_est[1, 1, 0, self.num_taps_backward // 2 + 1],
        ) = (1, 1)
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )
        self.optimizer.add_param_group({"params": self.h_est})

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
            # print("Updating learning rate")
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

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                (self.butterfly_forward.num_taps - 1)
                // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            ]
            loss, var = ELBO_DP_IQ(
                # loss, var = ELBO_DP(
                y[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.h_est,
                p_amps=self.demapper.p_symbols
                # p_constellation=self.demapper.p_symbols
            )

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

##############################################################################################
##############################################################################################


class VAE_LE_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels=None,
        taps_EQ = None,
        num_taps_EQ=None,
        taps_EST = None,
        num_taps_EST=None,
        demapper=None,
        sps=2,
        block_size=200,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        var_from_estimate=False,
        use_cpe=False,
        use_cpe_in_training=False,
        avg_filter_type="tri",
        use_corr_avoid=False,
        alpha_corr=1e-2,
        corr_window=50,
        cpe_window_length=None,
        cpe_train_window_length=None,
        temperature_per_epoch=None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(VAE_LE_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        if num_taps_EQ != None:
            self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_EQ))
        if num_taps_EST != None:
            self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_EST))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("var_from_estimate", torch.as_tensor(var_from_estimate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        self.register_buffer("use_cpe_in_training", torch.as_tensor(use_cpe_in_training))
        self.register_buffer("use_corr_avoid", torch.as_tensor(use_corr_avoid))
        self.avg_filter_type=avg_filter_type
        self.register_buffer("alpha_corr", torch.as_tensor(alpha_corr))
        self.register_buffer("corr_window", torch.as_tensor(corr_window))
        if cpe_window_length != None:
            if cpe_train_window_length != None:
                self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
                self.register_buffer("cpe_train_window_length", torch.as_tensor(cpe_train_window_length))
            else:
                self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
                self.register_buffer("cpe_train_window_length", torch.as_tensor(cpe_window_length))
        if temperature_per_epoch != None:
            self.register_buffer("temperature_per_epoch", torch.as_tensor(temperature_per_epoch))
        else:
            self.register_buffer("temperature_per_epoch", torch.as_tensor(1e-2))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))
        self.mode = mode
        ########## Equalizer ############
        if taps_EQ is None:
            if num_taps_EQ is None:
                raise ValueError("Please provide either a valid num_taps_EQ or taps_EQ.")
            else:
                self.butterfly_forward = ButterflyNxN(
                    num_taps=num_taps_EQ,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_forward = ButterflyNxN(
                taps=taps_EQ,
                num_channels=num_channels,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
        ########## Estimator ############
        if taps_EST is None:
            if num_taps_EST is None:
                raise ValueError("Please provide either a valid num_taps_EST or taps_EST.")
            else:
                self.butterfly_backward = ButterflyNxN(
                    num_taps=num_taps_EST,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_backward = ButterflyNxN(
                taps=taps_EST,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
        #######################
    
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

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                ##cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation.squeeze(),
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=self.temperature_per_epoch, #1e-3,
                    no_sectors=4,
                    PS_aware=False,
                    p_symbols=demapper.p_symbols.squeeze(),
                    noise_sigma=demapper.noise_sigma,
                    avg_filter_type=self.avg_filter_type,  #"rect",
                    trainable=False,
                )
        if use_cpe_in_training == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation.squeeze(),
                    self.cpe_train_window_length,
                    diff=True,
                    temperature_per_epoch=self.temperature_per_epoch,
                    no_sectors=4,
                    PS_aware=False,
                    p_symbols=demapper.p_symbols.squeeze(),
                    noise_sigma=demapper.noise_sigma,
                    avg_filter_type=self.avg_filter_type, #"rect",  #"rect","tri"
                    trainable=False,
                )
                # self.cpe = vandv.ViterbiViterbi(
                #     window_length=cpe_window_length
                # )
                # self.cpe_in_loop = vandv.ViterbiViterbi(
                #     window_length=cpe_window_length
                # )

    def init_cpe_for_train(self):
        self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    self.demapper.constellation.squeeze(),
                    self.cpe_train_window_length,
                    diff=True,
                    temperature_per_epoch=self.temperature_per_epoch,
                    no_sectors=4,
                    PS_aware=False,
                    p_symbols=self.demapper.p_symbols.squeeze(),
                    noise_sigma=self.demapper.noise_sigma,
                    avg_filter_type=self.avg_filter_type,  #"rect",
                    trainable=False,
                )

    def reset(self):
        self.lr = self.start_lr.clone()
        # self.butterfly_forward = ButterflyNxN(
        #     num_taps=self.num_taps_forward.item(),
        #     num_channels=self.num_channels,
        #     trainable=True,
        #     timedomain=True,
        #     device=self.butterfly_forward.taps.device,
        #     mode=self.mode,
        # )
        # self.butterfly_backward = ButterflyNxN(
        #     num_taps=self.num_taps_backward.item(),
        #     num_channels=self.num_channels.item(),
        #     trainable=True,
        #     timedomain=True,
        #     device=self.butterfly_forward.taps.device,
        #     mode=self.mode,
        # )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})

    def forward(self, y): #, x):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size
##########################
        # phi_ref_full = []
        # phi_full = []
        # ref_full = []
##########################
        out = []
        out_q = []
        var_out = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block
##########################
        # ref_up = torch.zeros_like(y)
        # ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]
##########################
        index_padding = (self.butterfly_forward.num_taps - 1) // 2 + self.cpe_window_length*self.sps
        phi_pre = torch.zeros(y.shape[0], device=y.device, dtype=torch.float32)
        piHalfTensor = torch.tensor(torch.pi/2, device=y.device)
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
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index])

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb_temp = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            y_symb = torch.empty_like(y_symb_temp)[:,self.cpe_window_length : -self.cpe_window_length ]  # y_symb_temp #
            phi_loop = torch.zeros_like(y_symb).real #.to(torch.float32)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                index_padding : -index_padding
            ]
##########################
            # temp_ref = ref_up[:,y_index]
            # ref = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 

            # phi_ref = torch.angle(y_symb_temp[:, self.cpe_window_length : -self.cpe_window_length] * ref.conj())
##########################
            if self.use_cpe_in_training == True:
                y_symb_temp1 =  y_symb_temp.clone()*(torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(y_symb_temp)**2, dim=-1))).unsqueeze(1).repeat(1,y_symb_temp.shape[-1])
                for cc in range(self.num_channels):
                    y_symb_temp2, phi_loop_temp2 = self.cpe_in_loop(y_symb_temp1[cc,:])  # [0]
                    y_symb[cc,:] = y_symb_temp2[self.cpe_window_length : -self.cpe_window_length ].clone()
                    phi_loop[cc,:] = phi_loop_temp2[self.cpe_window_length : -self.cpe_window_length ].clone()
                    ## make sure that there are no phase slips between blocks
                    diff_pre = (phi_pre[cc]-phi_loop[cc,0]) 
                    # if diff_pre > thresh_between_blocks:
                    #     phi_loop[cc,in_index] += torch.pi/2
                    # elif diff_pre < -thresh_between_blocks:
                    #     phi_loop[cc,in_index] -= torch.pi/2
                    if torch.abs(diff_pre+piHalfTensor) < torch.abs(diff_pre):
                        phi_loop[cc,:] -= piHalfTensor
                        y_symb[cc,:] *=  torch.exp(-1j * piHalfTensor)
                    elif torch.abs(diff_pre-piHalfTensor) < torch.abs(diff_pre):
                        phi_loop[cc,:] += piHalfTensor
                        y_symb[cc,:] *=  torch.exp(1j * piHalfTensor)
                    phi_pre[cc] = phi_loop[cc,-1]
            else:
                y_symb = y_symb_temp

            

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            
            loss, var = ELBO_NxN(
                # loss, var = ELBO_DP(
                y[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.butterfly_backward,
                phi=phi_loop,
                p_constellation=self.demapper.p_symbols,
                IQ_separate=self.IQ_separate,
            )

            if self.use_corr_avoid == True:
                #corr_loss = torch.zeros_like(loss)
                for kk in range(y.shape[0]):             # calculate correlation of each channel to all others
                    ind_corr = torch.arange(y.shape[0])
                    ind_corr = ind_corr[ind_corr!=kk]
                    loss += self.alpha_corr*torch.sum(torch.abs(
                        torch.conv1d(
                            y_symb[kk,:].unsqueeze(0),
                            y_symb[ind_corr,y_symb.shape[-1]-self.corr_window//2:y_symb.shape[-1]+self.corr_window//2].unsqueeze(1),
                            padding='valid',
                        )
                    )**2)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            if self.var_from_estimate == True:
                self.demapper.noise_sigma = torch.clamp(
                    torch.sqrt(torch.mean(var.detach().clone()) / 2),
                    min=torch.tensor(0.05, requires_grad=False, device=q_hat.device),
                    max=2
                    * self.demapper.noise_sigma.detach().clone(),  # torch.sqrt(var).detach()), min=0.1
                )

            # output_symbols = y_symb[
            #     :, : self.block_size
            # ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            # out.append(
            #     output_symbols
            # )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])
            if self.use_cpe == False:
                out.append(
                    y_symb[:, : self.block_size] #_temp[:, : self.block_size]
                ) 
            else:
                out.append(
                    y_symb_temp[:, self.cpe_window_length : self.block_size+self.cpe_window_length]
                ) 
            var_out.append(
                var
            )
##########################
            # phi_full.append(
            #     phi_loop[:,:self.block_size]
            # )
            # ref_full.append(
            #     ref[:,:self.block_size]
            # )
            # phi_ref_full.append(
            #     phi_ref[:,:self.block_size]
            # )
##########################
            output_q = q_hat[:, : self.block_size, :]
            out_q.append(output_q)

        out_const = torch.cat(out, axis=1).detach()#.clone()
        out_const *=  (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])
        var_hat = torch.cat(var_out, axis=-1).view(-1,self.num_channels).detach().mean(0) / 2.0 #.clone()
##########################
        # phi_full_full = torch.cat(phi_full, axis=1).detach()#.clone()
        # phi_ref_full_full = torch.cat(phi_ref_full, axis=1).detach()#.clone()
        # ref_full_full = torch.cat(ref_full, axis=1).detach()#.clone()
        
        # import matplotlib.pyplot as plt
        # plt.plot(phi_ref_full_full[0,:])
        # phi_ref_cumsum = torch.cumsum(phi_ref_full_full[0,:],dim=0)
        # plt.plot(unwrap((phi_ref_cumsum[20:]-phi_ref_cumsum[:-20])/20))
        # plt.plot((phi_ref_cumsum[20:]-phi_ref_cumsum[:-20])/20)
        # plt.plot(-phi_full_full[0,:])
        # plt.plot(-self.cpe(out_const[0,:])[1])
##########################

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(out_const, q_hat, var_hat, loss)
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_var):
        self.demapper.noise_sigma = new_var

##############################################################################################
##############################################################################################
class VAE_LE_NxN_orig(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels=None,
        taps_EQ = None,
        num_taps_EQ=None,
        taps_EST = None,
        num_taps_EST=None,
        demapper=None,
        sps=2,
        block_size=200,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        var_from_estimate=False,
        use_cpe=False,
        use_cpe_in_training=False,
        avg_filter_type="tri",
        use_corr_avoid=False,
        alpha_corr=1e-2,
        corr_window=50,
        cpe_window_length=None,
        cpe_train_window_length=None,
        temperature_per_epoch=None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(VAE_LE_NxN_orig, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        if num_taps_EQ != None:
            self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_EQ))
        if num_taps_EST != None:
            self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_EST))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("var_from_estimate", torch.as_tensor(var_from_estimate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        self.register_buffer("use_cpe_in_training", torch.as_tensor(use_cpe_in_training))
        self.register_buffer("use_corr_avoid", torch.as_tensor(use_corr_avoid))
        self.avg_filter_type=avg_filter_type
        self.register_buffer("alpha_corr", torch.as_tensor(alpha_corr))
        self.register_buffer("corr_window", torch.as_tensor(corr_window))
        if cpe_window_length != None:
            if cpe_train_window_length != None:
                self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
                self.register_buffer("cpe_train_window_length", torch.as_tensor(cpe_train_window_length))
            else:
                self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
                self.register_buffer("cpe_train_window_length", torch.as_tensor(cpe_window_length))
        if temperature_per_epoch != None:
            self.register_buffer("temperature_per_epoch", torch.as_tensor(temperature_per_epoch))
        else:
            self.register_buffer("temperature_per_epoch", torch.as_tensor(1e-2))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))
        self.mode = mode
        ########## Equalizer ############
        if taps_EQ is None:
            if num_taps_EQ is None:
                raise ValueError("Please provide either a valid num_taps_EQ or taps_EQ.")
            else:
                self.butterfly_forward = ButterflyNxN(
                    num_taps=num_taps_EQ,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_forward = ButterflyNxN(
                taps=taps_EQ,
                num_channels=num_channels,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
        ########## Estimator ############
        if taps_EST is None:
            if num_taps_EST is None:
                raise ValueError("Please provide either a valid num_taps_EST or taps_EST.")
            else:
                self.butterfly_backward = ButterflyNxN(
                    num_taps=num_taps_EST,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_backward = ButterflyNxN(
                taps=taps_EST,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
        #######################
    
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

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                ##cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation.squeeze(),
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=self.temperature_per_epoch, #1e-3,
                    no_sectors=4,
                    PS_aware=False,
                    p_symbols=demapper.p_symbols.squeeze(),
                    noise_sigma=demapper.noise_sigma,
                    avg_filter_type=self.avg_filter_type,  #"rect",
                    trainable=False,
                )
        if use_cpe_in_training == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation.squeeze(),
                    self.cpe_train_window_length,
                    diff=True,
                    temperature_per_epoch=self.temperature_per_epoch,
                    no_sectors=4,
                    PS_aware=False,
                    p_symbols=demapper.p_symbols.squeeze(),
                    noise_sigma=demapper.noise_sigma,
                    avg_filter_type=self.avg_filter_type, #"rect",  #"rect","tri"
                    trainable=False,
                )
                # self.cpe = vandv.ViterbiViterbi(
                #     window_length=cpe_window_length
                # )
                # self.cpe_in_loop = vandv.ViterbiViterbi(
                #     window_length=cpe_window_length
                # )

    def init_cpe_for_train(self):
        self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    self.demapper.constellation.squeeze(),
                    self.cpe_train_window_length,
                    diff=True,
                    temperature_per_epoch=self.temperature_per_epoch,
                    no_sectors=4,
                    PS_aware=False,
                    p_symbols=self.demapper.p_symbols.squeeze(),
                    noise_sigma=self.demapper.noise_sigma,
                    avg_filter_type=self.avg_filter_type,  #"rect",
                    trainable=False,
                )

    def reset(self):
        self.lr = self.start_lr.clone()
        # self.butterfly_forward = ButterflyNxN(
        #     num_taps=self.num_taps_forward.item(),
        #     num_channels=self.num_channels,
        #     trainable=True,
        #     timedomain=True,
        #     device=self.butterfly_forward.taps.device,
        #     mode=self.mode,
        # )
        # self.butterfly_backward = ButterflyNxN(
        #     num_taps=self.num_taps_backward.item(),
        #     num_channels=self.num_channels.item(),
        #     trainable=True,
        #     timedomain=True,
        #     device=self.butterfly_forward.taps.device,
        #     mode=self.mode,
        # )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})

    def forward(self, y, x):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size
##########################
        phi_ref_full = []
        phi_full = []
        ref_full = []
##########################
        out = []
        out_q = []
        var_out = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block
##########################
        ref_up = torch.zeros_like(y)
        ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]
##########################
        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        phi_pre = torch.zeros(y.shape[0], device=y.device, dtype=torch.float32)
        piHalfTensor = torch.tensor(torch.pi/2, device=y.device)
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
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,
                k + self.sps * self.block_size + index_padding,
            )
            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y[:, in_index])

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            y_symb_temp = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            y_symb = torch.empty_like(y_symb_temp)  # y_symb_temp #
            phi_loop = torch.zeros_like(y_symb_temp).real #.to(torch.float32)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                (self.butterfly_forward.num_taps - 1)
                // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            ]
##########################
            temp_ref = ref_up[:,y_index]
            ref = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 

            phi_ref = torch.angle(y_symb_temp * ref.conj())
##########################
            if self.use_cpe_in_training == True:
                y_symb_temp1 =  y_symb_temp.clone()*(torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(y_symb_temp)**2, dim=-1))).unsqueeze(1).repeat(1,y_symb_temp.shape[-1])
                for cc in range(self.num_channels):
                    y_symb[cc,:], phi_loop[cc,:] = self.cpe_in_loop(y_symb_temp1[cc,:].clone())  # [0]
                    ## make sure that there are no phase slips between blocks
                    diff_pre = (phi_pre[cc]-phi_loop[cc,0]) 
                    # if diff_pre > thresh_between_blocks:
                    #     phi_loop[cc,in_index] += torch.pi/2
                    # elif diff_pre < -thresh_between_blocks:
                    #     phi_loop[cc,in_index] -= torch.pi/2
                    if torch.abs(diff_pre+piHalfTensor) < torch.abs(diff_pre):
                        phi_loop[cc,:] -= piHalfTensor
                        y_symb[cc,:] *=  torch.exp(-1j * piHalfTensor)
                    elif torch.abs(diff_pre-piHalfTensor) < torch.abs(diff_pre):
                        phi_loop[cc,:] += piHalfTensor
                        y_symb[cc,:] *=  torch.exp(1j * piHalfTensor)
                    phi_pre[cc] = phi_loop[cc,-1]
            else:
                y_symb = y_symb_temp

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            
            loss, var = ELBO_NxN(
                # loss, var = ELBO_DP(
                y[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.butterfly_backward,
                phi=phi_loop,
                p_constellation=self.demapper.p_symbols,
                IQ_separate=self.IQ_separate,
            )

            if self.use_corr_avoid == True:
                #corr_loss = torch.zeros_like(loss)
                for kk in range(y.shape[0]):             # calculate correlation of each channel to all others
                    ind_corr = torch.arange(y.shape[0])
                    ind_corr = ind_corr[ind_corr!=kk]
                    loss += self.alpha_corr*torch.sum(torch.abs(
                        torch.conv1d(
                            y_symb[kk,:].unsqueeze(0),
                            y_symb[ind_corr,y_symb.shape[-1]-self.corr_window//2:y_symb.shape[-1]+self.corr_window//2].unsqueeze(1),
                            padding='valid',
                        )
                    )**2)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            if self.var_from_estimate == True:
                self.demapper.noise_sigma = torch.clamp(
                    torch.sqrt(torch.mean(var.detach().clone()) / 2),
                    min=torch.tensor(0.05, requires_grad=False, device=q_hat.device),
                    max=2
                    * self.demapper.noise_sigma.detach().clone(),  # torch.sqrt(var).detach()), min=0.1
                )

            # output_symbols = y_symb[
            #     :, : self.block_size
            # ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            # out.append(
            #     output_symbols
            # )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])
            out.append(
                y_symb_temp[:, : self.block_size]
            ) 
            var_out.append(
                var
            )
##########################
            phi_full.append(
                phi_loop[:,:self.block_size]
            )
            ref_full.append(
                ref[:,:self.block_size]
            )
            phi_ref_full.append(
                phi_ref[:,:self.block_size]
            )
##########################
            output_q = q_hat[:, : self.block_size, :]
            out_q.append(output_q)

        out_const = torch.cat(out, axis=1).detach()#.clone()
        out_const *=  (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])
        var_hat = torch.cat(var_out, axis=-1).view(-1,self.num_channels).detach().mean(0) / 2.0 #.clone()
##########################
        phi_full_full = torch.cat(phi_full, axis=1).detach()#.clone()
        phi_ref_full_full = torch.cat(phi_ref_full, axis=1).detach()#.clone()
        ref_full_full = torch.cat(ref_full, axis=1).detach()#.clone()
        
        # import matplotlib.pyplot as plt
        # plt.plot(phi_ref_full_full[0,:])
        # phi_ref_cumsum = torch.cumsum(phi_ref_full_full[0,:],dim=0)
        # plt.plot(unwrap((phi_ref_cumsum[20:]-phi_ref_cumsum[:-20])/20))
        # plt.plot((phi_ref_cumsum[20:]-phi_ref_cumsum[:-20])/20)
        # plt.plot(-phi_full_full[0,:])
        # plt.plot(-self.cpe(out_const[0,:])[1])
##########################

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(out_const, q_hat, var_hat, loss)
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_var):
        self.demapper.noise_sigma = new_var

##############################################################################################
##############################################################################################

class VAE_LE_flex_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels=None,
        taps_EQ = None,
        num_taps_EQ=None,
        taps_EST = None,
        num_taps_EST=None,
        demapper=None,
        sps=2,
        block_size=200,
        symb_per_step = 10,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        var_from_estimate=False,
        use_corr_avoid=False,
        alpha_corr=1e-2,
        corr_window=50,
        use_cpe=False,
        use_cpe_in_training=False,
        cpe_window_length = None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(VAE_LE_flex_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("symb_per_step", torch.as_tensor(symb_per_step))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        if num_taps_EQ != None:
            self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_EQ))
        if num_taps_EST != None:
            self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_EST))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("var_from_estimate", torch.as_tensor(var_from_estimate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        self.register_buffer("use_cpe_in_training", torch.as_tensor(use_cpe_in_training))
        if cpe_window_length != None:
            self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))
        self.register_buffer("use_corr_avoid", torch.as_tensor(use_corr_avoid))
        self.register_buffer("alpha_corr", torch.as_tensor(alpha_corr))
        self.register_buffer("corr_window", torch.as_tensor(corr_window))

        self.register_buffer("N_phi_ave", torch.as_tensor(32))      # 50 at code of IPQ
        self.mode = mode
        
        # f = scipy.io.loadmat("waveforms/data_from_IPQ/weights.mat")
        # weigh = f.get('wk_cma')
        # taps = torch.zeros(8,8,151, dtype=torch.complex64)
        # taps[:,:,1:] = torch.from_numpy(weigh.reshape(8,8,150)).permute(1,0,2)

        ########## Equalizer ############
        if taps_EQ is None:
            if num_taps_EQ is None:
                raise ValueError("Please provide either a valid num_taps_EQ or taps_EQ.")
            else:
                self.butterfly_forward = ButterflyNxN(
                    num_taps=num_taps_EQ,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
                self.pad_func = torch.nn.ConstantPad1d((num_taps_EQ - 1) // 2, 0)
        else:
            self.butterfly_forward = ButterflyNxN(
                taps=taps_EQ,
                num_channels=num_channels,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
            self.pad_func = torch.nn.ConstantPad1d((taps_EQ.shape[-1] - 1) // 2, 0)
        ########## Estimator ############
        if taps_EST is None:
            if num_taps_EST is None:
                raise ValueError("Please provide either a valid num_taps_EST or taps_EST.")
            else:
                self.butterfly_backward = ButterflyNxN(
                    num_taps=num_taps_EST,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_backward = ButterflyNxN(
                taps=taps_EST,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
        #######################
        taps_ma = torch.zeros(self.num_channels,self.num_channels,self.N_phi_ave,dtype=torch.complex64, device=device)
        for ma_idx in range(self.num_channels):
            taps_ma[ma_idx,ma_idx,:] = torch.ones(self.N_phi_ave,dtype=torch.complex64, device=device)
        self.moving_ave = ButterflyNxN(
            taps=taps_ma,
            #num_taps=num_taps,
            num_channels=num_channels,
            trainable=False,
            timedomain=True,
            device=device,
            mode="valid",
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

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                #cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=1e-2,
                    no_sectors=4,
                    avg_filter_type="tri", #"rect",
                    trainable=False,
                )
        if use_cpe_in_training == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=True,
                    temperature_per_epoch=1e-2,
                    no_sectors=4,
                    avg_filter_type="tri", #"rect",
                    trainable=False,
                )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )

    def init_cpe_for_train(self):
        self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    self.demapper.constellation.squeeze(),
                    self.cpe_window_length,
                    diff=True,
                    temperature_per_epoch=1e-2,
                    no_sectors=4,
                    avg_filter_type="tri",  #"rect",
                    trainable=False,
                )
        
    def reset(self):
        self.lr = self.start_lr.clone()
        # self.butterfly_forward = ButterflyNxN(
        #     num_taps=self.num_taps.item(),
        #     num_channels=self.num_channels,
        #     trainable=True,
        #     timedomain=True,
        #     device=self.butterfly_forward.taps.device,
        #     mode=self.mode,
        # )
        # self.optimizer = torch.optim.Adam(
        #     self.butterfly_forward.parameters(),
        #     lr=self.lr,
        # )

    def forward(self, y):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        assert self.N_phi_ave < self.butterfly_forward.num_taps     # TODO: implement for both cases

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        var_out=[]
        phi_plot = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        # ref_up = torch.zeros_like(y)
        # ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]

        mod_symb_per_step_2 = self.symb_per_step%2
        index_padding = (self.butterfly_forward.num_taps - 1) // 2
        temp_eqIn = torch.zeros(self.N_phi_ave, y.shape[0], self.sps * self.block_size + 2*index_padding, dtype=y.dtype, device=y.device)     # shift register with dim: N_phi_ave, num. of channels, block_size*sps
        # temp_phi = torch.zeros(self.N_phi_ave, dtype=y.dtype, device=y.device)
        #temp_ref = torch.zeros(self.N_phi_ave, y.shape[0], self.block_size, dtype=y.dtype, device=y.device)
        
        y_pad = self.pad_func(y)
        #ref_pad = self.pad_func(x)
        # # init ref matrix
        # temp_eqIn[self.N_phi_ave//2,:,:] = y[:, 0 : self.sps * self.block_size + 2*index_padding]
        # for jj in range(0,self.N_phi_ave//2):
        #     temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y[:, jj+1 : jj+1 + self.sps * self.block_size + 2*index_padding]  # later in time
        #     temp_eqIn[self.N_phi_ave//2 + jj,:,jj:] = y[:, : -jj + self.sps * self.block_size + 2*index_padding]  # earlier in time

        for i, k in enumerate(
            range(
                index_padding,
                num_samps
                - index_padding
                - self.sps
                * self.block_size,  # Back-off one block-size + filter_overlap from end to avoid overrunning
                self.sps * self.symb_per_step,
            )
        ):
            # if i % (20000//self.block_size) == 0 and i != 0:
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k- index_padding,   # padding already included in y_pad
                k + self.sps * self.block_size + index_padding,   # times two since padding already included in y_pad
            )
            # ref_index = torch.arange(
            #     i * self.symb_per_step + (index_padding+1)//2 + index_padding,  
            #     i * self.symb_per_step + (index_padding+1)//2 + index_padding + self.block_size,   
            # )

            # for jj in range(0,self.N_phi_ave//2):
            #     temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y_pad[:, self.sps*jj+1+in_index]  # later in time
            #     temp_eqIn[self.N_phi_ave//2 + jj,:,:] = y_pad[:, in_index - self.sps*jj]  # earlier in time
            #     #temp_ref[self.N_phi_ave//2 - jj-1,:,:] = ref_pad[:, jj+1+ref_index]  # later in time
            #     #temp_ref[self.N_phi_ave//2 + jj,:, :] = ref_pad[:, ref_index - jj] # earlier in time

            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y_pad[:, in_index])
            # y_hat = self.butterfly_forward(temp_eqIn)[ : , : , 0 :: self.sps ]

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            # y_symb_temp = y_hat[
            #     self.N_phi_ave//2, :, :  #0 :: self.sps
            # ].squeeze()  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]
            # y_symb = torch.empty_like(y_symb_temp)
            y_symb_temp = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            y_symb = torch.empty_like(y_symb_temp)  # y_symb_temp #
            phi_loop = torch.zeros_like(y_symb_temp).real #to(torch.float32)

            if self.use_cpe_in_training == True:
                for cc in range(self.num_channels):
                    y_symb[cc,:], phi_loop[cc,:] = self.cpe_in_loop(y_symb_temp[cc,:].clone())  # [0]
            else:
                y_symb = y_symb_temp

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                index_padding : -index_padding
            ]
            loss, var = ELBO_NxN(
                # loss, var = ELBO_DP(
                y_pad[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.butterfly_backward,
                phi=phi_loop,
                p_constellation=self.demapper.p_symbols,
                IQ_separate=self.IQ_separate,
            )

            if self.use_corr_avoid == True:
                #corr_loss = torch.zeros_like(loss)
                for kk in range(y.shape[0]):             # calculate correlation of each channel to all others
                    ind_corr = torch.arange(y.shape[0])
                    ind_corr = ind_corr[ind_corr!=kk]
                    loss += self.alpha_corr*torch.sum(torch.abs(
                        torch.conv1d(
                            y_symb[kk,:].unsqueeze(0),
                            y_symb[ind_corr,y_symb.shape[-1]-self.corr_window//2:y_symb.shape[-1]+self.corr_window//2].unsqueeze(1),
                            padding='valid',
                        )
                    )**2)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            if self.var_from_estimate == True:
                self.demapper.noise_sigma = torch.clamp(
                    torch.sqrt(torch.mean(var.detach().clone()) / 2),
                    min=torch.tensor(0.05, requires_grad=False, device=q_hat.device),
                    max=2
                    * self.demapper.noise_sigma.detach().clone(),  # torch.sqrt(var).detach()), min=0.1
                )



            # # y_index = in_index[
            # #     (self.butterfly_forward.num_taps - 1)
            # #     // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            # # ]
            # # temp_symb = temp_symb.roll(shifts=(1), dims=(0))
            # # #ref_index = in_index[ (self.butterfly_forward.num_taps - 1) : ] + (self.butterfly_forward.num_taps + 1)*self.sps + 86
            # # temp_ref = ref_up[:,y_index]
            # # temp_symb[0,:,:] = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 
            # # ref = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 
            # ref = temp_ref[self.N_phi_ave//2, :, :]
            
            # if self.use_cpe == True:
            #     # N_phi_ave = self.N_phi_ave
            #     phi_off = (torch.angle(y_hat) - torch.angle(temp_ref)).detach() # unwrap necessary?
            #     # y_phi_sum = torch.zeros_like(y_symb)[:,:-N_phi_ave]
            #     # self.moving_ave(torch.exp(1j*phi_off[16,:,:]))
            #     # for ii in range(N_phi_ave):
            #     #     y_phi_sum += torch.exp(1j*phi_off[:,ii:-(N_phi_ave-ii)])    # sum in the complex domain
            #     phi_ma = torch.angle(torch.sum(torch.exp(1j*phi_off) , dim=0) )
            #     ref_loss = ref *torch.exp(1j*phi_ma) 
            #     loss = self.loss_func( y_symb.real, ref_loss.real ) + self.loss_func( y_symb.imag, ref_loss.imag)
            #     # for cc in range(self.num_channels):
            #     #     ref[cc,:] *= torch.exp(1j*self.cpe(y_symb[cc,:])[1])    # rotate reference to make update independet of phase noise
            #     # loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)
            #     phi_plot.append(phi_ma)
            # else:
            #     loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)

            # # print("noise_sigma: ", self.demapper.noise_sigma)
            # loss.backward()
            # self.optimizer.step()
            # # self.optimizer_var.step()
            # self.optimizer.zero_grad()
            # # self.optimizer_var.zero_grad()

            # what about odd symbols_per_step?
            if i==0:
                ind_out = torch.arange( 0, (self.block_size+self.symb_per_step)//2 ) 
            else:
                ind_out = torch.arange( (self.block_size-self.symb_per_step)//2, (self.block_size+self.symb_per_step)//2 ) + mod_symb_per_step_2
            
            output_symbols =  y_symb_temp[ #y_symb[
                :, ind_out
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])
            var_out.append(
                var
            )

            output_q = q_hat[:, ind_out, :]
            out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        ###################
        var_hat = torch.cat(var_out, axis=-1).view(-1,self.num_channels).detach().mean(0) / 2.0 #.clone()
        out_const = torch.cat(out, axis=1).detach().clone()
        out_const *= (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(out_const, q_hat, var_hat, loss)
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################

class VAE_LE_overhead_NxN(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_channels=None,
        taps_EQ = None,
        num_taps_EQ=None,
        taps_EST = None,
        num_taps_EST=None,
        demapper=None,
        sps=2,
        block_size=200,
        symb_per_step = 10,
        lr=0.5e-2,
        requires_q=False,
        IQ_separate=False,
        var_from_estimate=False,
        use_corr_avoid=False,
        alpha_corr=1e-2,
        corr_window=50,
        use_cpe=False,
        use_cpe_in_training=False,
        cpe_window_length = None,
        angles_per_quadrant=None,
        device="cpu",
        mode="valid",
    ):
        super(VAE_LE_overhead_NxN, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("symb_per_step", torch.as_tensor(symb_per_step))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("num_channels", torch.as_tensor(num_channels))
        if num_taps_EQ != None:
            self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_EQ))
        if num_taps_EST != None:
            self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_EST))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.register_buffer("var_from_estimate", torch.as_tensor(var_from_estimate))
        self.register_buffer("use_cpe", torch.as_tensor(use_cpe))
        self.register_buffer("use_cpe_in_training", torch.as_tensor(use_cpe_in_training))
        if cpe_window_length != None:
            self.register_buffer("cpe_window_length", torch.as_tensor(cpe_window_length))
        if angles_per_quadrant != None:
            self.register_buffer("angles_per_quadrant", torch.as_tensor(angles_per_quadrant))
        self.register_buffer("use_corr_avoid", torch.as_tensor(use_corr_avoid))
        self.register_buffer("alpha_corr", torch.as_tensor(alpha_corr))
        self.register_buffer("corr_window", torch.as_tensor(corr_window))

        self.register_buffer("N_phi_ave", torch.as_tensor(32))      # 50 at code of IPQ
        self.mode = mode
        
        # f = scipy.io.loadmat("waveforms/data_from_IPQ/weights.mat")
        # weigh = f.get('wk_cma')
        # taps = torch.zeros(8,8,151, dtype=torch.complex64)
        # taps[:,:,1:] = torch.from_numpy(weigh.reshape(8,8,150)).permute(1,0,2)

        ########## Equalizer ############
        if taps_EQ is None:
            if num_taps_EQ is None:
                raise ValueError("Please provide either a valid num_taps_EQ or taps_EQ.")
            else:
                self.butterfly_forward = ButterflyNxN(
                    num_taps=num_taps_EQ,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
                self.pad_func = torch.nn.ConstantPad1d((num_taps_EQ - 1) // 2, 0)
        else:
            self.butterfly_forward = ButterflyNxN(
                taps=taps_EQ,
                num_channels=num_channels,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
            self.pad_func = torch.nn.ConstantPad1d((taps_EQ.shape[-1] - 1) // 2 + cpe_window_length*sps, 0)
        ########## Estimator ############
        if taps_EST is None:
            if num_taps_EST is None:
                raise ValueError("Please provide either a valid num_taps_EST or taps_EST.")
            else:
                self.butterfly_backward = ButterflyNxN(
                    num_taps=num_taps_EST,
                    num_channels=num_channels,
                    trainable=True,
                    timedomain=True,
                    device=device,
                    mode=mode,
                )
        else:
            self.butterfly_backward = ButterflyNxN(
                taps=taps_EST,
                trainable=True,
                timedomain=True,
                device=device,
                mode=mode,
            )
        #######################
        taps_ma = torch.zeros(self.num_channels,self.num_channels,self.N_phi_ave,dtype=torch.complex64, device=device)
        for ma_idx in range(self.num_channels):
            taps_ma[ma_idx,ma_idx,:] = torch.ones(self.N_phi_ave,dtype=torch.complex64, device=device)
        self.moving_ave = ButterflyNxN(
            taps=taps_ma,
            #num_taps=num_taps,
            num_channels=num_channels,
            trainable=False,
            timedomain=True,
            device=device,
            mode="valid",
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

        if use_cpe == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                #cpe_window_length = 50
                self.cpe = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=False,
                    temperature_per_epoch=1e-2,
                    no_sectors=4,
                    avg_filter_type="tri", #"rect",
                    trainable=False,
                )
        if use_cpe_in_training == True:
            if cpe_window_length is None or angles_per_quadrant is None:
                raise ValueError("Please provide cpe_window_length and angles_per_quadrant when using CPE.")
            else:
                self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    demapper.constellation,
                    self.cpe_window_length,
                    diff=True,
                    temperature_per_epoch=1e-2,
                    no_sectors=4,
                    avg_filter_type="tri", #"rect",
                    trainable=False,
                )
        # self.cpe = vandv.ViterbiViterbi(
        #                 window_length=cpe_window_length
        #             )

    def init_cpe_for_train(self):
        self.cpe_in_loop = BPS(
                    self.angles_per_quadrant,
                    self.demapper.constellation.squeeze(),
                    self.cpe_window_length,
                    diff=True,
                    temperature_per_epoch=1e-2,
                    no_sectors=4,
                    avg_filter_type="tri",  #"rect",
                    trainable=False,
                )
        
    def reset(self):
        self.lr = self.start_lr.clone()
        # self.butterfly_forward = ButterflyNxN(
        #     num_taps=self.num_taps.item(),
        #     num_channels=self.num_channels,
        #     trainable=True,
        #     timedomain=True,
        #     device=self.butterfly_forward.taps.device,
        #     mode=self.mode,
        # )
        # self.optimizer = torch.optim.Adam(
        #     self.butterfly_forward.parameters(),
        #     lr=self.lr,
        # )

    def forward(self, y):
        # We need to produce enough q values on each forward pass such that we can
        # calculate the ELBO loss in the backward pass & update the taps

        #assert self.N_phi_ave < self.butterfly_forward.num_taps     # TODO: implement for both cases

        num_samps = y.shape[1]
        # samples_per_step = self.butterfly_forward.num_taps + self.block_size

        out = []
        out_q = []
        var_out=[]
        phi_plot = []
        # We start our loop already at num_taps  (because we cannot equalize the start)
        # We will end the loop at num_samps - num_taps - sps*block_size (safety, so we don't overrun)
        # We will process sps * block_size - 2 * num_taps because we will cut out the first and last block

        # ref_up = torch.zeros_like(y)
        # ref_up[:,::self.sps] = x[:,:y.shape[-1]//self.sps]

        mod_symb_per_step_2 = self.symb_per_step%2
        index_padding = (self.butterfly_forward.num_taps - 1) // 2 + self.cpe_window_length*self.sps
        temp_eqIn = torch.zeros(self.N_phi_ave, y.shape[0], self.sps * self.block_size + 2*index_padding, dtype=y.dtype, device=y.device)     # shift register with dim: N_phi_ave, num. of channels, block_size*sps
        # temp_phi = torch.zeros(self.N_phi_ave, dtype=y.dtype, device=y.device)
        #temp_ref = torch.zeros(self.N_phi_ave, y.shape[0], self.block_size, dtype=y.dtype, device=y.device)
        
        y_pad = y #self.pad_func(y)
        #ref_pad = self.pad_func(x)
        # # init ref matrix
        # temp_eqIn[self.N_phi_ave//2,:,:] = y[:, 0 : self.sps * self.block_size + 2*index_padding]
        # for jj in range(0,self.N_phi_ave//2):
        #     temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y[:, jj+1 : jj+1 + self.sps * self.block_size + 2*index_padding]  # later in time
        #     temp_eqIn[self.N_phi_ave//2 + jj,:,jj:] = y[:, : -jj + self.sps * self.block_size + 2*index_padding]  # earlier in time

        for i, k in enumerate(
            range(
                index_padding,
                num_samps
                - index_padding
                - self.sps
                * self.block_size,  # Back-off one block-size + filter_overlap from end to avoid overrunning
                self.sps * self.symb_per_step,
            )
        ):
            # if i % (20000//self.block_size) == 0 and i != 0:
            # print("Updating learning rate")
            # self.update_lr(self.lr * 0.5)
            # logger.debug("VAE LE block: %s", i)
            in_index = torch.arange(
                k - index_padding,   # padding already included in y_pad
                k + self.sps * self.block_size + index_padding,   # times two since padding already included in y_pad
            )
            # ref_index = torch.arange(
            #     i * self.symb_per_step + (index_padding+1)//2 + index_padding,  
            #     i * self.symb_per_step + (index_padding+1)//2 + index_padding + self.block_size,   
            # )

            # for jj in range(0,self.N_phi_ave//2):
            #     temp_eqIn[self.N_phi_ave//2 - jj-1,:,:] = y_pad[:, self.sps*jj+1+in_index]  # later in time
            #     temp_eqIn[self.N_phi_ave//2 + jj,:,:] = y_pad[:, in_index - self.sps*jj]  # earlier in time
            #     #temp_ref[self.N_phi_ave//2 - jj-1,:,:] = ref_pad[:, jj+1+ref_index]  # later in time
            #     #temp_ref[self.N_phi_ave//2 + jj,:, :] = ref_pad[:, ref_index - jj] # earlier in time

            # Equalization will give sps * block_size samples (because we add (num_taps - 1) in the beginning)
            y_hat = self.butterfly_forward(y_pad[:, in_index])
            # y_hat = self.butterfly_forward(temp_eqIn)[ : , : , 0 :: self.sps ]

            # We downsample so we will have floor(((sps * block_size - num_taps + 1) / sps) = floor(block_size - (num_taps - 1)/sps)
            # y_symb_temp = y_hat[
            #     self.N_phi_ave//2, :, :  #0 :: self.sps
            # ].squeeze()  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]
            # y_symb = torch.empty_like(y_symb_temp)
            y_symb_temp = y_hat[
                :, 0 :: self.sps
            ]  # ---> y[0,(self.butterfly_forward.num_taps + 1)//2 +1 ::self.sps]

            y_symb = torch.empty_like(y_symb_temp)  # y_symb_temp #
            phi_loop = torch.zeros_like(y_symb_temp).to(torch.float32)

            if self.use_cpe_in_training == True:
                for cc in range(self.num_channels):
                    y_symb[cc,:], phi_loop[cc,:] = self.cpe_in_loop(y_symb_temp[cc,:].clone())  # [0]
            else:
                y_symb = y_symb_temp

            y_symb = y_symb[:, self.cpe_window_length : -self.cpe_window_length ]
            phi_loop = phi_loop[:, self.cpe_window_length : -self.cpe_window_length ]

            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    2 * num_level,
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        y_symb[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        y_symb[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    y_symb.shape[0],
                    y_symb.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=y_symb.device,
                    dtype=torch.float32,
                )
                for out_chan in range(y_symb.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        y_symb[out_chan, :]
                    ).unsqueeze(0)

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                index_padding : -index_padding
            ]
            loss, var = ELBO_NxN(
                # loss, var = ELBO_DP(
                y_pad[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.butterfly_backward,
                phi=phi_loop,
                p_constellation=self.demapper.p_symbols,
                IQ_separate=self.IQ_separate,
            )

            if self.use_corr_avoid == True:
                #corr_loss = torch.zeros_like(loss)
                for kk in range(y.shape[0]):             # calculate correlation of each channel to all others
                    ind_corr = torch.arange(y.shape[0])
                    ind_corr = ind_corr[ind_corr!=kk]
                    loss += self.alpha_corr*torch.sum(torch.abs(
                        torch.conv1d(
                            y_symb[kk,:].unsqueeze(0),
                            y_symb[ind_corr,y_symb.shape[-1]-self.corr_window//2:y_symb.shape[-1]+self.corr_window//2].unsqueeze(1),
                            padding='valid',
                        )
                    )**2)

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            if self.var_from_estimate == True:
                self.demapper.noise_sigma = torch.clamp(
                    torch.sqrt(torch.mean(var.detach().clone()) / 2),
                    min=torch.tensor(0.05, requires_grad=False, device=q_hat.device),
                    max=2
                    * self.demapper.noise_sigma.detach().clone(),  # torch.sqrt(var).detach()), min=0.1
                )



            # # y_index = in_index[
            # #     (self.butterfly_forward.num_taps - 1)
            # #     // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            # # ]
            # # temp_symb = temp_symb.roll(shifts=(1), dims=(0))
            # # #ref_index = in_index[ (self.butterfly_forward.num_taps - 1) : ] + (self.butterfly_forward.num_taps + 1)*self.sps + 86
            # # temp_ref = ref_up[:,y_index]
            # # temp_symb[0,:,:] = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 
            # # ref = temp_ref[temp_ref.nonzero(as_tuple=True)].reshape(self.num_channels,-1)   # .nonzero() downsamples to 1sps again 
            # ref = temp_ref[self.N_phi_ave//2, :, :]
            
            # if self.use_cpe == True:
            #     # N_phi_ave = self.N_phi_ave
            #     phi_off = (torch.angle(y_hat) - torch.angle(temp_ref)).detach() # unwrap necessary?
            #     # y_phi_sum = torch.zeros_like(y_symb)[:,:-N_phi_ave]
            #     # self.moving_ave(torch.exp(1j*phi_off[16,:,:]))
            #     # for ii in range(N_phi_ave):
            #     #     y_phi_sum += torch.exp(1j*phi_off[:,ii:-(N_phi_ave-ii)])    # sum in the complex domain
            #     phi_ma = torch.angle(torch.sum(torch.exp(1j*phi_off) , dim=0) )
            #     ref_loss = ref *torch.exp(1j*phi_ma) 
            #     loss = self.loss_func( y_symb.real, ref_loss.real ) + self.loss_func( y_symb.imag, ref_loss.imag)
            #     # for cc in range(self.num_channels):
            #     #     ref[cc,:] *= torch.exp(1j*self.cpe(y_symb[cc,:])[1])    # rotate reference to make update independet of phase noise
            #     # loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)
            #     phi_plot.append(phi_ma)
            # else:
            #     loss = self.loss_func( y_symb.real, ref.real ) + self.loss_func( y_symb.imag, ref.imag)

            # # print("noise_sigma: ", self.demapper.noise_sigma)
            # loss.backward()
            # self.optimizer.step()
            # # self.optimizer_var.step()
            # self.optimizer.zero_grad()
            # # self.optimizer_var.zero_grad()

            # what about odd symbols_per_step?
            if i==0:
                ind_out = torch.arange( 0, (self.block_size+self.symb_per_step)//2 ) 
            else:
                ind_out = torch.arange( (self.block_size-self.symb_per_step)//2, (self.block_size+self.symb_per_step)//2 ) + mod_symb_per_step_2
            
            output_symbols =  y_symb[ #y_symb_temp[ #y_symb[
                :, ind_out
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])
            var_out.append(
                var
            )

            output_q = q_hat[:, ind_out, :]
            out_q.append(output_q)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        ###################

        out_const = torch.cat(out, axis=1).detach().clone()
        out_const *= (torch.sqrt(torch.sum(self.demapper.p_symbols*torch.abs(self.demapper.constellation)**2)) / torch.sqrt(torch.mean(torch.abs(out_const)**2, dim=-1))).unsqueeze(1).repeat(1,out_const.shape[-1])
        var_hat = torch.cat(var_out, axis=-1).view(-1,self.num_channels).detach().mean(0) / 2.0 #.clone()

        if self.use_cpe == True:
            for cc in range(self.num_channels):
                out_const[cc,:] = self.cpe(out_const[cc,:])[0]
            if self.IQ_separate == True:
                num_level = self.demapper.constellation.shape[-1]
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    2 * num_level,
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :num_level] = self.demapper(
                        out_const[out_chan, :].real
                    ).unsqueeze(0)
                    q_hat[out_chan, :, num_level:] = self.demapper(
                        out_const[out_chan, :].imag
                    ).unsqueeze(0)
            else:
                q_hat = torch.zeros(
                    out_const.shape[0],
                    out_const.shape[1],
                    self.demapper.constellation.shape[-1],
                    device=out_const.device,
                    dtype=torch.float32,
                )
                for out_chan in range(out_const.shape[0]):
                    q_hat[out_chan, :, :] = self.demapper(
                        out_const[out_chan, :]
                    ).unsqueeze(0)
        else:
            q_hat = torch.cat(out_q, axis=1)

        # print("loss: ", loss, "\t\t\t var: ", var)
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(out_const, q_hat, var_hat, loss)
        return out_const

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


##############################################################################################
##############################################################################################

 ##############################################################################################
##############################################################################################


def VQVAE_loss_DP(
    y,
    q,
    eq,
    beta,
    sps,
    constellation_symbols,
    butterfly_filter,
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

    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2

    if IQ_separate == True:
        num_lev = constellation_symbols.shape[0]
        dec = torch.complex(
            constellation_symbols[torch.argmax(q[:, :, :num_lev], dim=-1)],
            constellation_symbols[torch.argmax(q[:, :, num_lev:], dim=-1)],
        )
    else:
        dec = constellation_symbols[torch.argmax(q, axis=-1)]

    temp_dec = torch.zeros(2, N, device=q.device, dtype=torch.complex64)
    temp_dec[:, ::sps] = eq + (dec - eq).detach()  # straigth-through gradient

    # Precompute h \ast E_Q{x}
    Est = butterfly_filter(
        temp_dec, mode="valid"
    )  # Due to definition that we assume the symbol is at the center tap we remove (filter_length - 1)//2 at start and end
    # Limit the length of y to the "computable space" because y depends on more past values than given
    # We try to generate the received symbol sequence with the estimated symbol sequence
    recon_loss = torch.mean(torch.abs(y[:, L_offset:-L_offset] - Est) ** 2)
    commit_loss = torch.mean(torch.abs(dec - eq) ** 2)

    loss = beta / 100 * commit_loss + (100 - beta) / 100 * recon_loss
    return loss


def VQVAE_loss_DP_hard(
    y,
    eq,
    beta,
    sps,
    constellation_symbols,
    bounds,
    butterfly_filter,
    IQ_separate=False,
):
    """
    Calculate dual-pol. ELBO loss for I/Q symmetric constellations.

    It splits the signal into in-phase/ quadrature parts and process them separately.
    It uses hard decision with decision boundaries adapted to prior distribution (p(x)).
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

    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2

    if IQ_separate == True:
        num_lev = constellation_symbols.shape[0]
        dec = torch.complex(
            constellation_symbols[torch.argmax(q[:, :, :num_lev], dim=-1)],
            constellation_symbols[torch.argmax(q[:, :, num_lev:], dim=-1)],
        )
    else:
        dec = constellation_symbols[torch.argmax(q, axis=-1)]

    temp_dec = torch.zeros(2, N, device=q.device, dtype=torch.complex64)
    temp_dec[:, ::sps] = eq + (dec - eq).detach()  # straigth-through gradient

    # Precompute h \ast E_Q{x}
    Est = butterfly_filter(
        temp_dec, mode="valid"
    )  # Due to definition that we assume the symbol is at the center tap we remove (filter_length - 1)//2 at start and end
    # Limit the length of y to the "computable space" because y depends on more past values than given
    # We try to generate the received symbol sequence with the estimated symbol sequence
    recon_loss = torch.mean(torch.abs(y[:, L_offset:-L_offset] - Est) ** 2)
    commit_loss = torch.mean(torch.abs(dec - eq) ** 2)

    loss = beta / 100 * commit_loss + (100 - beta) / 100 * recon_loss
    return loss


##############################################################################################


class VQVAE_LE_DP(torch.nn.Module):
    """
    Class that can be dropped in to perform equalization
    """

    def __init__(
        self,
        num_taps_forward,
        num_taps_backward,
        demapper,
        sps,
        block_size=200,
        lr=0.5e-2,
        beta=30,
        requires_q=False,
        IQ_separate=False,
        device="cpu",
    ):
        super(VQVAE_LE_DP, self).__init__()

        self.register_buffer("block_size", torch.as_tensor(block_size))
        self.register_buffer("sps", torch.as_tensor(sps))
        self.register_buffer("start_lr", torch.as_tensor(lr))
        self.register_buffer("lr", torch.as_tensor(lr))
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("num_taps_forward", torch.as_tensor(num_taps_forward))
        self.register_buffer("num_taps_backward", torch.as_tensor(num_taps_backward))
        self.register_buffer("requires_q", torch.as_tensor(requires_q))
        self.register_buffer("IQ_separate", torch.as_tensor(IQ_separate))
        self.butterfly_forward = Butterfly2x2(
            num_taps=num_taps_forward, trainable=True, timedomain=True, device=device
        )
        self.butterfly_backward = Butterfly2x2(
            num_taps=num_taps_backward, trainable=True, timedomain=True, device=device
        )
        self.demapper = demapper
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.start_lr,  # 0.5e-2,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})
        # self.noise_sigma = self.demapper.noise_sigma
        # self.demapper.noise_sigma.retain_grad = True
        # self.optimizer.add_param_group({"params": self.demapper.noise_sigma})

        # self.optimizer_var = torch.optim.Adam(
        #     [self.demapper.noise_sigma],
        #     lr=0.5,  # 0.5e-2,
        # )

    def reset(self):
        self.lr = self.start_lr.clone()
        self.butterfly_forward = Butterfly2x2(
            num_taps=self.num_taps_forward.item(),
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
        )
        self.butterfly_backward = Butterfly2x2(
            num_taps=self.num_taps_backward.item(),
            trainable=True,
            timedomain=True,
            device=self.butterfly_forward.taps.device,
        )
        self.optimizer = torch.optim.Adam(
            self.butterfly_forward.parameters(),
            lr=self.lr,
        )
        self.optimizer.add_param_group({"params": self.butterfly_backward.parameters()})
        # self.optimizer.add_param_group({"params": self.noise_sigma})
        # self.demapper.noise_sigma.retain_grad = True
        # self.optimizer.add_param_group({"params": self.demapper.noise_sigma})

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
            # print("Updating learning rate")
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
            loss = VQVAE_loss_DP(
                # loss, var = ELBO_DP(
                y[:, y_index],
                q_hat,
                y_symb,
                self.beta,
                self.sps,
                self.demapper.constellation,
                self.butterfly_backward,
                IQ_separate=self.IQ_separate,
            )

            # print("noise_sigma: ", self.demapper.noise_sigma)
            # self.demapper.noise_sigma.retain_grad()
            loss.backward()
            self.optimizer.step()
            # self.optimizer_var.step()
            self.optimizer.zero_grad()
            # self.optimizer_var.zero_grad()

            # self.demapper.noise_sigma = self.noise_sigma

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
        # out.append(y_symb[:, self.block_size - self.butterfly_forward.num_taps // 2 :])

        if self.requires_q == True:
            eq_out = namedtuple("eq_out", ["y", "q", "loss"])
            return eq_out(torch.cat(out, axis=1), torch.cat(out_q, axis=1), loss)
        return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_var):
        self.demapper.noise_sigma = new_var

    def update_beta(self, new_beta):
        self.beta = new_beta


##############################################################################################


def hard_decision_shaping(rx, tx, amp_levels, nu_sc, var):
    """
    Function to perform (hard) decision for PCS formats
    """
    # estimate symbol error rate from output constellation by considering PCS
    device = rx.device
    num_lev = amp_levels.shape[0]
    data = torch.empty_like(tx, device=device, dtype=torch.int32)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2, 2, 4, device=device, dtype=torch.float32)

    # calculate decision boundaries based on PCS
    d_vec = (1 + 2 * nu_sc * var[0]) * (amp_levels[:-1] + amp_levels[1:]) / 2
    d_vec0 = torch.cat(((-Inf * torch.ones(1, device=device)), d_vec), dim=0)
    d_vec1 = torch.cat((d_vec, Inf * torch.ones(1, device=device)))

    scale = (num_lev - 1) / 2
    data = torch.round(scale * tx.float() + scale).to(torch.int32)  # decode TX
    data_IQinv[:, 0, :], data_IQinv[:, 1, :] = data[:, 0, :], -(
        data[:, 1, :] - scale * 2
    )  # compensate potential IQ flip

    rx *= torch.mean(
        torch.sqrt(tx[:, 0, :].float() ** 2 + tx[:, 1, :].float() ** 2)
    ) / torch.mean(
        torch.sqrt(rx[:, 0, :] ** 2 + rx[:, 1, :] ** 2)
    )  # normalize constellation output

    ### zero phase-shift  torch.sqrt(2*torch.mean(rx[0,:N*sps:sps]**2))
    SER[0, :, 0] = dec_on_bound(rx, data, d_vec0, d_vec1)
    SER[1, :, 0] = dec_on_bound(rx, data_IQinv, d_vec0, d_vec1)

    ### pi phase-shift
    rx_pi = -(rx).detach().clone()
    SER[0, :, 1] = dec_on_bound(rx_pi, data, d_vec0, d_vec1)
    SER[1, :, 1] = dec_on_bound(rx_pi, data_IQinv, d_vec0, d_vec1)

    ### pi/4 phase-shift
    rx_pi4 = torch.empty_like(rx)
    rx_pi4[:, 0, :], rx_pi4[:, 1, :] = -(rx[:, 1, :]).detach().clone(), rx[:, 0, :]
    SER[0, :, 2] = dec_on_bound(rx_pi4, data, d_vec0, d_vec1)
    SER[1, :, 2] = dec_on_bound(rx_pi4, data_IQinv, d_vec0, d_vec1)

    ### 3pi/4 phase-shift
    rx_3pi4 = -(rx_pi4).detach().clone()
    SER[0, :, 3] = dec_on_bound(rx_3pi4, data, d_vec0, d_vec1)
    SER[1, :, 3] = dec_on_bound(rx_3pi4, data_IQinv, d_vec0, d_vec1)

    SER_out = torch.amin(SER, dim=(0, -1))  # choose minimum estimation per polarization
    return SER_out


def dec_on_bound(rx, tx_int, d_vec0, d_vec1):
    """
    Function to perform (hard) decision based on boundries
    """
    # hard decision based on the decision boundaries d_vec0 (lower) and d_vec1 (upper)
    SER = torch.zeros(rx.shape[0], dtype=torch.float32, device=rx.device)

    xI0 = d_vec0.index_select(dim=0, index=tx_int[0, 0, :])
    xI1 = d_vec1.index_select(dim=0, index=tx_int[0, 0, :])
    corr_xI = torch.bitwise_and((xI0 <= rx[0, 0, :]), (rx[0, 0, :] < xI1))
    xQ0 = d_vec0.index_select(dim=0, index=tx_int[0, 1, :])
    xQ1 = d_vec1.index_select(dim=0, index=tx_int[0, 1, :])
    corr_xQ = torch.bitwise_and((xQ0 <= rx[0, 1, :]), (rx[0, 1, :] < xQ1))

    yI0 = d_vec0.index_select(dim=0, index=tx_int[1, 0, :])
    yI1 = d_vec1.index_select(dim=0, index=tx_int[1, 0, :])
    corr_yI = torch.bitwise_and((yI0 <= rx[1, 0, :]), (rx[1, 0, :] < yI1))
    yQ0 = d_vec0.index_select(dim=0, index=tx_int[1, 1, :])
    yQ1 = d_vec1.index_select(dim=0, index=tx_int[1, 1, :])
    corr_yQ = torch.bitwise_and((yQ0 <= rx[1, 1, :]), (rx[1, 1, :] < yQ1))

    ex, ey = ~(torch.bitwise_and(corr_xI, corr_xQ)), ~(
        torch.bitwise_and(corr_yI, corr_yQ)
    )  # no error only if both I or Q are correct
    SER[0], SER[1] = (
        torch.sum(ex) / ex.nelement(),
        torch.sum(ey) / ey.nelement(),
    )  # SER = numb. of errors/ num of symbols
    return SER


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
        correct_start=True,
        regularize=False,
        decay_lr=False,
        lr_start=1e-3,
        lr_end=1e-4,
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
        self.correct_start = correct_start
        self.regularize = regularize
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_lr = decay_lr

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
        # correlation & finding peak)
        # The adaptive filter should be able to correct polarization flip on its own
        if self.correct_start:
            y_cut = correct_start_polarization(
                y, self.pilot_sequence_up[:, : y.shape[1]], correct_polarization=False
            )
        else:
            y_cut = y

        equalizer_length = self.butterfly_filter.taps.size()[1]
        eq_offset = ((equalizer_length - 1) // 2) // self.sps
        num_steps = self.pilot_sequence.shape[1] - 2 * eq_offset
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
            (4, (self.pilot_sequence.shape[1] - 2 * eq_offset)),
            dtype=torch.complex64,
        )
        peak_distortion = torch.zeros(
            (4, (self.pilot_sequence.shape[1] - 2 * eq_offset)),
            dtype=torch.complex64,
        )
        out = torch.zeros(
            2, (num_samp - equalizer_length) // self.sps, dtype=torch.complex64
        )

        if self.preeq_method is None:
            eq_method = self.method
            lr = self.lr  # 1e-3
        else:
            eq_method = self.preeq_method
            lr = self.lr * self.preeq_lradjust

        if eq_method in ("LMS"):
            regression_seq = y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
        elif eq_method in ("ZF", "ZFadv"):
            regression_seq = self.pilot_sequence_up.clone()
            # regression_seq = torch.concatenate((torch.zeros((2,1),dtype=torch.complex64),self.pilot_sequence_up.clone()))
        elif eq_method in ("LMSZF"):
            regression_seq = (
                torch.sqrt(torch.as_tensor(self.lmszf_weight))
                * y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
                + torch.sqrt(1.0 - torch.as_tensor(self.lmszf_weight))
                * self.pilot_sequence_up.clone()
            )
        elif eq_method in ("noise"):
            regression_seq = torch.zeros_like(self.pilot_sequence_up).normal_()
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
                        regression_seq = self.pilot_sequence_up.clone()
                        # regression_seq = torch.concatenate((torch.zeros((2,1),dtype=torch.complex64),self.pilot_sequence_up.clone()))
                    elif self.method == "LMSZF":
                        regression_seq = (
                            torch.sqrt(torch.as_tensor(self.lmszf_weight))
                            * y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
                            + torch.sqrt(1.0 - torch.as_tensor(self.lmszf_weight))
                            * self.pilot_sequence_up.clone()
                        )

                    elif self.method in ("noise"):
                        regression_seq = torch.zeros_like(
                            self.pilot_sequence_up
                        ).normal_()
                if i == self.preeq_offset:
                    lr = self.lr

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
                if self.decay_lr:
                    # num_steps is not exactly the number of steps due to block size constraints, but
                    # the learning rate will be linearly interpolated nonetheless
                    lr = self.lr_start * (
                        ((self.lr_end / self.lr_start - 1) / num_steps) * i + 1
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
                    if self.regularize:
                        regularize_param = 1e-3
                        self.butterfly_filter.taps = (
                            1 - regularize_param
                        ) * self.butterfly_filter.taps
                    update = (
                        2 * lr * torch.mean(u[:, i - self.block_size : i, :], dim=1)
                    )
                    self.butterfly_filter.taps = self.butterfly_filter.taps + update
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
