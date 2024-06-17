"""
PyTorch implementations of adaptive equalizers.

"""
from ..torch import Butterfly2x2
from ..torch import correct_start_polarization, correct_start, find_start_offset
from ...functional.torch import convolve_overlap_save
from ..torch import h2f
import torch
import logging

logger = logging.getLogger(__name__)

from collections import namedtuple


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
        singularity_length=3000,
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
        self.register_buffer("singularity_length", torch.as_tensor(singularity_length))
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
    #bias1 = 1e-8
    loss = torch.sum(A) + (N - L + 1) * torch.sum(torch.log(C + bias)) #/ sps
    var = C / (N - L + 1) #* sps #N
    return loss, var

##############################################################################################

def ELBO_DP_IQ(y, q, sps, amp_levels, h_est, p_amps=None):
    """
    Calculate dual-pol. ELBO loss for arbitrary complex constellations.

    Instead of splitting into in-phase and quadrature we can just
    the whole thing.
    This implements the dual-polarization case.
    """
    # Input is a sequence y of length N
    N = y.shape[1]
    h = h_est
    pol = 2 # dual-polarization
    # Now we have two polarizations in the first dimension
    # We assume the same transmit constellation for both, calculating
    # q needs to be shaped 2 x N x M  -> for each observation on each polarization we have M q-values
    # we have M constellation symbols
    L = h.shape[-1]
    L_offset = (L - 1) // 2
    if p_amps is None:
        p_amps = (
            torch.ones_like(amp_levels) / amp_levels.shape[0]
        )



    # # Precompute E_Q{c} = sum( q * c) where c is x and |x|**2
    E_Q_x = torch.zeros(2,2,N, device=q.device, dtype=torch.float32)
    Var = torch.zeros(2,N, device=q.device, dtype=torch.float32)
    num_lev = amp_levels.shape[0]
    E_Q_x[:,0,::sps] = torch.sum(q[:,:,:num_lev] * amp_levels.unsqueeze(0).unsqueeze(0), dim=-1)#.permute(0,2,1)
    E_Q_x[:,1,::sps] = torch.sum(q[:,:,num_lev:] * amp_levels.unsqueeze(0).unsqueeze(0), dim=-1)#.permute(0,2,1)
    Var[:,::sps] = torch.add( # Precompute E_Q{|x|^2}
            torch.sum(q[:,:,:num_lev] * (amp_levels**2).unsqueeze(0).unsqueeze(0), dim=-1),
            torch.sum(q[:,:,num_lev:] * (amp_levels**2).unsqueeze(0).unsqueeze(0), dim=-1)
        )
    Var[:,::sps] -=  torch.sum(E_Q_x[:,:,::sps]**2, dim=1)
    p_amps = p_amps.repeat(2)

    h_absq = torch.sum(h**2, dim=2)

    D_real = torch.zeros(2,N-2*L_offset, device=q.device, dtype=torch.float32)
    D_imag = torch.zeros(2,N-2*L_offset, device=q.device, dtype=torch.float32)
    E = torch.zeros(2, device=q.device, dtype=torch.float32)
    idx = torch.arange(2*L_offset,N)
    nm = idx.shape[0]

    for j in range(2*L_offset+1): # h[chi,nu,c,k]
        D_real += h[:,0,0:1,j].expand(-1,nm) * E_Q_x[0,0:1,idx-j].expand(pol,-1) - h[:,0,1:2,j].expand(-1,nm) * E_Q_x[0,1:2,idx-j].expand(pol,-1) \
            + h[:,1,0:1,j].expand(-1,nm) * E_Q_x[1,0:1,idx-j].expand(pol,-1) - h[:,1,1:2,j].expand(-1,nm) * E_Q_x[1,1:2,idx-j].expand(pol,-1)
        D_imag += h[:,0,1:2,j].expand(-1,nm) * E_Q_x[0,0:1,idx-j].expand(pol,-1) + h[:,0,0:1,j].expand(-1,nm) * E_Q_x[0,1:2,idx-j].expand(pol,-1) \
            + h[:,1,1:2,j].expand(-1,nm) * E_Q_x[1,0:1,idx-j].expand(pol,-1) + h[:,1,0:1,j].expand(-1,nm) * E_Q_x[1,1:2,idx-j].expand(pol,-1)
        Var_sum = torch.sum(Var[:,idx-j], dim=-1)
        E += h_absq[:,0,j] * Var_sum[0] + h_absq[:,1,j] * Var_sum[1]


    # Term A - sum all the things, but spare the first dimension, since the two polarizations
    # are sorta independent
    bias = 1e-14
    A = torch.sum(
        q[:,L_offset:-L_offset,:] * torch.log((q[:,L_offset:-L_offset,:] / p_amps.unsqueeze(0).unsqueeze(0)) + bias),
        dim=(1, 2),
    )# Limit the length of y to the "computable space" because y depends on more past values than given
    # We try to generate the received symbol sequence with the estimated symbol sequence
    C = torch.sum(
        y[:, L_offset:-L_offset].real ** 2 + y[:, L_offset:-L_offset].imag ** 2, axis=1
    )
    C += -2*torch.sum( y[:, L_offset:-L_offset].real*D_real + y[:, L_offset:-L_offset].imag*D_imag, dim=1) + torch.sum( D_real**2 + D_imag**2, dim=1) + E

    # We compute B without constants
    # B_tilde = -N * torch.log(C)
    loss = torch.sum(A) + (N - L + 1) * torch.sum(torch.log(C + 1e-8)) #/ sps
    var = C / (N - L + 1) #* sps
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
        self.optimizer = torch.optim.SGD(
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
        self.optimizer = torch.optim.SGD(
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

            # print("noise_sigma: ", self.demapper.noise_sigma)
            loss.backward()
            self.optimizer.step()
            #self.optimizer_var.step()
            self.optimizer.zero_grad()
            #self.optimizer_var.zero_grad()

            if self.var_from_estimate == True:
                self.demapper.noise_sigma = torch.clamp(
                    torch.sqrt(torch.mean(var.detach().clone())/2), min=torch.tensor(0.05, requires_grad=False, device=q_hat.device) , max=2*self.demapper.noise_sigma.detach().clone() #torch.sqrt(var).detach()), min=0.1
                )

            output_symbols = y_symb[
                :, : self.block_size
            ]  # - self.butterfly_forward.num_taps // 2]
            # logger.debug("VAE LE num output symbols: %s", output_symbols.shape[1])
            out.append(
                output_symbols
            )  # out.append(y_symb[:,:num_samps-self.butterfly_forward.num_taps +1])

            output_q = q_hat[
                :, : self.block_size, :
            ]
            out_q.append(
                output_q
            )

        #print("loss: ", loss, "\t\t\t var: ", var)
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
    Class that can be dropped in to perform equalization as in ...
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
        device='cpu'
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
        pol = 2 # dual-polarization
        self.h_est = torch.zeros([pol,pol,2,num_taps_backward])     # initialize estimated impulse response
        self.h_est[0,0,0,num_taps_backward//2+1], self.h_est[1,1,0,num_taps_backward//2+1] = 1, 1
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
            num_taps=self.num_taps_forward.item(), trainable=True, timedomain=True, device=self.butterfly_forward.taps.device
        )
        pol = 2 # dual-polarization
        self.h_est = torch.zeros([pol,pol,2,self.num_taps_backward])     # initialize estimated impulse response
        self.h_est[0,0,0,self.num_taps_backward//2+1], self.h_est[1,1,0,self.num_taps_backward//2+1] = 1, 1
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
                        ), axis=-1
                    ),
                    torch.cat(
                        (
                            self.demapper(y_symb[1, :].real).unsqueeze(0),
                            self.demapper(y_symb[1, :].imag).unsqueeze(0),
                        ), axis=-1
                    ),
                ), axis=0
            )

            # We calculate the loss with less symbols, since the forward operation with "valid"
            # is missing some symbols
            # We assume the symbol of interest is at the center tap of the filter
            y_index = in_index[
                (self.butterfly_forward.num_taps - 1)
                // 2 : -((self.butterfly_forward.num_taps - 1) // 2)
            ]
            loss, var = ELBO_DP_IQ(
            #loss, var = ELBO_DP(
                y[:, y_index],
                q_hat,
                self.sps,
                self.demapper.constellation,
                self.h_est,
                p_amps=self.demapper.p_symbols
                #p_constellation=self.demapper.p_symbols
            )

            # print("noise_sigma: ", self.demapper.noise_sigma)
            print(loss)
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
            eq_out = namedtuple("eq_out", ["y", "q", "var", "loss"])
            return eq_out(torch.cat(out, axis=1), torch.cat(out_q, axis=1), var, loss)
        return torch.cat(out, axis=1)

    def update_lr(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update_var(self, new_lr):
        self.lr = new_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr


def update_adaptive(y_hat_sym, pilot_seq, regression_seq, idx, length, sps):
    e_k = pilot_seq[idx] - y_hat_sym[idx]
    idx_up = idx * sps

    # print("Using regression sequence at indices: ", idx_up, " to ", idx_up + length)
    result = e_k * torch.flip(
        regression_seq[idx_up : idx_up + length].conj().resolve_conj(), dims=(0,)
    )
    return result, e_k


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
        block_size=1,
        adaptive_lr=False,
        adaptive_scale=0.1,
        preeq_method=None,
        preeq_offset=3000,
        preeq_lradjust=1.0,
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
        self.update = update_adaptive
        self.block_size = block_size
        self.adaptive_lr = adaptive_lr
        self.adaptive_scale = adaptive_scale
        self.preeq_method = preeq_method
        self.preeq_offset = preeq_offset
        self.preeq_lradjust = preeq_lradjust

    def reset(self):
        self.butterfly_filter = Butterfly2x2(num_taps=self.filter_length.item())
        self.butterfly_filter.taps[0, self.filter_length.item() // 2] = 1.0
        self.butterfly_filter.taps[2, self.filter_length.item() // 2] = 1.0

    def forward(self, y):
        # y_cut is perfectly aligned with pilot_sequence_up (after cross correlation & using peak)
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

        lmszf_weight = 0.5

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
                lmszf_weight * y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
                + (1 - lmszf_weight)
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
                            lmszf_weight
                            * y_cut.clone()[:, : self.pilot_sequence_up.shape[1]]
                            + (1 - lmszf_weight)
                            * self.pilot_sequence_up.clone().conj().resolve_conj()
                        )
                if i == self.preeq_offset:
                    lr = lr * self.preeq_lradjust

                if eq_method == "ZFadv":
                    # Update regression seq by calculating h from f and estimating \hat{y}
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
                    regression_seq[0, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[1, i, :], e01 = self.update(
                    out[0, :],
                    self.pilot_sequence[0, eq_offset:],
                    regression_seq[1, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[2, i, :], e11 = self.update(
                    out[1, :],
                    self.pilot_sequence[1, eq_offset:],
                    regression_seq[1, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                u[3, i, :], e10 = self.update(
                    out[1, :],
                    self.pilot_sequence[1, eq_offset:],
                    regression_seq[0, :],
                    i,
                    equalizer_length,
                    self.sps,
                )
                if self.adaptive_lr:
                    # For LMS according to Rupp 2011 this stepsize ensures the stability/robustness
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
        self.update = update_adaptive
        self.method = method

    def reset(self):
        self.taps.zero_()
        self.taps[self.taps.size()[0] // 2] = 1.0

    def forward(self, y):
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
