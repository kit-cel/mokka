"""
PyTorch implementations of adaptive equalizers.

"""
from ..torch import Butterfly2x2
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
            range(eq_offset, num_samp - 1 - eq_offset, self.sps * self.block_size)
        ):
            if i % 20000 == 0 and i != 0:
                lr = lr / 2.0
            if i == 10000 and self.no_singularity:
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
