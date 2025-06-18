"""PyTorch implementation of Mappers and Demappers."""

from .. import normalization, functional
from .numpy import QAM
from ..utils.bitops.torch import bits_to_onehot
from ..utils.generators.numpy import generate_all_bits
from . import numpy as classical
import torch
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)


class QAMConstellationMapper(torch.nn.Module):
    """Non-trainable QAM constellation Mapper."""

    def __init__(self, m):
        """Construct QAMConstellationMapper."""
        super(QAMConstellationMapper, self).__init__()
        self.register_buffer("m", torch.tensor(m))
        qam = QAM(m)
        constellation_symbols = qam.get_constellation().flatten()
        self.register_buffer(
            "symbols", torch.tensor(constellation_symbols, dtype=torch.complex64)
        )
        self.register_buffer("p_symbols", torch.full((2**m,), 1.0 / (2**m)))

    def get_constellation(self, *args):
        """
        Return all constellation symbols.

        :returns: all constellation symbols
        """
        b = torch.tensor(generate_all_bits(self.m.item()).copy()).to(
            self.symbols.device
        )
        symbols = self.forward(b, *args)
        return symbols

    def forward(self, b, *args):
        """
        Perform mapping of bitstrings.

        :returns: complex symbol per bitstring of length `self.m`
        """
        device = b.device
        logger.debug("b size: %s", b.size())
        B_hot = bits_to_onehot(b)

        c = torch.unsqueeze(self.symbols, 0).expand(*b.size()[:-1], -1).to(device)
        c = normalization.energy(c, self.p_symbols)
        x = torch.sum(B_hot * c, -1)
        x = torch.unsqueeze(x, 1)
        return x  # , idx[1]


class CustomConstellationMapper(torch.nn.Module):
    """Non-trainable custom constellation Mapper."""

    def __init__(self, m, constellation_symbols):
        """Construct CustomConstellationMapper."""
        super(CustomConstellationMapper, self).__init__()
        assert len(constellation_symbols) == 2**m
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer(
            "symbols",
            constellation_symbols.unsqueeze(1),
        )

    def get_constellation(self, *args):
        """
        Return all constellation symbols.

        :returns: all constellation symbols
        """
        return self.symbols

    def forward(self, b, *args):
        """
        Perform mapping of bitstrings.

        :returns: complex symbol per bitstring of length `self.m`
        """
        device = b.device
        logger.debug("b size: %s", b.size())

        B_hot = bits_to_onehot(b)

        c = torch.unsqueeze(self.symbols, 0).expand(*b.size()[:-1], -1).to(device)
        x = torch.sum(B_hot * c, 1)
        x = torch.unsqueeze(x, 1)
        return x  # , idx[1]


class SimpleConstellationMapper(torch.nn.Module):
    """
    Mapper which  maps using a simple list of weights  and a one-hot vector.

    :param m: bits per symbol
    """

    def __init__(self, m, qam_init=False):
        """Construct SimpleConstellationMapper."""
        super(SimpleConstellationMapper, self).__init__()
        self.register_buffer("m", torch.tensor(m))

        if qam_init:
            symbols = QAM(m).get_constellation().flatten()
            self.register_parameter(
                "weights",
                torch.nn.Parameter(
                    torch.tensor(
                        np.stack((symbols.real, symbols.imag), axis=-1),
                        dtype=torch.float32,
                    ),
                ),
            )
        else:
            self.weights = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.zeros((2**m, 2), dtype=torch.float32)
                )
            )

    def get_constellation(self, *args):
        """
        Return constellation for all input bits.

        :params args: same arguments as for forward() if constellation mapper is
                      parametrized
        :returns: tensor of constellation points
        """
        # Test bits
        B = generate_all_bits(self.m.item()).copy()
        bits = torch.from_numpy(B.copy()).to(self.weights.device)
        logger.debug("bits device: %s", bits.device)
        out = self.forward(bits)
        return out

    def forward(self, b, *args):
        """
        Perform mapping of bits to symbols.

        This mappper ignores all optional arguments as this Module
        is not parametrizable on call

        :params b: PyTorch tensor with bit vectors
        :returns: mapped constellation points
        """
        # Generate one-hot representatios for m bits
        # in vectors of length 2**m
        logger.debug("b size: %s", b.size())
        B_hot = bits_to_onehot(b)
        c = torch.view_as_complex(
            torch.unsqueeze(self.weights, 0).expand(b.size()[0], -1, -1),
        )
        c = normalization.energy(c)
        x = torch.sum(B_hot * c, -1)
        x = torch.unsqueeze(x, 1)
        return x

    @staticmethod
    def load_model(model_dict):
        """
        Load saved weights.

        :param model_dict: dictionary loaded with `torch.load()`
        """
        mapper_dict = {
            k[len("mapper.") :]: v
            # k.removeprefix("mapper."): v #
            # Reenable if terminal computers are Python 3.9+
            for k, v in model_dict.items()
            if k.startswith("mapper")
        }
        m = mapper_dict["m"].item()
        model = SimpleConstellationMapper(m)
        model.load_state_dict(mapper_dict)
        return model


class ConstellationMapper(torch.nn.Module):
    """
    Mapper which maps from bits in float 0,1 representation to \
    Symbols in the complex plane.

    :param m: bits per symbol
    :param mod_extra_params: index of extra parameters to feed the NN
    :param center_constellation: Apply operation at the end to center constellation
    :param normalize: Apply normalization to unit energy
    :param qam_init: Initialize the weights to form a Gray mapped QAM constellation
    """

    def __init__(
        self,
        m,
        mod_extra_params=None,
        center_constellation=False,
        normalize=True,
        qam_init=False,
    ):
        """Construct ConstellationMapper."""
        super(ConstellationMapper, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer("mod_extra_params", torch.tensor(mod_extra_params or []))
        self.register_buffer("center_constellation", torch.tensor(center_constellation))
        self.register_buffer("normalize", torch.tensor(normalize))
        # Mapper
        self.map1 = torch.nn.Linear(max(len(mod_extra_params or []), 1), 2 ** (m + 1))
        self.map2 = torch.nn.Linear(2 ** (m + 1), 2 ** (m + 1))
        self.register_buffer("p_symbols", torch.full((2**m,), 1.0 / (2**m)))
        if qam_init:
            with torch.no_grad():
                symbols = QAM(m).get_constellation().flatten()
                self.map1.weight.zero_()
                self.map1.bias.fill_(1)
                self.map2.weight = torch.nn.Parameter(
                    torch.diag(
                        torch.tensor(
                            np.stack((symbols.real, symbols.imag), axis=-1).flatten(),
                            dtype=torch.float32,
                        ),
                    )
                )
                self.map2.bias.zero_()
        else:
            torch.nn.init.xavier_normal_(self.map1.weight)
            torch.nn.init.xavier_normal_(self.map2.weight)

    def forward(self, b, *args, one_hot=False):
        """
        Perform mapping of bitstrings.

        By specifying one_hot=true symbol-wise geometric constellation
        shaping is possible calculating the loss w.r.t MI after demapping.

        :params b: input to ConstellationMapper either bits or one-hot vectors
        :params one_hot: specify if b is a one-hot vector

        :returns: complex symbol per bitstring of length `self.m`
        """
        # Generate one-hot representatios for m bits
        # in vectors of length 2**m
        device = b.device
        logger.debug("b size: %s", b.size())
        if one_hot:
            B_hot = b
        else:
            B_hot = bits_to_onehot(b)
        logger.debug("len args: %s", len(args))
        logger.debug("args: %s", args)
        if len(self.mod_extra_params):
            # Concatenate arguments along batch_axis
            mod_args = (
                torch.stack(
                    tuple(args[idx] for idx in self.mod_extra_params),
                    dim=1,
                )
                .to(device)
                .squeeze(2)
            )
            # Generate Constellation mapping c of size 2**m
            # c = self.ReLU(self.map1(snr_dB))
        else:
            # Just feed the network with zeros which will zero
            # influence of weights of first layer
            # and only train bias
            mod_args = torch.zeros((*b.size()[:-1], 1), device=device)
            # c = self.ReLU(self.map1(torch.zeros((torch.numel(B),), device=device)))
            # c = torch.unsqueeze(c, 0)
        logger.debug("mod_args: %s", mod_args)
        logger.debug("mod_args dim: %s", mod_args.size())
        c = self.ReLU(self.map1(mod_args))
        logger.debug("c size at creation: %s", c.size())
        c = self.map2(c)
        c = torch.view_as_complex(
            torch.reshape(c, (*b.size()[:-1], 2 ** self.m.item(), 2))
        )
        if self.center_constellation.item():
            c = normalization.centered_energy(c, self.p_symbols)
        elif self.normalize:
            c = normalization.energy(c, self.p_symbols)
        logger.debug("c device: %s", c.device)
        logger.debug("c size after scaling: %s", c.size())
        logger.debug("c energy for item 0: %s", torch.abs(c[0, :]) ** 2)
        # transmit (batchsize x symbols per training sample) symbols
        x = torch.sum(B_hot * c, -1)
        x = torch.unsqueeze(x, 1)
        logger.debug("x device: %s", x.device)
        return x

    def get_constellation(self, *args):
        """
        Return constellation for all input bits.

        :params args: same arguments as for forward() if constellation mapper is
                      parametrized
        :returns: tensor of constellation points
        """
        # Test bits
        mod_args = torch.tensor(args, dtype=torch.float32)
        mod_args = mod_args.repeat(2 ** self.m.item(), 1).split(1, dim=-1)
        B = generate_all_bits(self.m.item()).copy()
        bits = torch.from_numpy(B.copy()).to(self.map1.weight.device)
        logger.debug("bits device: %s", bits.device)
        out = self.forward(bits, *mod_args).flatten()
        return out

    @staticmethod
    def load_model(model_dict):
        """
        Load saved weights.

        :param model_dict: dictionary loaded with `torch.load()`
        """
        mapper_dict = {
            (k[len("mapper.") :] if k.startswith("mapper") else k): v
            # k.removeprefix("mapper."): v
            # Renable if terminal computers have Python 3.9+
            for k, v in model_dict.items()
            if not (k.startswith("demapper") or k.startswith("channel"))
        }
        m = mapper_dict["m"].item()
        mod_extra_params = mapper_dict["mod_extra_params"].tolist()
        model = ConstellationMapper(m, mod_extra_params)
        if "p_symbols" not in mapper_dict:
            mapper_dict["p_symbols"] = torch.full(
                (2 ** mapper_dict["m"],), 1.0 / (2 ** mapper_dict["m"])
            )
        model.load_state_dict(mapper_dict)
        return model


class SeparatedConstellationMapper(torch.nn.Module):
    """
    Model to map complex and imaginary parts separately \
    in two PAM modulations and add them after modulation.

    :param m: bits per symbol
    :param mod_extra_params: index of extra parameters to feed the NN
    :param center_constellation: Apply operation at the end to center constellation
    :param qam_init: Initialize the weights to form a Gray mapped QAM constellation
    """

    def __init__(self, m, m_real=None, m_imag=None, qam_init=False):
        """Construct SeparatedConstellationMapper."""
        super(SeparatedConstellationMapper, self).__init__()
        if m % 2 > 0 and m_real is None and m_imag is None:
            raise ValueError("m needs to be even or m_real/m_imag have to be specified")
        if m_real is not None and m_imag is not None:
            if m != (m_real + m_imag):
                raise ValueError("m_real and m_imag must add up to m")
        self.register_buffer("m", torch.tensor(m))
        if m_real is None and m_imag is None:
            m_real = m // 2
            m_imag = m // 2
        elif m_imag is None:
            m_imag = m - m_real
        elif m_real is None:
            m_real = m - m_imag

        self.register_buffer("m_real", torch.tensor(m_real))
        self.register_buffer("m_imag", torch.tensor(m_imag))

        if qam_init:
            real_gray_idx = (
                torch.tensor(
                    np.packbits(classical.gray(m_real), bitorder="big", axis=-1)
                )
                .squeeze()
                .to(torch.int64)
            )
            imag_gray_idx = (
                torch.tensor(
                    np.packbits(classical.gray(m_imag), bitorder="big", axis=-1)
                )
                .squeeze()
                .to(torch.int64)
            )

            real_pam = torch.linspace(
                -(2**m_real - 1), 2**m_real - 1, 2**m_real
            ) * np.sqrt(3 / (2 * (2**m - 1)))
            imag_pam = torch.linspace(
                -(2**m_imag - 1), 2**m_imag - 1, 2**m_imag
            ) * np.sqrt(3 / (2 * (2**m - 1)))
            real_sort_idx = torch.argsort(real_gray_idx)
            imag_sort_idx = torch.argsort(imag_gray_idx)
            self.real_weights = torch.nn.Parameter(real_pam[real_sort_idx].unsqueeze(1))
            self.imag_weights = torch.nn.Parameter(imag_pam[imag_sort_idx].unsqueeze(1))

        else:
            self.real_weights = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.zeros((2**m_real, 1), dtype=torch.float32)
                )
            )
            self.imag_weights = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.zeros((2**m_imag, 1), dtype=torch.float32)
                )
            )

    def get_constellation(self, *args):
        """
        Return constellation for all input bits.

        :params args: same arguments as for forward() if constellation mapper is
                      parametrized
        :returns: tensor of constellation points
        """
        # Test bits
        bits = torch.from_numpy(generate_all_bits(self.m.item()).copy()).to(
            self.real_weights.device
        )
        logger.debug("bits device: %s", bits.device)
        out = self.forward(bits, *args)
        return out

    def forward(self, b, *args):
        """
        Perform mapping of bitstrings.

        :returns: complex symbol per bitstring of length `self.m`
        """
        # Generate one-hot representatios for m bits
        # in vectors of length 2**m
        logger.debug("b size: %s", b.size())
        B_hot = bits_to_onehot(b)
        c = (
            self.real_weights.unsqueeze(1).repeat(1, self.imag_weights.size()[0], 1)
            + 1j
            * self.imag_weights.unsqueeze(0).repeat(self.real_weights.size()[0], 1, 1)
        ).flatten()
        c = torch.unsqueeze(c, 0).expand(b.size()[0], -1)
        c = normalization.energy(c)
        x = torch.sum(B_hot * c, -1)
        x = torch.unsqueeze(x, 1)
        return x

    @staticmethod
    def load_model(model_dict):
        """
        Load saved weights.

        :param model_dict: dictionary loaded with `torch.load()`
        """
        mapper_dict = {
            (k[len("mapper.") :] if k.startswith("mapper") else k): v
            # k.removeprefix("mapper."): v #
            # Reenable if terminal computers are Python 3.9+
            for k, v in model_dict.items()
            if not (k.startswith("demapper") or k.startswith("channel"))
        }
        m = mapper_dict["m"].item()
        m_real = mapper_dict["m_real"].item()
        m_imag = mapper_dict["m_imag"].item()
        model = SeparatedConstellationMapper(m, m_real, m_imag)
        model.load_state_dict(mapper_dict)
        return model


class ConstellationDemapper(torch.nn.Module):
    """
    Demap from a complex input with optional SNR to an output with \
    m-Levels with trainable neural networks.

    :param m: Bits per symbol
    :params depth: Number of hidden layers
    :params width: Neurons per layer
    :params with_logit: Output LLRS and not bit probabilities
    :params bitwise: Bitwise LLRs & probabilities
    """

    def __init__(
        self,
        m,
        depth=3,
        width=128,
        with_logit=True,
        bitwise=True,
        demod_extra_params=None,
    ):
        """Construct ConstellationDemapper."""
        super(ConstellationDemapper, self).__init__()
        self.with_logit = with_logit

        self.register_buffer(
            "demod_extra_params", torch.as_tensor(demod_extra_params or [])
        )
        self.register_buffer("m", torch.as_tensor(m))
        self.register_buffer("width", torch.as_tensor(width))
        self.register_buffer("depth", torch.as_tensor(depth))
        self.register_buffer("bitwise", torch.as_tensor(bitwise))

        self.ReLU = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

        self.demaps = torch.nn.ModuleList()
        input_width = 2 + len(demod_extra_params or [])
        self.demaps.append(torch.nn.Linear(input_width, width))
        for d in range(depth - 2):
            self.demaps.append(torch.nn.Linear(width, width))
        if self.bitwise:
            self.demaps.append(torch.nn.Linear(width, m))
        else:
            self.demaps.append(torch.nn.Linear(width, 2**m))

        for demap in self.demaps:
            torch.nn.init.xavier_normal_(demap.weight)

    def forward(self, y, *args):
        """
        Perform bitwise demapping of complex symbols.

        :returns: Approximated log likelihood ratio of dimension `self.m`
        """
        y = torch.view_as_real(y)
        y = torch.squeeze(y, 1)
        # Feed received symbols into decoder network
        if len(self.demod_extra_params):
            # Input is y: (batchsize x symbols per snr), snr: (batchsize x 1)
            y = torch.cat(
                (y, torch.cat(args[self.demod_extra_params].split(1, -1), 1)),
                1,
            )
        for demap in self.demaps[:-1]:
            y = demap(y)
            y = self.ReLU(y)
        # We need to define logit as p(b_j = 0 | y) = sigmoid(logit)
        # to have logit correspond to LLRs for later decoding
        # This changes the sign
        logit = self.demaps[-1](y)
        if not self.training:
            # Change the sign for evaluation
            logit = -1 * logit
        if self.with_logit:
            return logit
        if not self.bitwise:
            return self.softmax(logit)
        return self.sigmoid(logit)

    @staticmethod
    def load_model(model_dict, with_logit=True):
        """
        Load saved weights.

        :param model_dict: dictionary loaded with `torch.load()`
        """
        demapper_dict = {
            (k[len("demapper.") :] if k.startswith("demapper") else k): v
            for k, v in model_dict.items()
            if not (k.startswith("mapper") or k.startswith("channel"))
        }
        m = demapper_dict["m"].item()
        demod_extra_params = demapper_dict["demod_extra_params"].tolist()
        if len(demod_extra_params) == 0:
            demod_extra_params = None
        width = demapper_dict["width"].item()
        depth = demapper_dict["depth"].item()
        model = ConstellationDemapper(
            m,
            width=width,
            depth=depth,
            with_logit=with_logit,
            demod_extra_params=demod_extra_params,
        )
        model.load_state_dict(demapper_dict)
        return model


class ClassicalDemapper(torch.nn.Module):
    r"""
    Classical Bitwise Soft Demapper with Gaussian Noise assumption.

    It takes the noise_sigma and constellation sorted from [0,2**m -1] and constructs
    an more or less efficient demapper which outputs LLRS for all m bits for
    each received symbol

    :param noise_sigma: $\\sigma$ for the Gaussian assumption
    :param constellation: PyTorch tensor of complex constellation symbols
    :param optimize: Use $\\sigma$ as trainable paramater
    :param bitwise: Perform demapping bitwise returning LLRs
    :param p_symbols: PyTorch tensor with symbol probabilities
    """

    def __init__(
        self, noise_sigma, constellation, optimize=False, bitwise=True, p_symbols=None
    ):
        """Construct ClassicalDemapper."""
        super(ClassicalDemapper, self).__init__()
        if optimize:
            self.register_parameter(
                "noise_sigma",
                torch.nn.Parameter(torch.tensor(noise_sigma, dtype=torch.float32)),
            )
        else:
            self.noise_sigma = torch.as_tensor(noise_sigma)
        self.constellation = constellation
        self.softmax = torch.nn.Softmax()

        M = torch.numel(self.constellation)
        m = int(math.log2(M))
        self.register_buffer("m", torch.as_tensor(m))
        self.register_buffer("bitwise", torch.as_tensor(bitwise))

        self.bits = torch.tensor(generate_all_bits(self.m.item()).copy()).to(
            constellation.device
        )

        if p_symbols is None:
            self.register_buffer("p_symbols", torch.ones(M, dtype=torch.float) / M)
        else:
            self.register_buffer("p_symbols", p_symbols)

        with torch.no_grad():
            self.one_idx = torch.nonzero(self.bits)
            self.m_one_idx = [
                self.one_idx[torch.nonzero(self.one_idx[:, 1] == m_i)][:, 0][:, 0]
                for m_i in range(m)
            ]
            self.zero_idx = torch.nonzero(torch.abs(self.bits - 1))
            self.m_zero_idx = [
                self.zero_idx[torch.nonzero(self.zero_idx[:, 1] == m_i)][:, 0][:, 0]
                for m_i in range(m)
            ]

    def update_constellation(self, constellation):
        """Update saved constellation."""
        self.constellation = constellation.clone()

    def forward(self, y, *args):
        """
        Perform bitwise demapping of complex symbols.

        :params y: Received complex symbols of dimension batch_size x 1
        :returns: log likelihood ratios
        """
        if len(y.size()) < 2:
            y = y[:, None]
        dist = (
            torch.exp(
                (-1 * torch.abs(y - self.constellation.to(device=y.device)) ** 2)
                / (2 * torch.clip(self.noise_sigma.to(device=y.device), 0.001) ** 2)
            )
            * self.p_symbols[None, :].to(device=y.device)
            * (2 ** self.m.to(device=y.device))
        )  # batch_size x 2**m

        if self.bitwise:
            return self.forward_bitwise(dist)
        else:
            return self.forward_symbolwise(dist)

    def forward_bitwise(self, dist):
        """
        Perform bitwise demapping of complex symbols.

        :params y: Received complex symbols of dimension batch_size x 1
        :returns: log likelihood ratios
        """
        llrs = torch.zeros((dist.size()[0], self.m.item())).to(dist.device)
        bias = 1e-8  # required to stop sum inside log from becoming zero
        for bit in np.arange(self.m.item()):
            one_llr = torch.log(
                torch.sum(dist[:, self.m_one_idx[bit]] + bias, axis=1)
            ).clone()
            zero_llr = torch.log(
                torch.sum(dist[:, self.m_zero_idx[bit]] + bias, axis=1)
            ).clone()
            llrs[:, bit] = torch.squeeze(zero_llr - one_llr)
        if torch.allclose(llrs, torch.tensor(0.0)):
            raise ValueError("LLRs all became zero, convergence problem")
        return llrs

    def forward_symbolwise(self, dist, *args):
        """
        Perform symbolwise soft demapping of complex symbols.

        :params y: Received complex symbols
        :returns: q-values for each constellation symbol
        """
        # print("dist: ", dist)
        dist = dist + 1e-8
        q = dist / torch.sum(dist, axis=1)[:, None]
        return q


class GaussianDemapper(torch.nn.Module):
    """
    Classical Bitwise Soft Demapper with Gaussian Noise assumption.

    Learns bias and covariance matrix of noise for constellation sorted from [0,2**m -1]
    and constructs a more or less efficient demapper which outputs LLRS for all m bits
    for each received symbol
    """

    def __init__(self, constellation):
        """Construct GaussianDemapper."""
        super(GaussianDemapper, self).__init__()
        constellation = torch.squeeze(constellation).detach().clone()
        if constellation.dtype == torch.complex64:
            dim = 2
            self.constellation = torch.stack(
                (
                    torch.real(constellation.detach().clone()),
                    torch.imag(constellation.detach().clone()),
                ),
                1,
            )
        else:
            dim = 1
            self.constellation = constellation.detach().clone().unsqueeze(1)

        noise_mean = self.constellation.detach().clone().unsqueeze(2)

        M = torch.numel(constellation)
        m = int(math.log2(M))
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer("M", torch.tensor(M))
        self.register_buffer("dim", torch.tensor(dim))

        self.register_parameter(
            "cov_gen_mat",
            torch.nn.Parameter(
                0.01
                * torch.eye(dim, dtype=torch.float32, device=constellation.device)
                .unsqueeze(0)
                .repeat(M, 1, 1)
            ),
        )

        self.register_parameter(
            "noise_mean",
            torch.nn.Parameter(noise_mean),
        )

        self.bits = torch.tensor(generate_all_bits(m).copy()).to(constellation.device)
        with torch.no_grad():
            self.one_idx = torch.nonzero(self.bits)
            self.m_one_idx = [
                self.one_idx[torch.nonzero(self.one_idx[:, 1] == m_i)][:, 0][:, 0]
                for m_i in range(m)
            ]
            self.zero_idx = torch.nonzero(torch.abs(self.bits - 1))
            self.m_zero_idx = [
                self.zero_idx[torch.nonzero(self.zero_idx[:, 1] == m_i)][:, 0][:, 0]
                for m_i in range(m)
            ]

    def update_constellation(self, constellation):
        """
        Update saved constellation.

        :params constellation: Complex constellation symbols
        """
        if self.dim == 2:
            self.constellation = torch.stack(
                (
                    torch.real(constellation.detach().clone()),
                    torch.imag(constellation.detach().clone()),
                ),
                1,
            )

        else:
            self.constellation = constellation.detach().clone().unsqueeze(1)

    def warm_start(self, sym_idx, y, learning_fraction=0.3, single_circular=False):
        """
        Train covariance matrix with known symbols.

        :params sym_idx: Sent symbol index
        :param y: Received complex symbols
        :param learning_fraction: Learning rate
        :param single_circular: Toggle if a a single sigma is learned for all symbols
        """
        with torch.no_grad():
            meanfree_vals = torch.empty(0).to(y.device)
            for i in np.arange(self.M.item()):
                idx = (sym_idx == i).nonzero()
                idx = idx[
                    torch.randperm(len(idx))[
                        0 : int(np.ceil(learning_fraction * len(idx)))
                    ]
                ]
                rx_sym = y[idx].squeeze()
                if y.dtype == torch.complex64:
                    rx_sym = torch.stack(
                        (torch.real(rx_sym), torch.imag(rx_sym))
                    ).squeeze()
                else:
                    rx_sym = torch.transpose(rx_sym.unsqueeze(1), 0, 1)

                rx_mean = torch.mean(rx_sym, 1).unsqueeze(1)
                self.noise_mean[i] = rx_mean

                rx_meanfree = torch.sub(rx_sym, rx_mean)
                if not single_circular:
                    CovMatrix = torch.matmul(
                        rx_meanfree, torch.transpose(rx_meanfree, 0, 1)
                    ) / len(idx)
                    self.cov_gen_mat[i] = torch.linalg.cholesky(CovMatrix)
                else:
                    meanfree_vals = torch.cat(
                        [
                            meanfree_vals,
                            torch.flatten(rx_meanfree),
                        ]
                    )
            if single_circular:
                sigma = torch.sqrt(torch.sum(meanfree_vals**2) / len(meanfree_vals))
                for i in np.arange(self.M.item()):
                    self.cov_gen_mat[i] = sigma * torch.eye(self.dim, device=y.device)

    def forward(self, y, *args):
        """
        Perform bitwise demapping.

        :params y: Received complex symbols of dimension batch_size x 1
        """
        if self.dim == 2:
            symbols = torch.squeeze(torch.stack((torch.real(y), torch.imag(y))))
            noise_cov = torch.transpose(self.cov_gen_mat, 1, 2) * self.cov_gen_mat
            cov_det = (
                noise_cov[:, 0, 0] * noise_cov[:, 1, 1]
                - noise_cov[:, 0, 1] * noise_cov[:, 1, 0]
            )

            noise_inv_cov = torch.zeros(
                noise_cov.shape, dtype=torch.float32, device=y.device
            )
            noise_inv_cov[:, 0, 0] = noise_cov[:, 1, 1] / cov_det
            noise_inv_cov[:, 0, 1] = -noise_cov[:, 0, 1] / cov_det
            noise_inv_cov[:, 1, 0] = -noise_cov[:, 1, 0] / cov_det
            noise_inv_cov[:, 1, 1] = noise_cov[:, 0, 0] / cov_det
        else:
            symbols = torch.transpose(y, 0, 1)
            noise_cov = self.cov_gen_mat**2
            cov_det = noise_cov
            noise_inv_cov = 1 / noise_cov

        Q_YX = torch.zeros(symbols.shape[1], self.M).to(y.device)
        for i in np.arange(self.M.item()):
            symbols_minus_mean = torch.sub(symbols, self.noise_mean[i])
            Q_YX[:, i] = (
                1 / (torch.sqrt(cov_det[i] * ((2 * torch.pi) ** self.dim)))
            ) * torch.exp(
                -0.5
                * torch.sum(
                    symbols_minus_mean * (noise_inv_cov[i] @ symbols_minus_mean),
                    0,
                )
            )
        bias = 1e-8  # required to stop sum inside log2 from becoming zero
        llrs = torch.zeros((y.size()[0], self.m.item())).to(y.device)
        for bit in np.arange(self.m.item()):
            one_llr = torch.log(
                torch.sum(Q_YX[:, self.m_one_idx[bit]] + bias, axis=1)
            ).clone()
            zero_llr = torch.log(
                torch.sum(Q_YX[:, self.m_zero_idx[bit]] + bias, axis=1)
            ).clone()
            llrs[:, bit] = torch.squeeze(zero_llr - one_llr)

        if torch.allclose(llrs, torch.tensor(0.0)):
            raise ValueError("LLRs all became zero, convergence problem")
        return llrs, Q_YX

    # def visualize(self, N=100, sigma_scale=1):
    #     """
    #     :params N: Number of points on contour per constellation point

    #     Uses idea from https://commons.wikimedia.org/wiki/File:MultivariateNormal.png
    #     """
    #     with torch.no_grad():
    #         # Get the sigma ellipses by transform a circle by the cholesky decomp
    #         CovMatrix = np.zeros([2, 2])
    #         t = np.linspace(0.0, 2 * np.pi, num=N)
    #         circle_points = np.stack((np.cos(t), np.sin(t)))
    #         E_points = []
    #         for i in np.arange(self.M.item()):
    #             inv_cov_det = (
    #                 self.noise_inv_cov[i][0, 0] * self.noise_inv_cov[i][1, 1]
    #                 - self.noise_inv_cov[i][0, 1] * self.noise_inv_cov[i][1, 0]
    #             ).item()
    #             CovMatrix[0, 0] = (self.noise_inv_cov[i][1, 1] / inv_cov_det).item()
    #             CovMatrix[0, 1] = (-self.noise_inv_cov[i][0, 1] / inv_cov_det).item()
    #             CovMatrix[1, 0] = (-self.noise_inv_cov[i][1, 0] / inv_cov_det).item()
    #             CovMatrix[1, 1] = (self.noise_inv_cov[i][0, 0] / inv_cov_det).item()
    #             L = np.linalg.cholesky(CovMatrix)
    #             E_points.append(
    #                 (sigma_scale * L @ circle_points)
    #                 + self.noise_mean[i].detach().cpu().numpy()
    #             )

    # return E_points


class SeparatedSimpleDemapper(torch.nn.Module):
    """
    Simplified Demapper which approximates output LLRS of bits separated in real \
    and imaginary parts.

    Do simple function approximation with in -> Linear -> ReLU -> Linear -> out
    This should be easy to replicate in hardware with LUT
    """

    def __init__(self, m, demapper_width, demod_extra_params=()):
        """Construct SeparatedSimpleDemapper."""
        super(SeparatedSimpleDemapper, self).__init__()
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer("width", torch.tensor(demapper_width))
        self.demod_extra_params = demod_extra_params
        self.ReLU = torch.nn.ReLU()
        self.real1 = torch.nn.Linear(1 + len(demod_extra_params), demapper_width)
        self.real2 = torch.nn.Linear(demapper_width, m // 2)
        self.imag1 = torch.nn.Linear(1 + len(demod_extra_params), demapper_width)
        self.imag2 = torch.nn.Linear(demapper_width, m // 2)

    def forward(self, y, *args):
        """
        Perform bitwise demapping of complex symbols.

        :returns: Approximated log likelihood ratio of dimension `self.m`
        """
        y = torch.view_as_real(y)
        y = torch.squeeze(y, 1)
        if len(self.demod_extra_params):
            y_real = torch.cat(
                (torch.unsqueeze(y[:, 0], 1), torch.cat(args, 1)),
                1,
            )
            y_imag = torch.cat(
                (torch.unsqueeze(y[:, 1], 1), torch.cat(args, 1)),
                1,
            )
        else:
            y_real = torch.unsqueeze(y[:, 0], 1)
            y_imag = torch.unsqueeze(y[:, 1], 1)
        llr_real = self.real2(self.ReLU(self.real1(y_real)))
        llr_imag = self.imag2(self.ReLU(self.imag1(y_imag)))
        result = torch.cat((llr_real, llr_imag), 1)
        if self.training:
            return result
        return -1 * result

    @staticmethod
    def load_model(model_dict, with_logit=True):
        """
        Load saved weights.

        :param model_dict: dictionary loaded with `torch.load()`
        """
        demapper_dict = {
            (k[len("demapper.") :] if k.startswith("demapper") else k): v
            # k.removeprefix("demapper."): v
            for k, v in model_dict.items()
            if not (k.startswith("mapper") or k.startswith("channel"))
        }
        m = demapper_dict["m"].item()
        if "demap_extra_params" in demapper_dict:
            demod_extra_params = demapper_dict["demod_extra_params"].tolist()
        else:
            demod_extra_params = ()
        width = demapper_dict["width"].item()
        if len(demod_extra_params) == 0:
            demod_extra_params = ()
        model = SeparatedSimpleDemapper(
            m,
            demapper_width=width,
            demod_extra_params=demod_extra_params,
        )

        model.load_state_dict(demapper_dict)
        return model


class PCSSampler(torch.nn.Module):
    """
    Sample symbol indices from a learnable discrete probability distribution.

    :params m: bits per symbol
    :params l_init: Initial values for the per-symbol logits
    :param symmetries: number of times the probabilty vector is repeated to
                       obtain a probability distribution with uniform distribution
                       for certain bits in the bitstring.
    """

    def __init__(self, m, l_init=None, symmetries=0, pcs_extra_params=None):
        """Construct PCSSampler."""
        super(PCSSampler, self).__init__()
        self.m = torch.tensor(m, dtype=torch.float32)
        self.register_buffer("symmetries", torch.tensor(symmetries))
        self.register_buffer("pcs_extra_params", torch.tensor(pcs_extra_params or []))
        if pcs_extra_params:
            self.map1 = torch.nn.Linear(
                len(pcs_extra_params), 2**m // 2**symmetries
            )
            self.map2 = torch.nn.Linear(
                2**m // 2**symmetries, 2**m // 2**symmetries
            )
            torch.nn.init.xavier_normal_(self.map1.weight)
            torch.nn.init.xavier_normal_(self.map2.weight)
        else:
            if l_init is None:
                self.logits = torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.zeros((2**m // 2**symmetries, 1), dtype=torch.float32)
                    )
                )
            else:
                if l_init.shape[0] != 2**m // 2**symmetries:
                    raise ValueError("l_init must be size of 2**m/2**symmetries")
                self.logits = torch.nn.Parameter(l_init)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, batchsize, *args):
        """
        Generate symbol indices.

        :params batchsize: Number of indices to generate
        """
        # num_symbols = torch.min(
        #    (torch.round(self.p_symbols * batchsize)).type(torch.int32), 1
        # ).values
        idx = (
            functional.torch.distribution_quant_gumbel_softmax(
                self.p_symbols(*args), batchsize
            )
            .squeeze()
            .long()
        )
        # idx = torch.repeat_interleave(
        #     torch.arange(2 ** self.m.item(), device=num_symbols.device), num_symbols
        # )
        logger.debug("idx: %s", idx)
        return idx.squeeze()

    def p_symbols(self, *args):
        """Return current probability distribution."""
        if len(self.pcs_extra_params):
            pcs_args = (
                torch.stack(
                    tuple(args[idx][0, None] for idx in self.pcs_extra_params), dim=1
                )
                .to(self.map1.weight.device)
                .squeeze()
            )
            logits = self.map2(self.relu(self.map1(pcs_args)))
        else:
            logits = self.logits
        logger.debug("logits: %s", logits)
        return self.softmax(
            torch.tile(logits, (1, self.symmetries.item() + 1)).reshape((-1, 1))
        ).squeeze()


class MBPCSSampler(torch.nn.Module):
    """
    This class is supposed to use NNs to find a lambda [0,1] for each given parameter \
    (if given parameters) and then return p_symbols for further simulation.

    :params constellation_symbols: Complex constellation symbols
    """

    def __init__(
        self, constellation_symbols, pcs_extra_params, fixed_lambda=False, l_init=None
    ):
        """Construct MBPCSSampler."""
        super(MBPCSSampler, self).__init__()

        self.register_buffer("symbols", constellation_symbols)
        self.pcs_extra_params = pcs_extra_params
        self.fixed_lambda = fixed_lambda

        if self.fixed_lambda:
            self.register_buffer("_lambda", torch.tensor(0.5))
        else:
            if pcs_extra_params:
                self.map1 = torch.nn.Linear(len(pcs_extra_params), 32)
                self.map2 = torch.nn.Linear(32, 1)
                torch.nn.init.xavier_normal_(self.map1.weight)
                torch.nn.init.xavier_normal_(self.map2.weight)
            else:
                if l_init is None:
                    self.logits = torch.nn.Parameter(
                        torch.nn.init.xavier_normal_(
                            torch.zeros((1,), dtype=torch.float32)
                        )
                    )
                else:
                    if l_init.shape[0] != 1:
                        raise ValueError("l_init must be size of 1")
                    self.logits = torch.nn.Parameter(l_init)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batchsize, *args):
        """
        Generate symbol indices.

        :params batchsize: Number of indices to generate
        """
        num_symbols = (
            functional.torch.distribution_quantization(self.p_symbols(*args), batchsize)
            .squeeze()
            .long()
        )
        idx = torch.repeat_interleave(
            torch.arange(len(self.symbols), device=num_symbols.device), num_symbols
        )
        logger.debug("idx: %s", idx)
        return idx.squeeze()

    def p_symbols(self, *args):
        """Return current probability distribution."""
        lambda_mb = self.lambda_mb(*args)
        return MB_dist(lambda_mb, self.symbols)

    def lambda_mb(self, *args):
        """Return currently trained lambda."""
        if self.fixed_lambda:
            return self._lambda
        if len(self.pcs_extra_params):
            pcs_args = (
                torch.stack(
                    tuple(args[idx][0, None] for idx in self.pcs_extra_params), dim=1
                )
                .to(self.map1.weight.device)
                .squeeze()
            )
            logit = self.map2(self.relu(self.map1(pcs_args)))
        else:
            logit = self.logits
        return logit


def MB_dist(lmbd, symbols):
    """
    Calculate the Maxwell-Boltzmann distribution for a given constellation.

    :params lambda: Lambda parameter of the Maxwell-Boltzmann distribution
    :params symbols: Complex constellation symbols
    """
    mb_dist = torch.exp(-lmbd * torch.abs(symbols) ** 2)
    mb_dist_norm = mb_dist / torch.sum(mb_dist)
    return mb_dist_norm
