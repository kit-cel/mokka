"""Module with blocks for end-to-end simulations in PyTorch."""

import torch
from .. import channels
from ..mapping.torch import ConstellationMapper, ConstellationDemapper
import logging

logger = logging.getLogger(__name__)


class BitwiseAutoEncoder(torch.nn.Module):
    """
    Bitwise Auto-Encoder implementation.

    :param m: bits per symbol and input and output width of the Autoencoder
    :param demapper_width: Width of the hidden layers in the demapper
    :param demapper_depth: Number of hidden layers in the demapper

    """

    def __init__(
        self,
        m,
        demapper_width=127,
        demapper_depth=3,
        channel=None,
        mod_extra_params=(0,),
        demod_extra_params=(0,),
        mapper=None,
        demapper=None,
    ):
        """Construct BitwiseAutoEncoder."""
        super(BitwiseAutoEncoder, self).__init__()
        # Attributes
        self.m = m
        # Layers
        if mapper is None:
            self.mapper = ConstellationMapper(m, mod_extra_params=mod_extra_params)
        else:
            self.mapper = mapper
        if channel is not None:
            self.channel = channel
        else:
            self.channel = channels.ComplexAWGN()
        if demapper is None:
            self.demapper = ConstellationDemapper(
                m,
                depth=demapper_depth,
                width=demapper_width,
                demod_extra_params=demod_extra_params,
            )
        else:
            self.demapper = demapper

    def forward(self, b, *args):
        """
        Perform end-to-end simulation with the BitWiseAutoEncoder.

        Take input-bits and output LLRs, then outside of the auto-encoder the loss
        can be calculated.

        :param b: tensor of bit-strings.
        :returns: tensor of LLRs.
        """
        logger.debug("args: %s", args)
        x = self.mapper(b, *args)

        # Feed transmit symbols through channel with given parametrization

        logger.debug("x size: %s", x.size())
        y = self.channel(x, *args)
        logger.debug("y size: %s", y.size())

        # For symbols with E(|x|**2) = 1 N0 == 1/snr
        # with sqrt(N0) for I and sqrt(N0) for Q
        # to create circular Gaussian noise in the complex domain
        # TODO seed RNG for controlled/reproducible simulations
        x_hat = self.demapper(y, *args)
        logger.debug("x_hat size: %s", x_hat.size())
        return x_hat

    @staticmethod
    def load_model(model_dict):
        """Load model weights from a dictionary.

        :param model_dict: model weights from a file, opened with torch.load.
        :returns: BitwiseAutoEncoder model initialized with saved weights.

        """
        m = model_dict["demapper.m"].item()
        demod_extra_params = model_dict["demapper.demod_extra_params"].tolist()
        mod_extra_params = model_dict["mapper.mod_extra_params"].tolist()
        width = model_dict["demapper.width"].item()
        depth = model_dict["demapper.depth"].item()
        model = BitwiseAutoEncoder(
            m,
            demapper_width=width,
            demapper_depth=depth,
            demod_extra_params=demod_extra_params,
            mod_extra_params=mod_extra_params,
        )
        model.load_state_dict(model_dict)
        return model
