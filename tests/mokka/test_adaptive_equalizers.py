from mokka.equalizers.adaptive import torch as adaptive_eq
from mokka.mapping.torch import QAMConstellationMapper
from mokka.utils.generators.torch import generate_bits
from mokka.equalizers.torch import Butterfly2x2, h2f
from numpy.testing import assert_almost_equal
from mokka.channels.torch import ProakisChannel
import numpy as np

import torch

num_symbols = 100000
# CMA
cma_R = 1.0
cma_test_symbols = QAMConstellationMapper(m=2)(
    generate_bits((num_symbols, 2))
).squeeze()
cma_test_symbols_2x2 = torch.stack(
    [cma_test_symbols, torch.flip(cma_test_symbols, dims=(0,))]
)


def test_CMA_no_channel():
    with torch.no_grad():
        eq = adaptive_eq.CMA(R=1.0, sps=1, lr=1e-4)
        rx_symbols = eq(cma_test_symbols_2x2)
        assert_almost_equal(
            (
                torch.abs(
                    cma_test_symbols_2x2[:, : -eq.filter_length.item()] - rx_symbols
                )
                ** 2
            ).numpy(),
            0.0,
        )


def test_CMA_no_channel_even_filter():
    with torch.no_grad():
        eq = adaptive_eq.CMA(R=1.0, sps=1, lr=1e-4, filter_length=30)
        rx_symbols = eq(cma_test_symbols_2x2)
        assert_almost_equal(
            (
                torch.abs(
                    cma_test_symbols_2x2[:, : -eq.filter_length.item()] - rx_symbols
                )
                ** 2
            ).numpy(),
            0.0,
        )


def test_CMA_linear_channel_flat():
    with torch.no_grad():
        h = torch.as_tensor(
            [[0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16619, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.complex64,
        )
        ch = Butterfly2x2(taps=h, trainable=False)
        eq = adaptive_eq.CMA(R=1.0, sps=1, lr=1e-2, filter_length=5)
        rx_signal = ch(cma_test_symbols_2x2)
        rx_symbols = eq(rx_signal)
        assert_almost_equal(
            (
                torch.abs(
                    cma_test_symbols_2x2[:, 5000 + 2 : -eq.filter_length.item()]
                    - rx_symbols[:, 5000:]
                )
                ** 2
            ).numpy(),
            torch.zeros((2, num_symbols - 5002 - eq.filter_length.item())).numpy(),
            decimal=4,
        )


def test_CMA_linear_channel_ProakisA():
    with torch.no_grad():
        # h = torch.as_tensor(
        #     [[0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [1.16619, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #     dtype=torch.complex64,
        # )
        h = ProakisChannel("a")
        h_2x2 = torch.stack([h, torch.zeros_like(h), h, torch.zeros_like(h)])
        ch = Butterfly2x2(taps=h_2x2, trainable=False)
        eq = adaptive_eq.CMA(R=1.0, sps=1, lr=1e-3, filter_length=12)
        rx_signal = ch(cma_test_symbols_2x2)
        rx_symbols = eq(rx_signal)
        f_zf = h2f(h.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
        # We cannot perfectly equalize the signal anymore, but we can check if we achieve the channel inverse
        # (We should)
        assert_almost_equal(
            (
                (f_zf.squeeze() - eq.butterfly_filter.taps[0, 0]).detach().abs() ** 2
            ).numpy(),
            torch.zeros(eq.butterfly_filter.taps[0, 0].shape[0]).numpy(),
            decimal=1,
        )


# CMloss_NxN

# RDloss_NxN

# MSEloss_NxN

# MSEflex_NxN

# MSEflex_phiMA_NxN

# LMS_NxN

# ELBO_DP

# ELBO_NxN

# ELBO_DP_IQ

# VAE_LE_DP

# VAE_LE_DP_IQ

# VAE_LE_NxN

# VAE_LE_NxN_orig

# VAE_LE_flex_NxN

# VQVAE_LE_DP

# PilotAEQ_DP

# PilotAEQ_SP

# AEQ_SP
