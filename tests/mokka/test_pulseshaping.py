import mokka.pulseshaping.torch as pulseshape  # noqa
from mokka.channels.torch import ComplexAWGN
from mokka.utils import N0
import torch


def test_rrc_matched_filter():
    sps = 2
    num_symbols = 100000
    bpsk_symbols = (-1) ** torch.randint(0, 2, (num_symbols,))
    rrc = pulseshape.PulseShaping.get_rrc_ir(100, 0.2, sps)
    shaped_symbols = rrc(bpsk_symbols, sps)
    matched_symbols = rrc.matched(shaped_symbols, sps)
    error = torch.abs(matched_symbols - bpsk_symbols)
    assert error.allclose(torch.as_tensor(0.0), atol=1e-3)


def test_rrc_awgn_no_normalization():
    sps = 2
    snr = 25
    num_symbols = 500000
    bpsk_symbols = (-1) ** torch.randint(0, 2, (num_symbols,))
    bpsk_symbols = bpsk_symbols.to(torch.complex64)
    rrc = pulseshape.PulseShaping.get_rrc_ir(
        100, 0.2, sps, learnable=False, normalize=False, normalize_symbol=False
    )
    # Without normalization of the pulseshape we need to adjust for the increased energy
    rrc_energy = torch.sum(torch.abs(rrc.impulse_response) ** 2)
    awgn = ComplexAWGN(N0(snr) * rrc_energy)
    signal = rrc(bpsk_symbols, sps)
    noisy_signal = awgn(signal)
    rx_symbols = rrc.matched(noisy_signal, sps)
    error = 10 * torch.log10(
        1.0 / torch.mean(torch.abs(rx_symbols - bpsk_symbols) ** 2)
    )
    assert error.allclose(torch.as_tensor(snr, dtype=torch.float32), atol=1e-2)


def test_rrc_awgn_normalization():
    sps = 2
    snr = 25
    num_symbols = 500000
    bpsk_symbols = (-1) ** torch.randint(0, 2, (num_symbols,))
    bpsk_symbols = bpsk_symbols.to(torch.complex64)
    rrc = pulseshape.PulseShaping.get_rrc_ir(
        100, 0.2, sps, learnable=False, normalize=True, normalize_symbol=False
    )
    # Without normalization of the pulseshape we need to adjust for the increased energy
    awgn = ComplexAWGN(N0(snr))
    signal = rrc(bpsk_symbols, sps)
    noisy_signal = awgn(signal)
    rx_symbols = rrc.matched(noisy_signal, sps)
    error = 10 * torch.log10(
        1.0 / torch.mean(torch.abs(rx_symbols - bpsk_symbols) ** 2)
    )
    assert error.allclose(torch.as_tensor(snr, dtype=torch.float32), atol=1e-2)
