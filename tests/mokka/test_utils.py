from mokka.utils import generators
from mokka.utils import bitops
import torch
import numpy as np


def test_generatebits():
    bits = generators.numpy.generate_bits((1, 10))
    assert bits.shape == (1, 10)


def test_generate_all_bits():
    m = 4
    all_bits = generators.numpy.generate_all_bits(m)
    assert all_bits.shape == (2**m, m)


def test_one_hot():
    m = 4
    all_bits = generators.numpy.generate_all_bits(m)
    one_hot = bitops.torch.bits_to_onehot(torch.tensor(all_bits.copy()))
    expected_result = torch.tensor(np.zeros((2**m, 2**m)))
    for idx in range(2**m):
        expected_result[idx][idx] = 1.0
    assert torch.all(one_hot == expected_result)


def test_bits2idx():
    m = 4
    bits = generators.torch.generate_bits((100, m))
    idx = bitops.torch.bits2idx(bits)
    oh_vector = bitops.torch.bits_to_onehot(bits)
    oh_idx = bitops.torch.onehot_to_idx(oh_vector)
    assert torch.all(idx == oh_idx)


def test_bits2bits():
    m = 4
    bits = generators.torch.generate_bits((100, m))
    idx = bitops.torch.bits2idx(bits)
    idx_bits = bitops.torch.idx2bits(idx, m)
    assert torch.all(bits == idx_bits)
