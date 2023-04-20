from mokka import normalization
import torch


def test_energy_simple():
    test_values = torch.ones((1, 10))
    expected_result = test_values
    result = normalization.energy(test_values)
    assert torch.allclose(result, expected_result)


def test_energy_complex():
    test_values = torch.ones((2, 10)) + 1j * torch.ones((2, 10))
    expected_result = torch.full(
        (2, 10), (1 + 1j) / torch.sqrt(torch.tensor(2)), dtype=torch.complex64
    )
    result = normalization.energy(test_values)
    assert torch.allclose(result, expected_result)


def test_energy_weighted():
    test_values = torch.ones((2, 10)) + 1j * torch.ones((2, 10))
    expected_result = torch.full(
        (2, 10), (1 + 1j) / torch.sqrt(torch.tensor(2)), dtype=torch.complex64
    )
    p = torch.ones((10,)) / 10
    result = normalization.energy(test_values, p)
    assert torch.allclose(result, expected_result)
