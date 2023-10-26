import numpy as np
import mokka
import torch


def test_convolve_pytorch_real():
    a = np.arange(10)
    b = np.arange(3)
    reference = np.convolve(a, b)
    result = (
        mokka.functional.torch.convolve(torch.tensor(a), torch.tensor(b))
        .detach()
        .numpy()
    )
    assert np.allclose(reference, result)


def test_convolve_pytorch_complex():
    a = np.arange(10) + 1j * np.arange(10)
    b = np.arange(3) - 1j * np.arange(3)
    reference = np.convolve(a, b)
    result = (
        mokka.functional.torch.convolve(torch.tensor(a), torch.tensor(b))
        .detach()
        .numpy()
    )
    assert np.allclose(reference, result)


def test_unwrap():
    angle_diff = np.random.random(10000) * 2 * np.pi
    angles = np.cumsum(angle_diff)
    result = mokka.functional.torch.unwrap(torch.tensor(angles))
    expected_result = np.unwrap(angles)
    assert np.allclose(result, expected_result)
