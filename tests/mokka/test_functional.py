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


def test_convolve_pytorch_real_same():
    a = np.arange(10)
    b = np.arange(3)
    reference = np.convolve(a, b, "same")
    result = (
        mokka.functional.torch.convolve(torch.tensor(a), torch.tensor(b), "same")
        .detach()
        .numpy()
    )
    assert np.allclose(reference, result)


def test_convolve_pytorch_real_valid():
    a = np.arange(10)
    b = np.arange(3)
    reference = np.convolve(a, b, "valid")
    result = (
        mokka.functional.torch.convolve(torch.tensor(a), torch.tensor(b), "valid")
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


def test_convolve_pytorch_complex_same():
    a = np.arange(10) + 1j * np.arange(10)
    b = np.arange(3) - 1j * np.arange(3)
    reference = np.convolve(a, b, "same")
    result = (
        mokka.functional.torch.convolve(torch.tensor(a), torch.tensor(b), "same")
        .detach()
        .numpy()
    )
    assert np.allclose(reference, result)


def test_convolve_pytorch_complex_valid():
    a = np.arange(10) + 1j * np.arange(10)
    b = np.arange(3) - 1j * np.arange(3)
    reference = np.convolve(a, b, "valid")
    result = (
        mokka.functional.torch.convolve(torch.tensor(a), torch.tensor(b), "valid")
        .detach()
        .numpy()
    )
    assert np.allclose(reference, result)


def test_convolve_overlap_save_pytorch_complex():
    a = np.arange(10) + 1j * np.arange(10)
    b = np.arange(3) - 1j * np.arange(3)
    reference = np.convolve(a, b)
    result = (
        mokka.functional.torch.convolve_overlap_save(torch.tensor(a), torch.tensor(b))
        .detach()
        .numpy()
    )
    assert np.allclose(reference, result)


def test_convolve_overlap_save_full_pytorch_complex_long():
    a = np.random.standard_normal(1000) + 1j * np.random.standard_normal(1000)
    b = np.random.standard_normal(10) - 1j * np.random.standard_normal(10)
    reference = np.convolve(a, b, "full")
    result = (
        mokka.functional.torch.convolve_overlap_save(
            torch.tensor(a), torch.tensor(b), "full"
        )
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
