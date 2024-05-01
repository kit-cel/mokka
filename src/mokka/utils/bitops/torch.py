"""Operations for bit manipulation in PyTorch."""
import torch


def bits2idx(bits):
    """
    Convert bits to their integer representation.

    :param bits: Vectors of bits with the last dimension representing the size of words
    :returns: Vector of integers
    """
    coeff = 2 ** torch.arange(bits.shape[-1], device=bits.device)
    B = torch.sum(bits * torch.flip(coeff, dims=(0,)), -1)
    return B


def bits_to_onehot(bits):
    """
    Convert bits to one hot vectors.

    :param bits: Vectors of bits with the last dimension representing the size of words
    :returns: Vectors of one-hot vectors
    """
    B = bits2idx(bits)
    idx = torch.meshgrid(list(torch.arange(s) for s in B.size()), indexing="ij")
    B_hot = torch.zeros(
        (*B.size(), 2 ** bits.shape[-1]), dtype=torch.float32, device=bits.device
    )
    idx = (*idx, B.long())
    B_hot[idx] = 1.0
    return B_hot


def idx2bits(idxs, num_bits_per_symbol):
    """
    Convert integer representations of bit vectors back to bits.

    :param idxs: Vector of indices
    :param num_bits_per_symbol: size of the bit vectors
    :returns: vector of bit vectors
    """
    bits = torch.remainder(
        torch.bitwise_right_shift(
            torch.unsqueeze(idxs.long(), 1),
            torch.flip(
                torch.arange(num_bits_per_symbol, device=idxs.device), dims=(0,)
            ),
        ),
        2,
    )
    return bits


def onehot_to_idx(onehot):
    """
    Convert a one-hot vector to the corresponding index.

    :params onehot: Input one-hot vectors
    :returns: tensor of indices
    """
    idxs = torch.arange(onehot.size()[1]).unsqueeze(0)
    return torch.sum(idxs * onehot, axis=1)
