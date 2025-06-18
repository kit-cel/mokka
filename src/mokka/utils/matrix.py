import torch


def decompose_pmd_pdl(T):
    """
    Implements decomposition introduced in [1, II. 4)]


    [1] M. Karlsson, J. Brentel, and P. A. Andrekson, ‘Long-term measurement of PMD and polarization drift in installed fibers’, Journal of Lightwave Technology, vol. 18, no. 7, pp. 941–951, Jul. 2000, doi: 10.1109/50.850739.
    """
    H_2 = T.transpose(0, 1).conj().resolve_conj() @ T
    H = torch.sqrt(H_2)
    H_inv = torch.linalg.inv(H)
    U = H_inv @ T
    return (U, H)
