"""Contain metrics for evaluation of images."""
from typing import List, Tuple

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from tqdm import tqdm

from .validate import validate_arg_dim, validate_arg_type


def compute_ssim(
    X: Tensor, Y: Tensor, Y_hat: Tensor
) -> Tuple[List[float], List[float]]:
    """
    Compute structural simmilarity index measure (SSIM).

    Parameters
    ----------
    X, Y, Y_hat: 4-D tensors of shape (B, C, H, W)
        B: batch size

        C: number of color channels

        H: height of image

        W: width of image

    Returns
    -------
    List of input and output ssim.
    """

    validate_arg_type(X, Tensor)
    validate_arg_type(Y, Tensor)
    validate_arg_type(Y_hat, Tensor)

    validate_arg_dim(X, 4)
    validate_arg_dim(Y, 4)
    validate_arg_dim(Y_hat, 4)

    ssim_in, ssim_out = [], []
    for i in tqdm(range(len(X)), leave=False, position=0):
        ssim_in.append(
            ssim(
                X[i].numpy().transpose(1, 2, 0),
                Y[i].numpy().transpose(1, 2, 0),
                multichannel=True,
            )
        )
        ssim_out.append(
            ssim(
                Y_hat[i].numpy().transpose(1, 2, 0),
                Y[i].numpy().transpose(1, 2, 0),
                multichannel=True,
            )
        )
    assert len(ssim_in) == len(X)
    assert len(ssim_out) == len(X)
    return ssim_in, ssim_out


def compute_psnr(
    X: Tensor, Y: Tensor, Y_hat: Tensor
) -> Tuple[List[float], List[float]]:
    """
    Compute Peal-Signal-to-Noise-Ratio (PSNR).

    Parameters
    ----------
    X, Y, Y_hat: 4-D tensors of shape (B, C, H, W)
        B: batch size

        C: number of color channels

        H: height of image

        W: width of image

    Returns
    -------
    List of input and output PSNRs.
    """

    validate_arg_type(X, Tensor)
    validate_arg_type(Y, Tensor)
    validate_arg_type(Y_hat, Tensor)

    validate_arg_dim(X, 4)
    validate_arg_dim(Y, 4)
    validate_arg_dim(Y_hat, 4)

    psnr_in, psnr_out = [], []
    for i in tqdm(range(len(X)), leave=False, position=0):
        psnr_in.append(
            psnr(
                Y[i].numpy().transpose(1, 2, 0),
                X[i].numpy().transpose(1, 2, 0),
            )
        )
        psnr_out.append(
            psnr(
                Y[i].numpy().transpose(1, 2, 0),
                Y_hat[i].numpy().transpose(1, 2, 0),
            )
        )
    return psnr_in, psnr_out
