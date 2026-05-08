"""Coil combination for multi-coil MRI data.

Ported from coilCombinebao.m -- Walsh et al. adaptive reconstruction.
"""

import torch
import torch.nn.functional as F


def coil_combine(im1: torch.Tensor) -> torch.Tensor:
    """Combine multi-coil complex images using adaptive reconstruction.

    Input shape:  [sx, sy, N_slices, N_coils]
    Output shape: [sx, sy, N_slices]

    Based on: Walsh DO, Gmitro AF, Marcellin MW. Adaptive reconstruction
    of phased array MR imagery. Magn Reson Med 2000;43:682-690.
    """
    if im1.dim() == 2:
        # Single-coil case
        return im1.unsqueeze(-1)

    sx, sy, N_coils = im1.shape[0], im1.shape[1], im1.shape[-1]
    N = 1  # number of images (slices/arrays packed in last dim)
    filtsize = 7

    # im1: [sx, sy, N_coils] -- no extra N dimension in typical usage
    # Build correlation matrix using 2D smoothing
    Rs = torch.zeros(sx, sy, N_coils, N_coils, dtype=im1.dtype, device=im1.device)

    for kc1 in range(N_coils):
        for kc2 in range(N_coils):
            prod = im1[:, :, kc1] * torch.conj(im1[:, :, kc2])
            # filter2 with 'same' == 2D convolution, using uniform kernel
            prod_real = prod.real
            prod_imag = prod.imag
            kernel = torch.ones(1, 1, filtsize, filtsize, device=im1.device)
            smoothed_real = F.conv2d(
                prod_real.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=filtsize // 2,
            ).squeeze()
            smoothed_imag = F.conv2d(
                prod_imag.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=filtsize // 2,
            ).squeeze()
            Rs[:, :, kc1, kc2] = smoothed_real + 1j * smoothed_imag

    # SVD-based coil combination at each voxel
    im2 = torch.zeros(sx, sy, dtype=im1.dtype, device=im1.device)
    for kx in range(sx):
        for ky in range(sy):
            R_mat = Rs[kx, ky, :, :]  # [N_coils, N_coils]
            U, _, _ = torch.linalg.svd(R_mat)
            myfilt = U[:, 0]  # first left singular vector
            im2[kx, ky] = torch.dot(
                myfilt.conj(), im1[kx, ky, :]
            )

    return im2
