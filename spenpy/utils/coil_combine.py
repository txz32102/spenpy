"""Coil combination for multi-coil MRI data.

Ported from coilCombinebao.m -- Walsh et al. adaptive reconstruction.
"""

import torch
import torch.nn.functional as F


def coil_combine(im1: torch.Tensor) -> torch.Tensor:
    """Combine multi-coil complex images using adaptive reconstruction.

    Input shape:  [sx, sy], [sx, sy, N_coils], or [sx, sy, N_images, N_coils]
    Output shape: [sx, sy] or [sx, sy, N_images]

    Based on: Walsh DO, Gmitro AF, Marcellin MW. Adaptive reconstruction
    of phased array MR imagery. Magn Reson Med 2000;43:682-690.
    """
    if im1.dim() == 2:
        return im1
    if im1.dim() == 3:
        im = im1.unsqueeze(2)
        squeeze_n = True
    elif im1.dim() == 4:
        im = im1
        squeeze_n = False
    else:
        raise ValueError("coil_combine expects a 2D, 3D, or 4D tensor")

    sx, sy, n_images, n_coils = im.shape
    if n_coils == 1:
        out = im[..., 0]
        return out[..., 0] if squeeze_n else out

    filtsize = 7

    rs = torch.zeros(sx, sy, n_coils, n_coils, dtype=im.dtype, device=im.device)
    kernel = torch.ones(1, 1, filtsize, filtsize, dtype=im.real.dtype, device=im.device)

    for kc1 in range(n_coils):
        for kc2 in range(n_coils):
            acc = torch.zeros(sx, sy, dtype=im.dtype, device=im.device)
            for kn in range(n_images):
                prod = im[:, :, kn, kc1] * torch.conj(im[:, :, kn, kc2])
                smoothed_real = F.conv2d(
                    prod.real.unsqueeze(0).unsqueeze(0), kernel, padding=filtsize // 2
                ).squeeze()
                smoothed_imag = F.conv2d(
                    prod.imag.unsqueeze(0).unsqueeze(0), kernel, padding=filtsize // 2
                ).squeeze()
                acc = acc + smoothed_real + 1j * smoothed_imag
            rs[:, :, kc1, kc2] = acc

    im2 = torch.zeros(sx, sy, n_images, dtype=im.dtype, device=im.device)
    for kx in range(sx):
        for ky in range(sy):
            R_mat = rs[kx, ky, :, :]
            U, _, _ = torch.linalg.svd(R_mat)
            myfilt = U[:, 0]
            samples = im[kx, ky, :, :].transpose(0, 1)
            im2[kx, ky, :] = myfilt.conj() @ samples

    return im2[:, :, 0] if squeeze_n else im2
