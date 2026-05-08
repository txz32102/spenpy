"""1D regridding for EPI trajectory correction.

Ported from oneD_regriding_PV360.m and FGG_1d_type1.m.
"""

import numpy as np
import torch
from scipy.interpolate import pchip, splrep, splev
from spenpy.fft.transform import fft_xspace_to_kspace


def smooth_trajectory(epi_traj: np.ndarray) -> tuple:
    """Smooth EPI trajectory using pchip + spline approximation.

    Args:
        epi_traj: [1, N] trajectory array

    Returns:
        smoothed trajectory
    """
    x_traj = np.arange(1, epi_traj.shape[1] + 1)
    xq_traj = np.arange(-4, epi_traj.shape[1] + 6)

    # PCHIP interpolation
    traj_1d = epi_traj[0, :]
    pchip_func = pchip(x_traj, traj_1d)
    epi_traj_pchip = pchip_func(xq_traj)

    # Spline smoothing (equivalent to MATLAB spaps with tolerance 0.1)
    tck = splrep(xq_traj, epi_traj_pchip, s=0.1)
    epi_traj_smooth = splev(xq_traj, tck)

    # Trim the extended edges
    return epi_traj_smooth[5:-5]


def one_d_regridding_pv360(
    kfield_slice: np.ndarray,
    epi_traj: np.ndarray,
    n_segments: int,
    matrix_size: list,
) -> np.ndarray:
    """1D regridding of ParaVision data based on trajectory measurements.

    Args:
        kfield_slice: [readout, PE] slice of k-space data
        epi_traj: [1, N] measured EPI trajectory
        n_segments: number of SPEN segments
        matrix_size: target matrix size

    Returns:
        Regridded k-space [target_ro, PE]
    """
    # Smooth trajectory
    epi_traj_s = smooth_trajectory(epi_traj)

    # Build reversed trajectory for alternating lines
    epi_traj_1 = epi_traj_s
    epi_traj_2 = -np.flip(epi_traj_s) + np.max(epi_traj_s)

    max_traj = int(round(np.max(epi_traj_s)))
    n_pe = kfield_slice.shape[1]
    kfield_regrid = np.zeros((max_traj, n_pe), dtype=np.complex128)

    if n_segments % 2 == 0:
        for row_ind in range(1, n_pe + 1):
            seg_idx = (row_ind - 1) // n_segments + 1
            if seg_idx % 2 == 1:
                traj = epi_traj_2
            else:
                traj = epi_traj_1
            kfield_regrid[:, row_ind - 1] = _nufft_1d(
                kfield_slice[:, row_ind - 1], traj, max_traj, nufft_order=12
            )
    else:
        for row_ind in range(1, n_pe + 1):
            if row_ind % 2 == 1:
                traj = epi_traj_1
            else:
                traj = epi_traj_2
            kfield_regrid[:, row_ind - 1] = _nufft_1d(
                kfield_slice[:, row_ind - 1], traj, max_traj, nufft_order=12
            )

    # Zero out edge samples (NUFFT artifacts)
    kfield_regrid[:3, :] = 0
    kfield_regrid[-3:, :] = 0

    # Zero-fill to target matrix size
    target_size = matrix_size[0]
    kfield_zf = np.zeros((target_size, n_pe), dtype=np.complex128)
    trunc = target_size - max_traj
    if round(trunc / 2) * 2 != trunc:
        kfield_zf[round(trunc / 2):-round(trunc / 2), :] = kfield_regrid
    else:
        kfield_zf[round(trunc / 2) + 1:-round(trunc / 2) - 1, :] = kfield_regrid

    return kfield_zf


def _nufft_1d(signal: np.ndarray, traj: np.ndarray, n_out: int, nufft_order: int = 12) -> np.ndarray:
    """1D NUFFT using gridding with Kaiser-Bessel kernel.

    Converts nonuniformly sampled data to uniform k-space grid.

    Returns:
        [n_out] complex array (image-space via FFT)
    """
    from scipy.signal import windows

    # Normalize trajectory to [0, n_out)
    traj_norm = (traj - traj.min()) / (traj.max() - traj.min()) * (n_out - 1)

    # Oversampled grid
    oversample = 2
    n_grid = n_out * oversample

    # Kaiser-Bessel gridding
    grid = np.zeros(n_grid, dtype=np.complex128)
    weight = np.zeros(n_grid, dtype=np.float64)

    beta = 2.34 * nufft_order  # Kaiser beta
    half_w = nufft_order // 2

    for i, t in enumerate(traj_norm):
        center = int(round(t))
        for offset in range(-half_w, half_w + 1):
            idx = center + offset
            if 0 <= idx < n_grid:
                dist = abs(t - idx)
                if dist <= half_w:
                    kb = np.i0(beta * np.sqrt(1 - (dist / half_w) ** 2)) / np.i0(beta)
                    grid[idx] += signal[i] * kb
                    weight[idx] += kb

    # Normalize
    mask = weight > 0
    grid[mask] /= weight[mask]

    # FFT and crop
    image = np.fft.ifft(np.fft.ifftshift(grid))
    image = np.fft.fftshift(image)

    # Crop to target size
    start = (n_grid - n_out) // 2
    return image[start:start + n_out]
