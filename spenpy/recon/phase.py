"""Phase estimation and correction for SPEN reconstruction.

Ported from EvenOddFixPEOddNum.m and related phase correction functions.
"""

import torch
import numpy as np
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.utils.tensor import mult_mat_tensor
from spenpy.utils.polyfit import polyval2, polyfit_weighted_2_subset
from spenpy.utils.smooth import smooth2a
from spenpy.core.matrix import calcInvA


def even_odd_fix_pe_odd_num(
    img_even: torch.Tensor,
    img_odd: torch.Tensor,
    mag_even: torch.Tensor,
    mag_odd: torch.Tensor,
    polyfit_order: int = 2,
    std2_noise_thresh: float = 4.0,
    mask: torch.Tensor | None = None,
):
    """Estimate even/odd phase map from half-resolution SPEN reconstructions.

    Ported from EvenOddFixPEOddNum.m.

    Returns:
        smooth_phase: smoothed phase map to apply to even rows
    """
    # 1. Combine phase difference across channels
    even_odd_phase_dif = img_even * torch.conj(img_odd)
    combined = even_odd_phase_dif.sum(dim=-1) if even_odd_phase_dif.dim() > 2 else even_odd_phase_dif

    # 2. Magnitude-based mask
    if mask is None:
        mag_combined = (mag_even * mag_odd).sum(dim=-1) if mag_even.dim() > 2 else mag_even * mag_odd
        noise_level = mag_combined.std()
        mask = mag_combined > std2_noise_thresh * noise_level

    # 3. Estimate phase
    phase_diff = torch.angle(combined)

    # 4. Fit 2D polynomial to phase difference within mask
    ro_coords = torch.arange(phase_diff.shape[1], device=phase_diff.device, dtype=torch.float64)
    pe_coords = torch.arange(phase_diff.shape[0], device=phase_diff.device, dtype=torch.float64)

    if mask.sum() > 0:
        # Use weighted polynomial fit
        weights = mask.float()
        p = polyfit_weighted_2_subset(ro_coords, pe_coords, phase_diff, polyfit_order, weights)

        # Evaluate polynomial phase map
        poly_phase = polyval2(p, ro_coords, pe_coords)

        # 5. Combine polynomial fit with raw phase difference
        final_phase = poly_phase * mask.float()
        smooth_phase = smooth2a(final_phase, 5, 5)
    else:
        smooth_phase = torch.zeros_like(phase_diff)

    return smooth_phase


def apply_phase_correction(
    data: torch.Tensor,
    phase_map: torch.Tensor,
    row_indices: slice | list | torch.Tensor,
):
    """Apply phase correction to specific rows of data.

    data: [SPEN/PE, readout, ...]
    phase_map: [readout, SPEN/PE_subset] or matching shape
    row_indices: which rows to correct
    """
    data_corrected = data.clone()
    correction = torch.exp(-1j * phase_map)

    if isinstance(row_indices, slice):
        data_corrected[row_indices] *= correction
    elif isinstance(row_indices, list):
        for i, idx in enumerate(row_indices):
            data_corrected[idx] *= correction[:, i] if correction.dim() > 1 else correction

    return data_corrected


def polynomial_phase_optimization(
    data: torch.Tensor,
    central_idxs: list,
    outer_idxs: list,
    inv_a: torch.Tensor,
    n_shots: int,
    initial_p: torch.Tensor | None = None,
    max_iter: int = 2000,
):
    """Optimize polynomial phase parameters by minimizing central-vs-outer PE consistency cost.

    Ported from PenltyWholeSPEN_For_3DSPEN_PEZ.m + fminsearch loop.
    """
    from scipy.optimize import minimize

    def cost_fn(p_use):
        p_tensor = torch.tensor(p_use, dtype=data.dtype, device=data.device)
        ro_coords = torch.arange(data.shape[1], device=data.device, dtype=torch.float64)
        pe_coords = torch.arange(data.shape[0], device=data.device, dtype=torch.float64)

        phase_map = polyval2(p_tensor, ro_coords, pe_coords)
        phase_correction = torch.exp(-1j * phase_map)

        corrected = data * phase_correction
        reconstructed = mult_mat_tensor(inv_a, corrected)

        # Cost: difference between central and outer PE regions
        central = reconstructed[central_idxs].abs().mean()
        outer = reconstructed[outer_idxs].abs().mean()
        return (central - outer).abs().item()

    if initial_p is None:
        initial_p = np.zeros(7)

    result = minimize(cost_fn, initial_p.cpu().numpy() if isinstance(initial_p, torch.Tensor) else initial_p,
                      method="Nelder-Mead", options={"maxiter": max_iter})

    best_p = torch.tensor(result.x, dtype=data.dtype, device=data.device)
    return best_p, result.fun
