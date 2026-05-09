"""PV360 SPEN phase correction helpers.

The routines here port the one-shot path used by:

* ``FixAndReconRefflessMultishotHybridSPEN_Robust_NewWhole_back.m``
* ``FixAndReconRefflessMultishotHybridSPEN_Robust_OddNum.m``
* ``EvenOddFix.m`` / ``EvenOddFixPEOddNum.m``

They intentionally focus on the odd, single-segment PV360 scans present in
the reference dataset. Multi-shot and user-drawn mask branches are still kept
out of the public reconstruction path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import minimize


@dataclass
class EvenOddPhaseFit:
    coeffs: np.ndarray
    phase_map: np.ndarray
    smooth_phase: np.ndarray
    mask: np.ndarray


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().resolve_conj().numpy()
    return np.asarray(value)


def _matlab_col(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).reshape(-1, order="F")


def _coeff_mask(mask: list[int] | np.ndarray | None, n_coeffs: int) -> np.ndarray:
    if mask is None:
        return np.ones(n_coeffs, dtype=bool)
    out = np.zeros(n_coeffs, dtype=bool)
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    out[: min(flat.size, n_coeffs)] = flat[:n_coeffs]
    return out


def polyval2_np(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MATLAB ``polyval2`` for coefficients ``[1, x, y, x^2, xy, ...]``."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lx = x.size
    ly = y.size
    lp = p.size

    xx = np.tile(x.reshape(1, lx), (ly, 1)).reshape(-1, order="F")
    yy = np.tile(y.reshape(ly, 1), (1, lx)).reshape(-1, order="F")

    v = np.ones((lx * ly, lp), dtype=np.float64)
    ordercolumn = 0
    n = int(round((np.sqrt(1 + 8 * lp) - 3) / 2))
    for order in range(1, n + 1):
        for c in range(ordercolumn + 1, ordercolumn + order + 1):
            v[:, c] = xx * v[:, c - order]
        ordercolumn += order + 1
        v[:, ordercolumn] = yy * v[:, ordercolumn - order - 1]

    return (v @ p).reshape((ly, lx), order="F")


def polyfitweighted2_subset_np(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    order: int,
    weights: np.ndarray,
    coeffs_fit: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """MATLAB ``polyfitweighted2Subset``."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ly, lx = len(y), len(x)
    if z.shape != (ly, lx) or weights.shape != (ly, lx):
        raise ValueError(
            f"z and weights must have shape [len(y), len(x)]={(ly, lx)}, "
            f"got {z.shape} and {weights.shape}"
        )

    n_coeffs = (order + 1) * (order + 2) // 2
    use = _coeff_mask(coeffs_fit, n_coeffs)

    xx = np.tile(x.reshape(1, lx), (ly, 1)).reshape(-1, order="F")
    yy = np.tile(y.reshape(ly, 1), (1, lx)).reshape(-1, order="F")
    zz = z.reshape(-1, order="F")
    ww = weights.reshape(-1, order="F")

    v = np.zeros((zz.size, n_coeffs), dtype=np.float64)
    v[:, 0] = ww
    ordercolumn = 0
    for deg in range(1, order + 1):
        for c in range(ordercolumn + 1, ordercolumn + deg + 1):
            v[:, c] = xx * v[:, c - deg]
        ordercolumn += deg + 1
        v[:, ordercolumn] = yy * v[:, ordercolumn - deg - 1]

    v_use = v[:, use]
    rhs = ww * zz
    if v_use.size == 0 or np.count_nonzero(np.any(v_use != 0, axis=1)) == 0:
        return np.zeros(n_coeffs, dtype=np.float64)

    try:
        p_subset, *_ = np.linalg.lstsq(v_use, rhs, rcond=None)
    except np.linalg.LinAlgError:
        p_subset = np.zeros(int(np.sum(use)), dtype=np.float64)

    p = np.zeros(n_coeffs, dtype=np.float64)
    p[use] = p_subset
    if np.any(~np.isfinite(p)):
        p[:] = 0
    return p


def smooth2a_np(matrix: np.ndarray, nr: int, nc: int | None = None) -> np.ndarray:
    """MATLAB ``smooth2a`` mean filter with truncated edge windows."""
    if nc is None:
        nc = nr
    matrix = np.asarray(matrix, dtype=np.float64)
    nan_mask = np.isnan(matrix)
    data = matrix.copy()
    data[nan_mask] = 0
    kernel = np.ones((2 * nr + 1, 2 * nc + 1), dtype=np.float64)
    summed = ndimage.convolve(data, kernel, mode="constant", cval=0.0)
    counts = ndimage.convolve((~nan_mask).astype(np.float64), kernel, mode="constant", cval=0.0)
    out = summed / counts
    out[nan_mask] = np.nan
    return out


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _imopen(mask: np.ndarray, radius: int) -> np.ndarray:
    return ndimage.binary_opening(mask.astype(bool), structure=_disk(radius))


def _fft_xspace_to_kspace_np(x: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
    )


def _cyclic_shift_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    dim_shift: int = 0,
    dim_other: int = 1,
) -> np.ndarray:
    axes = list(range(max(img1.ndim, dim_shift + 1, dim_other + 1)))
    perm = axes.copy()
    perm[dim_shift] = 0
    perm[0] = dim_shift
    if dim_other == 0:
        if dim_shift != 1:
            perm[1] = 0
            perm[dim_shift] = 1
    else:
        perm[dim_other] = perm[1]
        perm[1] = dim_other

    img1_use = np.transpose(img1, perm)
    img2_use = np.transpose(img2, perm)
    flipped = np.flip(img2_use, axis=0)
    corr = np.fft.ifft(np.fft.fft(img1_use, axis=0) * np.fft.fft(flipped, axis=0), axis=0)
    inv_perm = np.argsort(perm)
    return np.transpose(corr, inv_perm)


def _get_1d_linear_phase_with_unwrap2(
    image_in: np.ndarray,
    image_mask: np.ndarray,
    linear_dimension: int,
    threshold: float = 0.0,
) -> float:
    image_in = np.asarray(image_in)
    image_mask = np.asarray(image_mask)
    if linear_dimension != 0:
        image_in = np.swapaxes(image_in, 0, linear_dimension)
        image_mask = np.swapaxes(image_mask, 0, linear_dimension)

    corr_img = np.concatenate([image_in, np.zeros_like(image_in)], axis=0)
    corr_mask = np.concatenate([image_mask, np.zeros_like(image_mask)], axis=0)
    phase_corr = np.angle(
        np.sum(_cyclic_shift_correlation(corr_img, np.conj(corr_img), 0, 1), axis=1)
    )
    amp_corr = np.sum(
        np.abs(_cyclic_shift_correlation(np.abs(corr_mask), np.abs(corr_mask), 0, 1)),
        axis=1,
    )

    n_use = int(np.floor(phase_corr.shape[0] / 2) - 1)
    if n_use <= 2 or amp_corr[0] == 0:
        return 0.0

    phase_use = np.unwrap(phase_corr[:n_use])
    amp_use = amp_corr[:n_use] / amp_corr[0]
    amp_use[amp_use < threshold] = 0
    pass_threshold = amp_use > threshold
    if not np.any(pass_threshold):
        return 0.0

    pass_change = np.concatenate([[0], np.diff(pass_threshold.astype(int))])
    jump_idx_compact = np.flatnonzero(pass_change[pass_threshold] == 1)
    jump_idx_compact = jump_idx_compact[jump_idx_compact > 0]
    phase_compact = phase_use[pass_threshold].copy()
    for jump_idx in jump_idx_compact:
        phase_jump = phase_compact[jump_idx] - phase_compact[jump_idx - 1]
        phase_compact[jump_idx:] -= phase_jump

    amp_compact = amp_use[pass_threshold]
    n_compact = phase_compact.size
    if n_compact < 2:
        return 0.0

    p = polyfitweighted2_subset_np(
        np.array([1.0]),
        np.arange(1, n_compact + 1, dtype=np.float64),
        phase_compact.reshape(n_compact, 1),
        1,
        amp_compact.reshape(n_compact, 1),
        [0, 0, 1],
    )
    return float(p[2])


def _get_mag_based_mask(
    imgs1: np.ndarray,
    imgs2: np.ndarray,
    std2_noise_thresh_factor: float,
) -> np.ndarray:
    abs1 = np.abs(imgs1)
    abs2 = np.abs(imgs2)
    diff = abs2 - abs1
    std = np.std(diff.reshape(-1, diff.shape[-1], order="F"), axis=0, ddof=0)
    thresh = std.reshape((1, 1, -1)) * std2_noise_thresh_factor
    return (abs1 > thresh) & (abs2 > thresh)


def _get_phase_gradient_based_mask(
    complex_phase: np.ndarray,
    threshold: float = 0.7,
    neighborhood_size: tuple[int, int] = (7, 7),
) -> np.ndarray:
    phase = complex_phase / np.maximum(np.abs(complex_phase), np.finfo(np.float64).eps)
    masks = np.zeros(phase.shape, dtype=bool)
    for idx in range(phase.shape[-1]):
        gy, gx = np.gradient(phase[:, :, idx])
        g = np.sqrt(gx * np.conj(gx) + gy * np.conj(gy))
        filtered = ndimage.median_filter(np.abs(g), size=neighborhood_size, mode="reflect")
        masks[:, :, idx] = filtered < threshold
    return masks


def even_odd_phase_fit(
    img_odd: np.ndarray,
    img_even: np.ndarray,
    mag_odd: np.ndarray | None = None,
    mag_even: np.ndarray | None = None,
    coeffs_fit: list[int] | np.ndarray | None = None,
    polyfit_order: int = 2,
    std2_noise_thresh_factor: float = 2.0,
    morph_open: bool = False,
    return_smooth_phase: bool = False,
) -> EvenOddPhaseFit:
    """Port the phase fit core of MATLAB ``EvenOddFix*`` for one image."""
    odd = np.asarray(img_odd, dtype=np.complex128)
    even = np.asarray(img_even, dtype=np.complex128)
    if odd.ndim == 2:
        odd = odd[:, :, np.newaxis, np.newaxis]
        even = even[:, :, np.newaxis, np.newaxis]
    elif odd.ndim == 3:
        odd = odd[:, :, :, np.newaxis]
        even = even[:, :, :, np.newaxis]

    if mag_odd is None or mag_even is None:
        mag_odd_arr = np.abs(odd)
        mag_even_arr = np.abs(even)
    else:
        mag_odd_arr = np.asarray(mag_odd, dtype=np.float64)
        mag_even_arr = np.asarray(mag_even, dtype=np.float64)
        if mag_odd_arr.ndim == 2:
            mag_odd_arr = mag_odd_arr[:, :, np.newaxis, np.newaxis]
            mag_even_arr = mag_even_arr[:, :, np.newaxis, np.newaxis]
        elif mag_odd_arr.ndim == 3:
            mag_odd_arr = mag_odd_arr[:, :, :, np.newaxis]
            mag_even_arr = mag_even_arr[:, :, :, np.newaxis]

    num_pe, num_ro, _num_channels, num_images = odd.shape
    mag_odd_combined = np.sqrt(np.sum(mag_odd_arr**2, axis=2))
    mag_even_combined = np.sqrt(np.sum(mag_even_arr**2, axis=2))
    min_mag = np.minimum(mag_even_combined, mag_odd_combined)
    phase_diff_combined = np.sum(even * np.conj(odd), axis=2)

    mask = _get_mag_based_mask(mag_even_combined, mag_odd_combined, std2_noise_thresh_factor)

    if num_ro > 1:
        gross = np.angle(phase_diff_combined[:, 1:, :] * np.conj(phase_diff_combined[:, :-1, :]))
        diff_mask = mask[:, 1:, :] & mask[:, :-1, :]
        denom = np.sum(diff_mask, axis=(0, 1))
        mean_gross = np.divide(
            np.sum(gross * diff_mask, axis=(0, 1)),
            denom,
            out=np.zeros(num_images, dtype=np.float64),
            where=denom != 0,
        )
        gross_mod = np.mod(gross, 2 * np.pi)
        mean_gross_mod = np.divide(
            np.sum(gross_mod * diff_mask, axis=(0, 1)),
            denom,
            out=np.zeros(num_images, dtype=np.float64),
            where=denom != 0,
        )
        var_gross = np.divide(
            np.sum((gross * diff_mask) ** 2, axis=(0, 1)),
            denom,
            out=np.zeros(num_images, dtype=np.float64),
            where=denom != 0,
        )
        var_gross_mod = np.divide(
            np.sum((gross_mod * diff_mask) ** 2, axis=(0, 1)),
            denom,
            out=np.zeros(num_images, dtype=np.float64),
            where=denom != 0,
        )
        use_mod = var_gross_mod < var_gross
        mean_gross[use_mod] = mean_gross_mod[use_mod]
        mean_gross[~np.isfinite(mean_gross)] = 0
    else:
        mean_gross = np.zeros(num_images, dtype=np.float64)

    ro_idxs = np.arange(num_ro, dtype=np.float64)
    pe_idxs = np.arange(num_pe, dtype=np.float64)
    n_coeffs = (polyfit_order + 1) * (polyfit_order + 2) // 2
    p_array = np.zeros((n_coeffs, num_images), dtype=np.float64)
    phase_maps = np.zeros((num_pe, num_ro, num_images), dtype=np.float64)
    smooth_maps = np.zeros_like(phase_maps)
    final_mask = mask.copy()

    for img_idx in range(num_images):
        phase_img = phase_diff_combined[:, :, img_idx]
        gross_lin = mean_gross[img_idx]
        modified = phase_img * np.exp(-1j * gross_lin * ro_idxs)[np.newaxis, :]
        mask_initial = mask[:, :, img_idx]
        phase_mask = _get_phase_gradient_based_mask(modified[:, :, np.newaxis])[:, :, 0]
        mask_img = mask_initial & phase_mask
        if morph_open:
            mask_img = _imopen(mask_img, 2)
        final_mask[:, :, img_idx] = mask_img

        fine_lin = _get_1d_linear_phase_with_unwrap2(
            modified,
            mask_initial.astype(np.float64) * modified,
            1,
            0.1,
        )
        modified = modified * np.exp(-1j * fine_lin * ro_idxs)[np.newaxis, :]
        fine_lin_pe = 0.0

        if np.any(mask_initial):
            const_phase = np.angle(np.mean(modified[mask_initial]))
            if not np.isfinite(const_phase):
                const_phase = 0.0
        else:
            const_phase = 0.0
        modified = modified * np.exp(-1j * const_phase)

        weights = mask_img.astype(np.float64) * min_mag[:, :, img_idx]
        p0 = polyfitweighted2_subset_np(
            ro_idxs,
            pe_idxs,
            np.angle(modified),
            polyfit_order,
            weights,
            coeffs_fit,
        )
        p = p0.copy()
        if polyfit_order > 0 and p.size >= 3:
            p[1] = p0[1] + gross_lin + fine_lin
            p[2] = p0[2] + fine_lin_pe
        p[0] = p[0] + const_phase

        phase_map = polyval2_np(p, ro_idxs, pe_idxs)
        p_array[:, img_idx] = p
        phase_maps[:, :, img_idx] = phase_map

        if return_smooth_phase:
            final_phase = polyval2_np(p - p0, ro_idxs, pe_idxs) + np.angle(modified)
            final_phase = final_phase * mask_img.astype(np.float64)
            aa = phase_map
            zeros = final_phase == 0
            final_phase[zeros] = aa[zeros]
            smooth_maps[:, :, img_idx] = smooth2a_np(final_phase, 5, 5)
        else:
            smooth_maps[:, :, img_idx] = phase_map

    return EvenOddPhaseFit(
        coeffs=p_array[:, 0] if num_images == 1 else p_array,
        phase_map=phase_maps[:, :, 0] if num_images == 1 else phase_maps,
        smooth_phase=smooth_maps[:, :, 0] if num_images == 1 else smooth_maps,
        mask=final_mask[:, :, 0] if num_images == 1 else final_mask,
    )


def _apply_one_seg_optimized_even_phase(
    signal: np.ndarray,
    inv_a: np.ndarray,
    image_index: int,
    max_iter: int,
) -> np.ndarray:
    num_pe, num_ro = signal.shape[:2]
    mid_npe = int(np.floor(num_pe))
    central = np.arange(int(mid_npe / 2 - 10), int(mid_npe / 2 + 10) + 1)
    central = central[(central >= 0) & (central < mid_npe)]
    outer = np.setdiff1d(np.arange(mid_npe), central)
    ro_idxs = np.arange(num_ro, dtype=np.float64)
    pe_even_idxs = np.arange(1, num_pe // 2 + 1, dtype=np.float64)
    one_image = signal[:, :, 0, image_index]

    def penalty(p_use: np.ndarray) -> float:
        fixed = one_image.copy()
        phase = polyval2_np(p_use, ro_idxs, pe_even_idxs)
        fixed[1::2, :] = fixed[1::2, :] * np.exp(-1j * phase)
        sr = inv_a @ fixed
        sr_pe = np.abs(_fft_xspace_to_kspace_np(np.abs(sr), axis=0))
        profile = np.sum(np.abs(sr_pe), axis=1)
        denom = np.sum(profile[outer])
        if denom == 0:
            return 0.0
        return float(-np.sum(profile[central]) / denom)

    result = minimize(
        penalty,
        np.zeros(6, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "maxfev": max_iter, "xatol": 1e-4, "fatol": 1e-4},
    )
    best = result.x if np.all(np.isfinite(result.x)) else np.zeros(6, dtype=np.float64)
    phase = polyval2_np(best, ro_idxs, pe_even_idxs)
    signal[1::2, :, 0, image_index] *= np.exp(-1j * phase)
    return best


def apply_pv360_one_shot_phase_correction(
    roffted_data: torch.Tensor,
    inv_a: torch.Tensor,
    one_shot_odd_inv: torch.Tensor,
    one_shot_even_inv: torch.Tensor,
    optimize: bool = True,
    smooth_motion_phase_between_shots: bool = True,
    max_optimizer_iter: int = 2000,
) -> torch.Tensor:
    """Apply MATLAB's default one-shot PV360 phase-correction sequence."""
    signal = _to_numpy(roffted_data).astype(np.complex128, copy=True)
    inv_a_np = _to_numpy(inv_a).astype(np.complex128, copy=False)
    odd_inv = _to_numpy(one_shot_odd_inv).astype(np.complex128, copy=False)
    even_inv = _to_numpy(one_shot_even_inv).astype(np.complex128, copy=False)

    num_pe, num_ro, _num_channels, num_images = signal.shape
    if num_pe % 2 != 0:
        return roffted_data

    ro_idxs = np.arange(num_ro, dtype=np.float64)
    pe_one = np.array([1.0], dtype=np.float64)
    pe_even_idxs = np.arange(1, num_pe // 2 + 1, dtype=np.float64)

    # MATLAB NewWhole_back: first 1D odd/even phase correction.
    odd_sr = np.einsum("ab,brcn->arcn", odd_inv, signal[0::2, :, :, :])
    even_sr = np.einsum("ab,brcn->arcn", even_inv, signal[1::2, :, :, :])
    for img_idx in range(num_images):
        fit = even_odd_phase_fit(
            odd_sr[:, :, 0, img_idx],
            even_sr[:, :, 0, img_idx],
            np.abs(odd_sr[:, :, 0, img_idx]),
            np.abs(even_sr[:, :, 0, img_idx]),
            coeffs_fit=[1, 1, 0, 1, 0, 0],
            morph_open=False,
            return_smooth_phase=False,
        )
        phase_1d = polyval2_np(fit.coeffs, ro_idxs, pe_one)[0, :]
        signal[1::2, :, 0, img_idx] *= np.exp(-1j * phase_1d)[np.newaxis, :]

    if optimize:
        for img_idx in range(num_images):
            _apply_one_seg_optimized_even_phase(
                signal,
                inv_a_np,
                img_idx,
                max_optimizer_iter,
            )

    # MATLAB OddNum: refined odd/even phase correction using SmoothPhase.
    odd_sr = np.einsum("ab,brcn->arcn", odd_inv, signal[0::2, :, :, :])
    even_sr = np.einsum("ab,brcn->arcn", even_inv, signal[1::2, :, :, :])
    for img_idx in range(num_images):
        fit = even_odd_phase_fit(
            odd_sr[:, :, 0, img_idx],
            even_sr[:, :, 0, img_idx],
            np.abs(odd_sr[:, :, 0, img_idx]),
            np.abs(even_sr[:, :, 0, img_idx]),
            coeffs_fit=[1, 1, 1, 1, 1, 1],
            morph_open=True,
            return_smooth_phase=True,
        )
        signal[1::2, :, 0, img_idx] *= np.exp(-1j * fit.smooth_phase)

    if smooth_motion_phase_between_shots:
        whole_sr = np.einsum("ab,brcn->arcn", inv_a_np, signal)
        mag_whole = np.abs(whole_sr)
        for img_idx in range(num_images):
            fit = even_odd_phase_fit(
                mag_whole[:, :, 0, img_idx],
                whole_sr[:, :, 0, img_idx],
                mag_whole[:, :, 0, img_idx],
                mag_whole[:, :, 0, img_idx],
                coeffs_fit=[1, 1, 1, 1, 1, 1],
                morph_open=True,
                return_smooth_phase=True,
            )
            signal[:, :, 0, img_idx] *= np.exp(-1j * fit.smooth_phase)

    return torch.from_numpy(signal).to(device=roffted_data.device, dtype=roffted_data.dtype)
