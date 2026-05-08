"""Main SPEN reconstruction pipeline.

Ported from Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360.m.
"""

import torch
import numpy as np
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.utils.tensor import mult_mat_tensor
from spenpy.utils.coil_combine import coil_combine
from spenpy.utils.zero_fill import zero_filling_pv6, rm_zero_filling_pv6
from spenpy.core.matrix import calcInvA
from spenpy.bruker.param import read_pv_param


class SpenReconResult:
    """Container for SPEN reconstruction outputs."""

    def __init__(self):
        self.images: torch.Tensor | None = None
        self.imag_origin: torch.Tensor | None = None
        self.imag_low: torch.Tensor | None = None
        self.spen_az: dict = {}


def reconstruct_odd_segments(
    fid_dir: str,
    kfield: torch.Tensor | None = None,
    device: str = "cpu",
    debug_show: bool = False,
    process_with_pre_phase_corr: bool = True,
    smooth_motion_phase_between_shots: bool = True,
    b_amp_correct: bool = False,
) -> SpenReconResult:
    """Main SPEN reconstruction for odd-numbered segments.

    This is the Python equivalent of:
        Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360(fid_dir)

    Args:
        fid_dir: path to Bruker scan directory
        kfield: pre-loaded k-space data (if None, will be loaded from fid_dir)
        device: torch device
        debug_show: enable debug plotting
        process_with_pre_phase_corr: enable motion/inter-shot phase correction
        smooth_motion_phase_between_shots: smooth shot-to-shot phase maps
        b_amp_correct: enable amplitude correction

    Returns:
        SpenReconResult with images, imag_origin, imag_low, spen_az
    """
    result = SpenReconResult()

    # Read acquisition parameters
    fov = read_pv_param(fid_dir, "PVM_Fov")
    if fov is None:
        fov = [40.0, 40.0]
    if isinstance(fov, (int, float)):
        fov = [fov, fov]
    fov = [f / 10 for f in fov]  # mm to cm

    lpe = fov[1]  # cm
    lro = fov[0]

    n_segments = read_pv_param(fid_dir, "NSegments")
    if n_segments is None:
        n_segments = 1

    matrix_size_param = read_pv_param(fid_dir, "PVM_Matrix")
    if matrix_size_param is None:
        matrix_size_param = [256, 256]
    if isinstance(matrix_size_param, int):
        matrix_size_param = [matrix_size_param, matrix_size_param]

    enc_n_receivers = read_pv_param(fid_dir, "PVM_EncNReceivers")
    if enc_n_receivers is None:
        enc_n_receivers = 1

    pvm_n_echo_images = read_pv_param(fid_dir, "PVM_NEchoImages")
    if pvm_n_echo_images is None:
        pvm_n_echo_images = 1

    spen_gy = read_pv_param(fid_dir, "SpenGyGaussStren")
    if spen_gy is None:
        spen_gy = 0.0

    spat_enc_duration = read_pv_param(fid_dir, "SpatEncDuration")
    if spat_enc_duration is None:
        spat_enc_duration = 0.0
    tp = spat_enc_duration / 1000  # ms to sec

    gamma_hz = 4257.4  # Hz/G

    # Chirp phase coefficient
    a_sign = -1
    rfwdth = tp
    a_rad2cmsqr = a_sign * 2 * np.pi * gamma_hz * spen_gy * rfwdth / lpe

    # SPEN acquisition sign
    spen_acquire_sign = -a_sign
    shift_pe = 0.0
    ky1_relative_pos = 0.0
    gauss_relative_width = 0.8

    # If kfield not provided, load from raw data
    if kfield is None:
        from spenpy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel

        kfield_np = read_bruker_kspace_pv360_fid_multichannel(fid_dir)
        kfield = torch.from_numpy(kfield_np).to(device)

    # Process echoes
    num_echoes = pvm_n_echo_images
    slice_num = kfield.shape[2] if kfield.dim() > 2 else 1
    array_num = kfield.shape[3] if kfield.dim() > 3 else 1

    # Reshape kfield: [readout, PE, slice*array, echo, ...]
    if kfield.dim() >= 4:
        kfield = kfield.reshape(
            kfield.shape[0], kfield.shape[1], slice_num * array_num, -1
        )

    for i_ne in range(num_echoes):
        if kfield.dim() == 4:
            kfield_echo = kfield[:, :, :, i_ne]
        else:
            kfield_echo = kfield

        # Even echo: flip PE direction and change chirp sign
        phase_factor = -1 if i_ne % 2 == 0 else 1
        a_rad2cmsqr_echo = phase_factor * a_rad2cmsqr

        if phase_factor == -1:
            kfield_echo = torch.flip(kfield_echo, [1])

        # Permute: [PE, readout, channel, image_index]
        cmplx_data = kfield_echo.permute(1, 0, 2) if kfield_echo.dim() == 3 else kfield_echo.permute(1, 0, 2).unsqueeze(2)
        if cmplx_data.dim() == 3:
            cmplx_data = cmplx_data.unsqueeze(2)  # [PE, RO, 1, image_index]

        # Image before correction (coil-combined origin image)
        imag = fft_kspace_to_xspace(cmplx_data, dim=1)
        result.imag_origin = coil_combine(imag.squeeze(2).permute(0, 1, 3, 2)) if imag.dim() == 4 else imag.squeeze()

        # Readout-FFT data
        roffted_data = fft_kspace_to_xspace(cmplx_data, dim=1)

        # Build InvA
        inv_a, a_final = calcInvA(
            a_rad2cmsqr_echo, lpe, cmplx_data.shape[0],
            shift_pe, spen_acquire_sign, ky1_relative_pos, gauss_relative_width
        )

        # Apply SR reconstruction
        sr_data = mult_mat_tensor(inv_a, roffted_data)

        # Store reconstruction operators
        if result.spen_az is None:
            result.spen_az = {}
        result.spen_az["tmpInvAZ"] = inv_a
        result.spen_az["tmpAFinal"] = a_final

        # Reshape and coil-combine
        if sr_data.dim() == 4:
            sr_reshaped = sr_data.permute(0, 1, 3, 2)  # [PE, RO, image, channel]
            result.images = coil_combine(sr_reshaped.squeeze(2))
            result.imag_low = coil_combine(sr_reshaped.squeeze(2))
        else:
            result.images = sr_data.squeeze()
            result.imag_low = sr_data.squeeze()

    return result
