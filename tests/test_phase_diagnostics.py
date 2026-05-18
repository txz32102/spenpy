"""Tests for optional PV360 phase-correction diagnostics."""

import torch

from spenpy.recon.phase import (
    PhaseCorrectionDiagnostics,
    apply_pv360_one_shot_phase_correction,
)


def _sample_inputs():
    torch.manual_seed(0)
    real = torch.randn(4, 6, 1, 1)
    imag = torch.randn(4, 6, 1, 1)
    roffted_data = torch.complex(real, imag)
    inv_a = torch.eye(4, dtype=torch.complex64)
    one_shot_odd_inv = torch.eye(2, dtype=torch.complex64)
    one_shot_even_inv = torch.eye(2, dtype=torch.complex64)
    return roffted_data, inv_a, one_shot_odd_inv, one_shot_even_inv


def test_phase_correction_default_return_is_tensor():
    corrected = apply_pv360_one_shot_phase_correction(
        *_sample_inputs(),
        optimize=False,
        smooth_motion_phase_between_shots=False,
    )

    assert isinstance(corrected, torch.Tensor)
    assert corrected.shape == (4, 6, 1, 1)


def test_phase_correction_can_return_diagnostics():
    corrected, diagnostics = apply_pv360_one_shot_phase_correction(
        *_sample_inputs(),
        optimize=False,
        smooth_motion_phase_between_shots=False,
        return_diagnostics=True,
    )

    assert corrected.shape == (4, 6, 1, 1)
    assert isinstance(diagnostics, PhaseCorrectionDiagnostics)
    assert len(diagnostics.first_pass) == 1
    assert len(diagnostics.refined_even_odd) == 1
    assert diagnostics.motion_between_shots == []

    first = diagnostics.first_pass[0]
    refined = diagnostics.refined_even_odd[0]
    assert first.phase_map.shape == (2, 6)
    assert first.phase_difference.shape == (2, 6)
    assert refined.smooth_phase.shape == (2, 6)
    assert refined.mask.shape == (2, 6)
