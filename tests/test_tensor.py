"""Tests for tensor utility functions."""

import torch
import numpy as np
from spenpy.utils.tensor import mult_mat_tensor
from spenpy.utils.coil_combine import coil_combine
from spenpy.utils.zero_fill import zero_filling_pv6, rm_zero_filling_pv6


class TestMultMatTensor:
    def test_2d(self):
        M = torch.randn(3, 4, dtype=torch.complex64)
        T = torch.randn(4, 5, dtype=torch.complex64)
        result = mult_mat_tensor(M, T)
        expected = M @ T
        assert torch.allclose(result, expected, atol=1e-5)

    def test_3d(self):
        M = torch.randn(3, 4, dtype=torch.complex64)
        T = torch.randn(4, 5, 6, dtype=torch.complex64)
        result = mult_mat_tensor(M, T)
        assert result.shape == (3, 5, 6)

    def test_matches_loop(self):
        """Verify einsum matches explicit loop for 4D case."""
        M = torch.randn(8, 16, dtype=torch.complex64)
        T = torch.randn(16, 32, 2, 3, dtype=torch.complex64)

        result = mult_mat_tensor(M, T)

        # Manual loop version
        expected = torch.zeros(8, 32, 2, 3, dtype=torch.complex64)
        for i in range(2):
            for j in range(3):
                expected[:, :, i, j] = M @ T[:, :, i, j]

        assert torch.allclose(result, expected, atol=1e-4)


class TestZeroFilling:
    def test_zero_fill_roundtrip(self):
        """Zero-fill then remove should recover original."""
        x = torch.randn(64, 128, dtype=torch.complex64)
        zf = zero_filling_pv6(x, 128)
        x_back = rm_zero_filling_pv6(zf, 64)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_zero_fill_larger(self):
        x = torch.randn(32, 64, dtype=torch.complex64)
        zf = zero_filling_pv6(x, 64)
        assert zf.shape[0] == 64

    def test_no_op_when_already_large(self):
        x = torch.randn(64, 128, dtype=torch.complex64)
        zf = zero_filling_pv6(x, 32)  # smaller than current
        assert zf.shape[0] == 64
