"""
spenpy - SPEN (Spatiotemporally Encoded) MRI reconstruction in Python.

A Python port of traditional MATLAB SPEN reconstruction code for Bruker ParaVision data.
"""

from spenpy.core import calcInvA, calcSRMatrixApprox
from spenpy.fft import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.utils import mult_mat_tensor
from spenpy.utils.coil_combine import coil_combine

__version__ = "0.1.0"

__all__ = [
    "calcInvA",
    "calcSRMatrixApprox",
    "fft_kspace_to_xspace",
    "fft_xspace_to_kspace",
    "mult_mat_tensor",
    "coil_combine",
]
