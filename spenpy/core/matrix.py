"""SPEN encoding matrix: A and its weighted adjoint (InvA).

Ported from calcInvA.m and CalcSRMatrixApprox.m.
"""

import torch
import numpy as np
from spenpy.core.sinc import math_sinc


@torch.no_grad()
def calcSRMatrixApprox(
    MaxPhase: float,
    NumPixels: int,
    k: torch.Tensor,
    Partitions: torch.Tensor,
    b: float | None = None,
    ZeroThreshold: float | None = None,
):
    """Compute the approximate SPEN encoding matrix A.

    Matches MATLAB CalcSRMatrixApprox.m exactly.

    Returns:
        A: encoding matrix [NumKs, NumPixels]
        ADerivative: derivative of A
        IdxPositions: pixel center positions
        PartitionsUsed: partition borders
    """
    DefaultZeroThreshold = 10 * torch.finfo(torch.float32).eps

    if ZeroThreshold is None:
        ZeroThreshold = DefaultZeroThreshold

    aEffective = MaxPhase / (Partitions[-1] - Partitions[0]) ** 2

    if b is None:
        b = -(2 * aEffective * Partitions[1] + k[0])

    Partitions = Partitions.reshape(-1, 1)
    IdxPositions = (Partitions[:-1] + Partitions[1:]) / 2
    delta = Partitions[1:] - IdxPositions

    k = k.reshape(-1, 1)
    NumKs = len(k)

    deltaMat = delta.view(1, -1).expand(NumKs, -1)
    IdxPosMat = IdxPositions.view(1, -1).expand(NumKs, -1)
    kMat = k.expand(-1, NumPixels)

    LinCoeffMat = 2 * aEffective * IdxPosMat + b + kMat
    LinCoeff_x_delta_Mat = LinCoeffMat * deltaMat

    SincInput = LinCoeff_x_delta_Mat
    ExpInput = aEffective * IdxPosMat**2 + b * IdxPosMat + kMat * IdxPosMat

    HighOrder2 = 2 * (
        (LinCoeff_x_delta_Mat**2 - 2) * torch.sin(LinCoeff_x_delta_Mat)
        + 2 * LinCoeff_x_delta_Mat * torch.cos(LinCoeff_x_delta_Mat)
    ) / (LinCoeffMat**3)

    ZeroLinCoeffMatIdxs = torch.abs(LinCoeffMat) < ZeroThreshold
    HighOrder2[ZeroLinCoeffMatIdxs] = (2 / 3) * deltaMat[ZeroLinCoeffMatIdxs] ** 3

    DerivativeOrder1 = (
        2j / LinCoeffMat**2
        * (
            torch.sin(LinCoeff_x_delta_Mat)
            - LinCoeff_x_delta_Mat * torch.cos(LinCoeff_x_delta_Mat)
        )
    )

    A = torch.exp(1j * ExpInput) * (
        (2 * deltaMat) * math_sinc(SincInput) + 1j * aEffective * HighOrder2
    )
    ADerivative = torch.exp(1j * ExpInput) * DerivativeOrder1

    return A, ADerivative, IdxPositions, Partitions


@torch.no_grad()
def calcInvA(
    a_rad2cmsqr: float,
    LPE: float,
    NumPE: int,
    ShiftPE: float,
    SPENAcquireSign: int,
    ky1RelativePos: float,
    GaussRelativeWidth: float,
):
    """Build the SPEN super-resolution encoding matrix and weighted adjoint.

    Ported from calcInvA.m.

    Returns:
        InvA: weighted adjoint reconstruction operator [NumPE, NumPE]
        AFinal: encoding matrix
    """
    MaxPhase = a_rad2cmsqr * LPE**2

    NumPixels = NumPE
    NumPixelsFinal = NumPE

    Partitions = SPENAcquireSign * torch.linspace(
        -LPE / 2, LPE / 2, NumPixels + 1
    ) + ShiftPE / 10
    PartitionsFinal = SPENAcquireSign * torch.linspace(
        -LPE / 2, LPE / 2, NumPixelsFinal + 1
    ) + ShiftPE / 10

    ky = (
        -2
        * SPENAcquireSign
        * a_rad2cmsqr
        * torch.arange(NumPE).float()
        * LPE
        / NumPE
    )

    b = -ky[0] + -2 * a_rad2cmsqr * (
        Partitions[0] + (Partitions[1] - Partitions[0]) * ky1RelativePos
    )

    AFinal = calcSRMatrixApprox(
        MaxPhase, NumPixelsFinal, ky, PartitionsFinal, b
    )[0]

    GaussWeightVar = (GaussRelativeWidth * np.pi * NumPixelsFinal**2 / MaxPhase) ** 2

    yk = -(b + ky) / (2 * a_rad2cmsqr)
    yPixels = (PartitionsFinal[:-1] + PartitionsFinal[1:]) / 2

    DistMat = NumPixelsFinal / LPE * (yk.unsqueeze(1) - yPixels.unsqueeze(0))
    GaussWeight = torch.exp(-DistMat**2 / (2 * GaussWeightVar))

    AGaussWeighted = AFinal * GaussWeight

    # InvA is the conjugate transpose (weighted adjoint), NOT a matrix inverse
    InvA = AGaussWeighted.conj().T

    return InvA, AFinal
