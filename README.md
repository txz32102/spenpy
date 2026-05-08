# spenpy

SPEN (Spatiotemporally Encoded) MRI reconstruction in Python.

A port of traditional MATLAB SPEN reconstruction code for Bruker ParaVision data, with PyTorch support for GPU acceleration and differentiable simulation.

## Installation

```bash
pip install -e .
```

## Quick Start

### Simulation

```python
import torch
from spenpy.sim import SpenSim

# Create simulator
sim = SpenSim(
    L=[4, 4],           # FOV in cm
    acq_point=[256, 256],
    nseg=1,             # single shot
    device="cuda",
)

# Generate k-space from an image
H = torch.randn(1, 256, 256)  # [batch, H, W]
kspace = sim.sim(H)
```

### Reconstruction

```python
from spenpy.recon import reconstruct_odd_segments
from spenpy.core import calcInvA, calcSRMatrixApprox
from spenpy.fft import fft_kspace_to_xspace, fft_xspace_to_kspace
from spenpy.utils import mult_mat_tensor, coil_combine

# Reconstruct from Bruker scan directory
result = reconstruct_odd_segments("/path/to/bruker/scan/")

# Access outputs
print(result.images.shape)       # Final reconstructed image
print(result.imag_origin.shape)  # Image before SPEN correction
print(result.spen_az)            # Encoding matrices
```

### Core Math

```python
import torch
from spenpy.core import calcInvA, calcSRMatrixApprox

# Build SPEN encoding matrix
a_rad2cmsqr = -2.5e6  # chirp phase coefficient
LPE = 4.0              # PE FOV in cm
NumPE = 256            # number of SPEN samples

InvA, A = calcInvA(
    a_rad2cmsqr, LPE, NumPE,
    ShiftPE=0, SPENAcquireSign=1,
    ky1RelativePos=0, GaussRelativeWidth=0.8,
)
```

## Project Structure

```
spenpy/
  core/        SPEN encoding matrix (calcInvA, calcSRMatrixApprox)
  fft/         FFT transforms matching MATLAB conventions
  recon/       Reconstruction pipeline, phase correction, regridding
  bruker/      Bruker ParaVision data readers
  utils/       Tensor ops, coil combine, polyfit, smoothing
  sim/         SPEN forward simulator
  cli/         Command-line interface
```

## MATLAB Equivalents

| MATLAB | Python |
|---|---|
| `calcInvA.m` | `spenpy.core.calcInvA` |
| `CalcSRMatrixApprox.m` | `spenpy.core.calcSRMatrixApprox` |
| `FFTKSpace2XSpace.m` | `spenpy.fft.fft_kspace_to_xspace` |
| `FFTXSpace2KSpace.m` | `spenpy.fft.fft_xspace_to_kspace` |
| `MultMatTensor.m` | `spenpy.utils.mult_mat_tensor` |
| `coilCombinebao.m` | `spenpy.utils.coil_combine` |
| `ReadPVParam.m` | `spenpy.bruker.read_pv_param` |
| `Function_Process_..._PV360.m` | `spenpy.recon.reconstruct_odd_segments` |
| `EvenOddFixPEOddNum.m` | `spenpy.recon.phase.even_odd_fix_pe_odd_num` |
| `oneD_regriding_PV360.m` | `spenpy.recon.gridding.one_d_regridding_pv360` |
| `smooth2a.m` | `spenpy.utils.smooth.smooth2a` |
| `Zero_filling_PV6.m` | `spenpy.utils.zero_filling_pv6` |

## Notes

- FFT conventions match MATLAB exactly: `fftshift(fft(ifftshift(x)))`
- `InvA` is the weighted adjoint (conjugate transpose), not a matrix inverse
- All phase correction uses complex arrays
- Dimension order after permute: `[SPEN/PE, readout, channel, image_index]`
