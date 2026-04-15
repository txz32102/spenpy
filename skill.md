---
name: spenpy-library-summary
description: Quick orientation for AI tools reviewing the SPENPy repository, its demo workflow, imports, and reconstruction concepts.
---

# SPENPy quick look

## What this repository is
SPENPy is a small Python simulation library for SPEN reconstruction experiments. It focuses on:
- simulating SPEN acquisition from a grayscale image,
- producing corrupted / phase-affected data,
- reconstructing with an approximate inverse system matrix `InvA`,
- offering a faster degradation path through the forward matrix `AFinal`.

The core implementation is in `spenpy/spen.py`.

## Quick start
**Context:** Given a grayscale image tensor `img` of shape `(B, W, H)`.

**1. Full Simulation (Includes Phase Map Degradation)**
Simulate the full Single Point Encoding (SPEN) sequence to generate corrupted data and its corresponding phase map.
```python
from spenpy.spen import spen
import torch

# Get corrupted k-space data and phase map
corrupted_data = spen(noise_level=0.0, acq_point=(96, 96)).sim(img) 
phase_map = spen(noise_level=0.0, acq_point=(96, 96)).get_phase_map(img)
```

**2. Fast Forward Degradation (No Phase Map)**
For a faster simulation that bypasses phase artifacts, apply the forward matrix `AFinal` directly to the complex image.
```python
InvA, AFinal = spen().get_InvA()

# Fast degradation
degraded_data = torch.matmul(AFinal, img.to(torch.complex64))
```

**3. Direct Matrix Reconstruction**
Reconstruct the image from the data using the inverse matrix `InvA`. *(Note: `InvA * AFinal` is not a perfect identity matrix, so minor reconstruction artifacts are expected).*
```python
# Simple reconstruction
reconstructed_img = torch.matmul(InvA, degraded_data)
```

**4. Full Reconstruction with Phase Correction**
To reconstruct the fully simulated `corrupted_data`, apply the `phase_map` to correct the even lines before matrix multiplication.
```python
# Correct the phase of the even lines
even_lines = corrupted_data[:, 1::2, :].clone()
even_lines *= torch.exp(-1j * phase_map)

# Replace even lines and reconstruct
corrected_data = corrupted_data.clone()
corrected_data[:, 1::2, :] = even_lines

# Final reconstruction
final_reconstruction = torch.matmul(InvA, corrected_data)
```

## Main files to read first
- `demo.ipynb` — best practical overview of intended usage.
- `spenpy/spen.py` — main simulation and reconstruction logic.
- `spenpy/data/brain.png` — sample grayscale input image used in the demo.
- `imgs/sim_example.png` — example visualization.


## Demo notebook workflow
The notebook demonstrates the intended workflow in 4 stages:

1. **Load a grayscale image**
   - opens `spenpy/data/brain.png`
   - converts it to a tensor with `torchvision.transforms.ToTensor()`
   - keeps only the first channel

2. **Build operators and simulate acquisition**
   - `InvA, AFinal = spen().get_InvA()`
   - `final_rxyacq_ROFFT = spen(noise_level=0.0).sim(img.unsqueeze(0))`
   - `phase_map = spen().get_phase_map(img.unsqueeze(0), noise_level=0.)`

3. **Apply phase correction to even lines**
   - extracts even lines from simulated data
   - multiplies them by `exp(-1j * phase_map)`
   - writes corrected lines back
   - compares reconstruction before and after applying `InvA`

4. **Show a faster matrix-based degradation/correction path**
   - creates degraded data with `AFinal @ (img * 1j)`
   - injects phase on even lines
   - corrects the phase and reconstructs with `InvA`
   - compares original, degraded, blurred, corrected, and phase map views

There is also a two-stage example showing:
- stage 1 reconstruction from phase-corrupted data,
- stage 2 reconstruction after explicit even-line phase correction in k-space.

## Core API surface
The main class is `spen` in `spenpy/spen.py`.

### Important methods
- `get_InvA()`
  - returns `(InvA, AFinal)`
  - `InvA` is the approximate inverse / adjoint-style reconstruction matrix
  - `AFinal` is the forward degradation matrix

- `get_phase_map(H, noise_level=0.0)`
  - builds a phase map from a low-frequency image-derived map
  - output shape in the demo is `(1, 128, 256)` for a `256 x 256` acquisition
  - phase is meant for even/odd line inconsistency correction

- `sim(H, return_phase_map=False, return_good_image=False)`
  - runs the SPEN simulation
  - produces phase-corrupted acquisition-domain data transformed into `final_rxyacq_ROFFT`
  - optionally returns the phase map and a “good” image before corruption
