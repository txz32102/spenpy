# SPENPy

**SPENPy** is a Python library for **SPEN (Spatiotemporally-Encoded) MRI
reconstruction**, with two complementary use cases under one roof:

1. **Simulation** — generate SPEN-encoded / phase-corrupted images from a
   ground-truth picture, build the encoding matrices `AFinal` / `InvA`,
   and study reconstruction trade-offs. Useful for training deep
   ghost-correction networks and for teaching. Entry point:
   [`demo.ipynb`](demo.ipynb).
2. **Real Bruker PV360 reconstruction** — read raw `fid` /
   `2dseq` data acquired with a Bruker PV360 scanner, run the same
   reconstruction MATLAB does in
   [`spen_matlab/pv360.m`](../spen_matlab/pv360.m), and write
   MATLAB-compatible `.mat` files plus PNG figures. Entry point:
   [`demo/`](demo/) (markdown + four annotated scripts) and the
   `spenpy.cli.pv360_full` CLI.

> The MATLAB MEX kernel `FGG_Convolution1D.c` used inside MATLAB's
> NUFFT step is reimplemented in **pure Python** in
> [`spenpy/recon/gridding.py`](spenpy/recon/gridding.py). Numerically
> faithful to the MATLAB output (`abs_corr > 0.999` against the
> reference `.mat`), at the cost of some speed.

---

## Installation

The project ships as a normal Python package. From the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -e ./spenpy
```

The minimum runtime dependencies (resolved from `pyproject.toml`) are:

| Package | Purpose |
| --- | --- |
| `numpy>=1.24` | Numerical arrays |
| `torch>=2.0` | Encoding matrix math, GPU acceleration, `mult_mat_tensor` |
| `scipy>=1.10` | `.mat` I/O, smoothing splines used in EPI regridding |
| `pillow>=9.0` | Demo image loading |

Optional extras:

```bash
.venv/bin/pip install -e ./spenpy[dev]   # adds pytest + pytest-cov
.venv/bin/pip install matplotlib         # required for the demos
```

Two console scripts are exposed:

```bash
spenpy             # = python -m spenpy.cli.pv360            (no figures)
spenpy-pv360-full  # = python -m spenpy.cli.pv360_full       (with PNG figures)
```

---

## Quick start

### A. Reconstruct real Bruker PV360 SPEN data

This is the closest analogue to running `pv360.m` on a Bruker dataset.
Given a study folder with `datalist.txt` / `<scan_id>/fid` /
`<scan_id>/method` / `<scan_id>/pdata/1/2dseq` files, dump everything to
`/tmp/pv360_python`:

```bash
python -m spenpy.cli.pv360_full \
    --file-dir <bruker_study_dir> \
    --output   /tmp/pv360_python
```

Output layout:

```text
/tmp/pv360_python/
├── ratbrain_RARE.mat
├── ratbrain_EPI.mat
├── ratbrain_SPEN_96_1.mat ... ratbrain_SPEN_96_<N>.mat
├── pv360_full_summary.json
└── figures/
    ├── ratbrain_RARE.png
    ├── ratbrain_EPI.png
    └── ratbrain_SPEN_96_1.png ... ratbrain_SPEN_96_<N>.png
```

Equivalent programmatic API:

```python
from spenpy.cli.pv360_full import run_pv360_full

run_pv360_full(
    file_dir="<bruker_study_dir>",
    export_dir="/tmp/pv360_python",
    save_figures=True,
)
```

For more options (single SPEN index, skip-existing, continue-on-error,
in-process figure generation, side-by-side parity check against MATLAB),
see the four illustrative demo scripts in [`demo/`](demo/) and their
[`README.md`](demo/README.md).

### B. Simulate SPEN data from a ground-truth image

```python
import torch, torchvision.transforms as T
from PIL import Image
from spenpy.spen import spen

img = T.ToTensor()(Image.open("spenpy/data/brain.png"))[0]  # (H, W) grayscale

# Encoding matrices
InvA, AFinal = spen(acq_point=list(img.shape)).get_InvA()

# Full simulation (corrupted ROFFT image + phase estimate + clean reference)
corrupted, phase_map, good_lr = spen(noise_level=0.01).sim(
    img.unsqueeze(0),
    return_phase_map=True,
    return_good_lr_image=True,
)

# Recon — order matters: phase correction first, then InvA
even = corrupted[:, 1::2, :] * torch.exp(-1j * phase_map)
corr = corrupted.clone()
corr[:, 1::2, :] = even
recon = torch.matmul(InvA, corr)
```

The full walkthrough — including a discussion of *why* `InvA @ AFinal`
is not the identity matrix and *why* phase correction must precede
`InvA` — is in [`demo.ipynb`](demo.ipynb).

---

## Repository layout

```text
spenpy/
├── README.md                 ← you are here
├── demo.ipynb                ← simulation walkthrough (runnable Jupyter notebook)
├── demo/                     ← real Bruker PV360 reconstruction demos
│   ├── README.md             ← detailed walkthrough + pv360.m mapping
│   ├── 01_run_pv360_pipeline.py       ← full pv360.m mimic
│   ├── 02_single_spen_reconstruction.py
│   ├── 03_inspect_intermediate_steps.py
│   └── 04_compare_with_matlab.py
├── spenpy/                   ← package source
│   ├── spen.py               ← encoding matrices + simulation
│   ├── bruker/               ← PV360 raw fid / 2dseq / param readers
│   │   ├── image.py          ← read 2dseq + visu_pars (RARE / EPI)
│   │   ├── param.py          ← Bruker parameter file parser
│   │   └── raw.py            ← read fid + reshape (multi-channel, multi-slice)
│   ├── core/                 ← SR encoding-matrix builders
│   │   ├── matrix.py         ← calcSRMatrixApprox, calcInvA
│   │   └── sinc.py
│   ├── fft/transform.py      ← MATLAB-style fftshift(fft(ifftshift(.))) wrappers
│   ├── recon/                ← reconstruction pipeline
│   │   ├── gridding.py       ← 1D regridding + pure-Python NUFFT
│   │   ├── phase.py          ← per-shot motion phase correction
│   │   └── spen_recon.py     ← main reconstruct_odd_segments() + orient
│   ├── utils/                ← coil combine, zero fill, polyfit, smoothing
│   ├── sim/spen_sim.py       ← simulation helpers
│   └── cli/                  ← pv360.m equivalents
│       ├── pv360.py          ← run_pv360 (.mat only)
│       └── pv360_full.py     ← run_pv360_full (.mat + PNG)
├── tests/                    ← pytest suite (see "Testing" below)
├── pyproject.toml
├── setup.py
└── LICENSE                   ← MIT
```

---

## Module map

| Layer | Module | Highlights |
| --- | --- | --- |
| Bruker I/O | `spenpy.bruker.image` | `read_bruker_2dseq` — RARE / EPI 2dseq with `visu_pars` parsing, slope/offset rescaling, Fortran reshape order. |
| Bruker I/O | `spenpy.bruker.raw` | `read_bruker_kspace_pv360_fid_multichannel` — multi-channel `fid` reader matching `ReadBrukerkSpace_PV360_fid_multichannel.m` byte-for-byte. |
| Bruker I/O | `spenpy.bruker.param` | `read_pv_param` — read any field from `acqp` / `method` / `visu_pars`. |
| Encoding | `spenpy.core.matrix` | `calcSRMatrixApprox`, `calcInvA` — closed-form SR matrix and weighted adjoint. |
| FFT | `spenpy.fft.transform` | `fft_kspace_to_xspace` / `fft_xspace_to_kspace` (PyTorch, MATLAB convention). |
| Reconstruction | `spenpy.recon.gridding` | `one_d_regridding_pv360` + `fgg_1d_type1` — pure-Python NUFFT replacing the MATLAB MEX. |
| Reconstruction | `spenpy.recon.phase` | One-shot odd-segment phase correction (`apply_pv360_one_shot_phase_correction`). |
| Reconstruction | `spenpy.recon.spen_recon` | `reconstruct_odd_segments` — top-level pipeline + `orient_pv360_spen_image`. |
| Utilities | `spenpy.utils.tensor` | `mult_mat_tensor` — batched matrix-tensor product (replaces MATLAB `MultMatTensor`). |
| Utilities | `spenpy.utils.coil_combine` | `coil_combine` — adaptive coil combination port of `coilCombinebao.m`. |
| Utilities | `spenpy.utils.zero_fill` | `zero_filling_pv6`, `rm_zero_filling_pv6`. |
| Simulation | `spenpy.spen` | `spen` class — `sim`, `get_InvA`, `get_phase_map`. |
| CLI | `spenpy.cli.pv360` | Programmatic `run_pv360`, `process_rare_epi`, `process_spen`, `read_datalist`. |
| CLI | `spenpy.cli.pv360_full` | `run_pv360_full` — same plus PNG figures and richer arg parsing. |

---

## MATLAB ↔ Python mapping (full pipeline)

| MATLAB step (`pv360.m` / `Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV360.m`) | Python equivalent |
| --- | --- |
| `addpath(genpath('spen')); addpath(genpath('pvtools'));` | `import spenpy` |
| `datalist = textread('datalist.txt', '%u');` | `spenpy.cli.pv360.read_datalist` |
| `imageObj = ImageDataObject(...).data; permute([2 1 3 4])` | `spenpy.bruker.image.read_bruker_2dseq` + axis swap inside `process_rare_epi` |
| `ReadBrukerkSpace_PV360_fid_multichannel(spen_dir)` | `spenpy.bruker.raw.read_bruker_kspace_pv360_fid_multichannel` |
| `oneD_regriding_PV360(kField, EpiTraj, NSeg, MS)` | `spenpy.recon.gridding.one_d_regridding_pv360` |
| `FGG_1d_type1(...)` + `FGG_Convolution1D.mexa64` | `spenpy.recon.gridding.fgg_1d_type1` (pure-Python NUFFT — no MEX) |
| `Zero_filling_PV6(kField, ZF)` | `spenpy.utils.zero_fill.zero_filling_pv6` |
| `FFTKSpace2XSpace(CmplxData, 2)` | `spenpy.fft.transform.fft_kspace_to_xspace(..., dim=1)` |
| `[InvA, AFinal] = calcInvA(...)` | `spenpy.core.matrix.calcInvA` |
| `MultMatTensor(InvA, ROFFTedData)` | `spenpy.utils.tensor.mult_mat_tensor` |
| `coilCombinebao(...)` | `spenpy.utils.coil_combine.coil_combine` |
| `Image_SPEN = flip(flip(images, 1), 2)` | `spenpy.recon.spen_recon.orient_pv360_spen_image` |
| `imshow3Dfull(...)`, `imagesc(...)` | `matplotlib` figures via `save_volume_figure` / `save_spen_figure` |
| `save(fullfile(...), 'Imag_low', 'Imag_origin', 'Image_SPEN', 'SPEN_AZ')` | `scipy.io.savemat(...)` inside `process_spen` |

### Orientation note

`pv360.m` only applies `flip(flip(.,1),2)` to `Image_SPEN`, leaving
`Imag_low` / `Imag_origin` in the raw orientation, so the three panels
in the saved `.mat` are 180° rotated relative to each other. SPENPy
applies the same flip to **all three** images so figures rendered from
the resulting `.mat` files share a consistent anatomical orientation.
If you need the unflipped low-resolution arrays for byte-exact MATLAB
compatibility, call `spenpy.recon.reconstruct_odd_segments(...)`
directly and use `recon.imag_low` / `recon.imag_origin`, which are the
pre-flip tensors.

---

## Testing

```bash
.venv/bin/pip install pytest
.venv/bin/pytest spenpy/tests             # unit tests only (no MATLAB needed)
.venv/bin/pytest spenpy/tests/test_pv360_matlab_parity.py
                                           # ↑ requires matlab on PATH + Bruker data
```

| Test file | What it covers | MATLAB needed? |
| --- | --- | --- |
| `tests/test_core.py` | `calcSRMatrixApprox`, `calcInvA` shape / dtype / weighted-adjoint identity | No |
| `tests/test_fft.py` | `fft_kspace_to_xspace` / `fft_xspace_to_kspace` round-trip & MATLAB-shift convention | No |
| `tests/test_tensor.py` | `mult_mat_tensor` (2D/3D/4D); `zero_filling_pv6` round-trip | No |
| `tests/test_pv360_matlab_parity.py` | Bit-exact equality of `read_datalist`, `process_rare_epi` and `read_bruker_kspace_pv360_fid_multichannel` against MATLAB output. SPEN reconstruction: `abs_corr > 0.999` for `Imag_low` / `Imag_origin` / `Image_SPEN`. | Yes |
| `tests/verify_pv360_reconstruction.py` | Visual + numeric verification harness (not collected by pytest). Renders Python vs MATLAB panels and writes `*_metrics.json`. | Optional |

---

## Demos at a glance

| File | Purpose |
| --- | --- |
| [`demo.ipynb`](demo.ipynb) | Simulation walkthrough on a 2D image: build `AFinal`/`InvA`, run `sim()`, demonstrate phase correction (right vs wrong order), inspect `\|InvA @ AFinal\|`, fast forward path. |
| [`demo/01_run_pv360_pipeline.py`](demo/01_run_pv360_pipeline.py) | Full pv360.m mimic — RARE + EPI + every SPEN scan → `.mat` + PNG. |
| [`demo/02_single_spen_reconstruction.py`](demo/02_single_spen_reconstruction.py) | Reconstruct one SPEN scan; inspect `Imag_low` / `Imag_origin` / `Image_SPEN`. |
| [`demo/03_inspect_intermediate_steps.py`](demo/03_inspect_intermediate_steps.py) | Multi-panel view: raw k-space → regrid → readout-FFT → SR matrix → final image. |
| [`demo/04_compare_with_matlab.py`](demo/04_compare_with_matlab.py) | Numeric + visual parity check against an existing MATLAB `.mat`. |

---

## License

MIT — see [`LICENSE`](LICENSE).

## References

* Liu et al., *Unsupervised deep learning model for correcting Nyquist
  ghosts of single-shot spatiotemporal encoding*, included in this
  folder as
  [`Liu et al. - Unsupervised deep learning model …`](Liu%20et%20al.%20-%20Unsupervised%20deep%20learning%20model%20for%20correcting%20Nyquist%20ghosts%20of%20single‐shot%20spatiotemporal%20encodin.pdf).
* MATLAB reference: [`spen_matlab/SPEN_RECONSTRUCTION_GUIDE.md`](../spen_matlab/SPEN_RECONSTRUCTION_GUIDE.md)
  and [`spen_matlab/pv360.m`](../spen_matlab/pv360.m).
