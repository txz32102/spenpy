"""YAML-configurable SPEN forward simulation.

The simulator keeps the original ``spen(...).sim(...)`` return contract but
adds scanner-like randomization hooks that are useful for synthetic training
data.  The defaults are intentionally close to the historical single-shot
path; richer behavior is enabled through a YAML profile.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from spenpy.bruker.param import read_pv_param
from spenpy.core.matrix import calcInvA
from spenpy.fft.transform import fft_kspace_to_xspace, fft_xspace_to_kspace


DEFAULT_SIM_CONFIG: dict[str, Any] = {
    "version": 1,
    "scanner": {
        "L": [4.0, 4.0],
        "acq_point": [256, 256],
        "nseg": 1,
        "chirp_rvalue": 120.0,
        "tblip": 128e-6,
        "gamma_hz": 4257.4,
        "sw_hz": 250000.0,
        "oversample_pe": 16,
        "a_sign": -1,
        "gauss_relative_width": 0.9,
    },
    "randomization": {
        "seed": None,
    },
    "artifacts": {
        "b0": {
            "enabled": False,
            "coef_ranges_cm": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        },
        "shot_phase": {
            "enabled": False,
            "poly_coeff_ranges_rad": [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            "smooth_std_range_rad": [0.0, 0.0],
            "smooth_grid": 6,
        },
        "even_odd": {
            "enabled": True,
            "apply_when_nseg_odd": True,
            "constant_range_rad": [-np.pi, np.pi],
            "linear_range_rad_per_cm": [-np.pi, np.pi],
            "quadratic_range_rad_per_cm2": [0.0, 0.0],
            "object_phase_scale_range_rad": [2 * np.pi, 2 * np.pi],
            "smooth_std_range_rad": [0.0, 0.0],
            "estimate_error_std_rad": 0.0,
        },
        "trajectory": {
            "segment_shift_range_cm": [0.0, 0.0],
            "readout_shift_range_px": [0.0, 0.0],
            "phase_shift_range_px": [0.0, 0.0],
            "line_dropout_probability": 0.0,
            "line_dropout_width": 1,
        },
        "intensity": {
            "gain_range": [1.0, 1.0],
            "bias_field_std_range": [0.0, 0.0],
            "bias_grid": 5,
            "gamma_range": [1.0, 1.0],
        },
        "noise": {
            "complex_std": [0.0, 0.0],
            "relative_to_signal": False,
            "kspace_spike_probability": 0.0,
            "kspace_spike_scale": [0.0, 0.0],
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_sim_config(config: str | Path | dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a simulation config and merge it over ``DEFAULT_SIM_CONFIG``."""
    if config is None:
        return deepcopy(DEFAULT_SIM_CONFIG)
    if isinstance(config, (str, Path)):
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency declared
            raise ModuleNotFoundError(
                "PyYAML is required to load SPEN simulator YAML configs."
            ) from exc
        with open(config, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Simulation config must be a YAML mapping: {config}")
        return _deep_merge(DEFAULT_SIM_CONFIG, loaded)
    if isinstance(config, dict):
        return _deep_merge(DEFAULT_SIM_CONFIG, config)
    raise TypeError(f"Unsupported simulation config type: {type(config)!r}")


def save_sim_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a simulation config as YAML."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency declared
        raise ModuleNotFoundError("PyYAML is required to write simulator YAML configs.") from exc
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_float(value: Any, default: float = 0.0) -> float:
    vals = _as_list(value, [default])
    return float(vals[0]) if vals else default


def _as_int(value: Any, default: int = 1) -> int:
    vals = _as_list(value, [default])
    return int(vals[0]) if vals else default


def _range_tuple(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v
    vals = list(value)
    if len(vals) == 0:
        return default
    if len(vals) == 1:
        v = float(vals[0])
        return v, v
    return float(vals[0]), float(vals[1])


def _normalize_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = torch.abs(x).reshape(x.shape[0], -1)
    scale = flat.max(dim=1).values.clamp_min(eps)
    return x / scale.view(-1, *([1] * (x.ndim - 1)))


class SpenSim:
    """SPEN forward simulator with optional YAML-driven randomization.

    Parameters remain backward-compatible with the original simulator.  Passing
    ``config=...`` or using :meth:`from_yaml` enables richer scanner-like
    artifacts without changing the old ``sim`` return tuple.
    """

    def __init__(
        self,
        L: list[float] | tuple[float, float] | None = None,
        acq_point: list[int] | tuple[int, int] | None = None,
        nseg: int | None = None,
        chirp_rvalue: float | None = None,
        tblip: float | None = None,
        gamma_hz: float | None = None,
        device: str = "cpu",
        noise_level: float | None = None,
        config: str | Path | dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        self.config = load_sim_config(config)
        scanner = self.config["scanner"]

        if L is not None:
            scanner["L"] = [float(v) for v in L]
        if acq_point is not None:
            scanner["acq_point"] = [int(v) for v in acq_point]
        if nseg is not None:
            scanner["nseg"] = int(nseg)
        if chirp_rvalue is not None:
            scanner["chirp_rvalue"] = float(chirp_rvalue)
        if tblip is not None:
            scanner["tblip"] = float(tblip)
        if gamma_hz is not None:
            scanner["gamma_hz"] = float(gamma_hz)
        if noise_level is not None:
            self.config["artifacts"]["noise"]["complex_std"] = [float(noise_level), float(noise_level)]
        if seed is not None:
            self.config["randomization"]["seed"] = int(seed)

        self.L = [float(v) for v in scanner["L"]]
        self.acq_point = [int(v) for v in scanner["acq_point"]]
        self.nseg = int(scanner["nseg"])
        if self.nseg <= 0:
            raise ValueError("nseg must be positive")
        if self.acq_point[1] % self.nseg != 0:
            raise ValueError("acq_point[1] must be divisible by nseg")

        self.chirp_rvalue = float(scanner["chirp_rvalue"])
        self.tblip = float(scanner["tblip"])
        self.gamma_hz = float(scanner["gamma_hz"])
        self.device = device
        self.sw = float(scanner["sw_hz"])
        self.oversample_pe = int(scanner["oversample_pe"])
        self.a_sign = int(scanner["a_sign"])
        self.gauss_relative_width = float(scanner["gauss_relative_width"])
        self.noise_level = float(_range_tuple(self.config["artifacts"]["noise"]["complex_std"])[1])

        self.N = [self.acq_point[0], self.acq_point[1] * self.oversample_pe]
        self.x = torch.linspace(-self.L[0] / 2, self.L[0] / 2, self.N[0], device=device)
        self.y = torch.linspace(-self.L[1] / 2, self.L[1] / 2, self.N[1], device=device)
        self.Ydire_inhomo_coef = torch.zeros(4, device=device)

        self.one_shot_num = self.acq_point[1] / self.nseg
        self.Ta = (self.acq_point[0] / self.sw + self.tblip) * self.one_shot_num
        self.chirp_tp = self.Ta / 2

        self.procpar_struct = {
            "np": self.acq_point[0],
            "ne": 1,
            "nv": self.acq_point[1],
            "nseg": self.nseg,
            "nf": self.acq_point[1],
            "arraydim": 1,
            "Rvol": self.chirp_rvalue,
            "lpe": self.L[0],
            "lro": self.L[1],
            "ppe": 0,
            "Tp": self.chirp_tp,
            "Gchip": self.chirp_rvalue / self.chirp_tp / self.L[1] / self.gamma_hz,
        }

        self.rfwdth = self.procpar_struct["Tp"]
        self.GPEe = self.procpar_struct["Gchip"]
        self.alfa = (
            self.a_sign
            * 2
            * np.pi
            * self.gamma_hz
            * self.GPEe
            * self.rfwdth
            / self.procpar_struct["lpe"]
        )

        seed_value = self.config["randomization"].get("seed")
        self._generator = None
        if seed_value is not None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(seed_value))

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> "SpenSim":
        """Build a simulator from a YAML config file."""
        return cls(config=path, **kwargs)

    @classmethod
    def from_bruker_scan(
        cls,
        scan_dir: str | Path,
        config: str | Path | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "SpenSim":
        """Initialize scanner dimensions from a Bruker SPEN ``method`` file.

        The YAML/default config still controls artifact randomization.  Bruker
        values populate the deterministic sequence geometry when available.
        """
        cfg = load_sim_config(config)
        scanner = cfg["scanner"]
        scan_dir = str(scan_dir)

        matrix = [int(v) for v in _as_list(read_pv_param(scan_dir, "PVM_Matrix"), scanner["acq_point"])]
        while len(matrix) < 2:
            matrix.append(matrix[-1])
        fov_cm = [float(v) for v in _as_list(read_pv_param(scan_dir, "PVM_FovCm"), scanner["L"])]
        while len(fov_cm) < 2:
            fov_cm.append(fov_cm[-1])

        scanner["acq_point"] = matrix[:2]
        scanner["L"] = fov_cm[:2]
        scanner["nseg"] = _as_int(read_pv_param(scan_dir, "NSegments"), scanner["nseg"])

        spen_gy = read_pv_param(scan_dir, "SpenGyGaussStren")
        tp_ms = read_pv_param(scan_dir, "SpatEncDuration")
        if spen_gy is not None and tp_ms is not None:
            tp_s = _as_float(tp_ms) / 1000
            scanner["chirp_rvalue"] = _as_float(spen_gy) * tp_s * scanner["L"][1] * scanner["gamma_hz"]

        echo_spacing_ms = read_pv_param(scan_dir, "PVM_EpiEchoSpacing")
        if echo_spacing_ms is not None:
            scanner["tblip"] = _as_float(echo_spacing_ms) / 1000

        return cls(config=cfg, **kwargs)

    def get_InvA(self):
        """Get the reconstruction operator."""
        inv_a, a_final = calcInvA(
            self.alfa,
            self.L[0],
            self.N[0],
            0,
            self.a_sign,
            0,
            self.gauss_relative_width,
        )
        return inv_a.to(torch.complex64), a_final.to(torch.complex64)

    def sample_config(self) -> dict[str, Any]:
        """Return the effective merged config used by this simulator."""
        return deepcopy(self.config)

    def _rand(self, shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.rand(shape, device=self.device, generator=self._generator)

    def _randn(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(shape, device=self.device, dtype=dtype, generator=self._generator)

    def _uniform(self, value: Any) -> float:
        lo, hi = _range_tuple(value)
        if lo == hi:
            return lo
        return (lo + (hi - lo) * self._rand(()).item())

    def _sample_coeffs(self, ranges: list[Any], n: int) -> torch.Tensor:
        coeffs = [self._uniform(ranges[i] if i < len(ranges) else [0.0, 0.0]) for i in range(n)]
        return torch.tensor(coeffs, device=self.device, dtype=torch.float32)

    def _prepare_input(self, H: torch.Tensor) -> torch.Tensor:
        if H.ndim == 2:
            H = H.unsqueeze(0)
        if H.ndim != 3:
            raise ValueError("H must have shape [B, H, W] or [H, W]")
        H = H.to(self.device)

        def interp(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                x.permute(0, 2, 1).unsqueeze(1),
                size=(self.acq_point[0], self.acq_point[1] * self.oversample_pe),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if torch.is_complex(H):
            return interp(H.real.float()) + 1j * interp(H.imag.float())
        return interp(H.float())

    def _poly2_phase(self, coeffs: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xg = x.view(-1, 1)
        yg = y.view(1, -1)
        terms = [
            torch.ones_like(xg * yg),
            xg.expand(-1, y.numel()),
            yg.expand(x.numel(), -1),
            xg.square().expand(-1, y.numel()),
            (xg * yg),
            yg.square().expand(x.numel(), -1),
        ]
        return sum(coeffs[i] * terms[i] for i in range(min(len(coeffs), len(terms))))

    def _smooth_random_map(self, batch: int, h: int, w: int, std: float, grid: int) -> torch.Tensor:
        if std == 0:
            return torch.zeros((batch, h, w), device=self.device)
        grid = max(2, int(grid))
        noise = self._randn((batch, 1, grid, grid))
        smooth = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
        smooth = smooth - smooth.mean(dim=(1, 2), keepdim=True)
        smooth_std = smooth.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return smooth / smooth_std * std

    def _apply_intensity_model(self, Hb: torch.Tensor) -> torch.Tensor:
        intensity = self.config["artifacts"]["intensity"]
        mag = torch.abs(Hb) if torch.is_complex(Hb) else Hb.clamp_min(0)

        gamma = self._uniform(intensity.get("gamma_range", [1.0, 1.0]))
        if gamma != 1.0:
            mag_max = mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            mag = (mag / mag_max).clamp_min(0).pow(gamma) * mag_max
            if torch.is_complex(Hb):
                Hb = mag * torch.exp(1j * torch.angle(Hb))
            else:
                Hb = mag

        bias_std = self._uniform(intensity.get("bias_field_std_range", [0.0, 0.0]))
        if bias_std:
            bias = self._smooth_random_map(
                Hb.shape[0],
                Hb.shape[1],
                Hb.shape[2],
                bias_std,
                intensity.get("bias_grid", 5),
            )
            Hb = Hb * torch.exp(bias)

        gain = self._uniform(intensity.get("gain_range", [1.0, 1.0]))
        return Hb * gain

    def _b0_coeffs(self) -> torch.Tensor:
        b0 = self.config["artifacts"]["b0"]
        if not b0.get("enabled", False):
            return self.Ydire_inhomo_coef
        return self._sample_coeffs(b0.get("coef_ranges_cm", []), 4)

    def _encode_clean_lr(self, Hb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        batch = Hb.shape[0]
        n_pe_per_seg = self.acq_point[1] // self.nseg
        encoded = torch.zeros(
            (batch, self.acq_point[1], self.acq_point[0]),
            dtype=torch.complex64,
            device=self.device,
        )
        good_lr = torch.zeros_like(encoded)
        shot_phase = torch.zeros(
            (batch, self.nseg, self.acq_point[0], n_pe_per_seg),
            dtype=torch.float32,
            device=self.device,
        )
        sampled: dict[str, Any] = {"segment_shift_cm": [], "b0_coeffs_cm": [], "shot_phase_coeffs_rad": []}

        traj = self.config["artifacts"]["trajectory"]
        shot = self.config["artifacts"]["shot_phase"]

        for k in range(self.nseg):
            seg_shift = self._uniform(traj.get("segment_shift_range_cm", [0.0, 0.0]))
            sampled["segment_shift_cm"].append(seg_shift)

            start = -self.L[1] / 2 + k * self.L[1] / self.acq_point[1] + seg_shift
            step = self.L[1] / n_pe_per_seg
            temp_yacq = start + torch.arange(n_pe_per_seg, device=self.device) * step

            coeff = self._b0_coeffs()
            sampled["b0_coeffs_cm"].append([float(v) for v in coeff.detach().cpu()])
            b0y = sum(coeff[i] * self.y**i for i in range(4))
            b0acq = sum(coeff[i] * temp_yacq**i for i in range(4))

            y_grid, temp_yacq_grid = torch.meshgrid(self.y, temp_yacq, indexing="ij")
            b0y_grid, b0acq_grid = torch.meshgrid(b0y, b0acq, indexing="ij")
            part1 = (y_grid + b0y_grid) - (temp_yacq_grid + b0acq_grid)
            part2 = temp_yacq_grid + b0acq_grid
            exp_term = torch.exp(1j * self.alfa * (part1.square() - part2.square())).to(torch.complex64)

            acquired = torch.matmul(Hb.to(torch.complex64), exp_term)
            good_lr[:, k::self.nseg, :] = acquired.permute(0, 2, 1)

            if shot.get("enabled", False):
                coeffs = self._sample_coeffs(shot.get("poly_coeff_ranges_rad", []), 6)
                smooth_std = self._uniform(shot.get("smooth_std_range_rad", [0.0, 0.0]))
                phase = self._poly2_phase(coeffs, self.x, temp_yacq).unsqueeze(0)
                phase = phase + self._smooth_random_map(
                    batch,
                    self.acq_point[0],
                    n_pe_per_seg,
                    smooth_std,
                    shot.get("smooth_grid", 6),
                )
                acquired = acquired * torch.exp(1j * phase)
                shot_phase[:, k, :, :] = phase
                sampled["shot_phase_coeffs_rad"].append([float(v) for v in coeffs.detach().cpu()])
            else:
                sampled["shot_phase_coeffs_rad"].append([0.0] * 6)

            encoded[:, k::self.nseg, :] = acquired.permute(0, 2, 1)

        return encoded, good_lr, {"shot_phase_map": shot_phase, "sampled": sampled}

    def _object_low_frequency_map(self, H: torch.Tensor, h: int, w: int) -> torch.Tensor:
        base = torch.abs(H.to(self.device))
        if base.ndim == 2:
            base = base.unsqueeze(0)
        base = F.interpolate(base.float().unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        blur_h = max(3, min(31, (h // 8) * 2 + 1))
        blur_w = max(3, min(31, (w // 8) * 2 + 1))
        base = F.avg_pool2d(base.unsqueeze(1), kernel_size=(blur_h, blur_w), stride=1, padding=(blur_h // 2, blur_w // 2)).squeeze(1)
        base = base - base.amin(dim=(1, 2), keepdim=True)
        base = base / base.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return base

    def _build_even_odd_phase(self, H: torch.Tensor, target_h: int, target_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        even_odd = self.config["artifacts"]["even_odd"]
        batch = H.shape[0] if H.ndim == 3 else 1
        if not even_odd.get("enabled", True):
            z = torch.zeros((batch, target_h, target_w), device=self.device)
            return z, z
        if even_odd.get("apply_when_nseg_odd", True) and self.nseg % 2 == 0:
            z = torch.zeros((batch, target_h, target_w), device=self.device)
            return z, z

        x = torch.linspace(-self.L[0] / 2, self.L[0] / 2, target_w, device=self.device)
        constant = self._uniform(even_odd.get("constant_range_rad", [0.0, 0.0]))
        linear = self._uniform(even_odd.get("linear_range_rad_per_cm", [0.0, 0.0]))
        quadratic = self._uniform(even_odd.get("quadratic_range_rad_per_cm2", [0.0, 0.0]))
        phase = constant + linear * x + quadratic * x.square()
        phase = phase.view(1, 1, target_w).expand(batch, target_h, target_w).clone()

        object_scale = self._uniform(even_odd.get("object_phase_scale_range_rad", [0.0, 0.0]))
        if object_scale:
            phase = phase + object_scale * self._object_low_frequency_map(H, target_h, target_w)

        smooth_std = self._uniform(even_odd.get("smooth_std_range_rad", [0.0, 0.0]))
        if smooth_std:
            phase = phase + self._smooth_random_map(batch, target_h, target_w, smooth_std, 6)

        estimate = phase.clone()
        err_std = float(even_odd.get("estimate_error_std_rad", 0.0) or 0.0)
        if err_std:
            estimate = estimate + self._randn(tuple(estimate.shape)) * err_std
        return phase.float(), estimate.float()

    def _fourier_shift(self, x: torch.Tensor, shift_px: float, dim: int) -> torch.Tensor:
        if shift_px == 0:
            return x
        n = x.shape[dim]
        freq = torch.fft.fftfreq(n, device=x.device)
        shape = [1] * x.ndim
        shape[dim] = n
        ramp = torch.exp(-2j * torch.pi * shift_px * freq.reshape(shape))
        k = torch.fft.fft(x, dim=dim)
        return torch.fft.ifft(k * ramp, dim=dim).to(torch.complex64)

    def _apply_trajectory_model(self, encoded: torch.Tensor) -> torch.Tensor:
        traj = self.config["artifacts"]["trajectory"]
        read_shift = self._uniform(traj.get("readout_shift_range_px", [0.0, 0.0]))
        phase_shift = self._uniform(traj.get("phase_shift_range_px", [0.0, 0.0]))
        encoded = self._fourier_shift(encoded, read_shift, dim=2)
        encoded = self._fourier_shift(encoded, phase_shift, dim=1)
        return encoded

    def _add_noise_and_kspace_events(self, kspace: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        noise = self.config["artifacts"]["noise"]
        std = self._uniform(noise.get("complex_std", [0.0, 0.0]))
        if noise.get("relative_to_signal", False):
            signal_scale = torch.abs(kspace).reshape(kspace.shape[0], -1).std(dim=1).clamp_min(1e-8)
            std_tensor = signal_scale.view(-1, 1, 1) * std
        else:
            std_tensor = torch.as_tensor(std, device=self.device)
        if std:
            real = self._randn(tuple(kspace.shape))
            imag = self._randn(tuple(kspace.shape))
            kspace = kspace + (real + 1j * imag).to(torch.complex64) * std_tensor

        spike_probability = float(noise.get("kspace_spike_probability", 0.0) or 0.0)
        spikes = 0
        if spike_probability and self._rand(()).item() < spike_probability:
            scale = self._uniform(noise.get("kspace_spike_scale", [0.0, 0.0]))
            batch_idx = torch.arange(kspace.shape[0], device=self.device)
            pe_idx = torch.randint(0, kspace.shape[1], (kspace.shape[0],), device=self.device, generator=self._generator)
            ro_idx = torch.randint(0, kspace.shape[2], (kspace.shape[0],), device=self.device, generator=self._generator)
            phase = torch.exp(1j * 2 * torch.pi * self._rand((kspace.shape[0],)))
            kspace[batch_idx, pe_idx, ro_idx] += scale * phase
            spikes = int(kspace.shape[0])

        traj = self.config["artifacts"]["trajectory"]
        dropout_probability = float(traj.get("line_dropout_probability", 0.0) or 0.0)
        dropped_lines = 0
        if dropout_probability and self._rand(()).item() < dropout_probability:
            width = max(1, int(traj.get("line_dropout_width", 1)))
            center = int(torch.randint(0, kspace.shape[1], (), device=self.device, generator=self._generator).item())
            lo = max(0, center - width // 2)
            hi = min(kspace.shape[1], lo + width)
            kspace[:, lo:hi, :] = 0
            dropped_lines = hi - lo

        return kspace, {"noise_std": float(std), "spikes": spikes, "dropped_lines": dropped_lines}

    @torch.no_grad()
    def get_phase_map(self, H: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        """Generate an even/odd phase-map estimate for ``H``.

        ``noise_level`` is an additional estimate perturbation in radians,
        kept for compatibility with the historical API.
        """
        phase_true, phase_est = self._build_even_odd_phase(H, self.acq_point[1] // 2, self.acq_point[0])
        if noise_level:
            phase_est = phase_est + self._randn(tuple(phase_est.shape)) * float(noise_level)
        return phase_est

    @torch.no_grad()
    def sim(
        self,
        H: torch.Tensor,
        return_phase_map: bool = False,
        return_good_lr_image: bool = False,
        return_metadata: bool = False,
    ):
        """Forward SPEN simulation.

        Args:
            H: input image ``[B, H, W]`` or ``[H, W]``.
            return_phase_map: include the even-line phase estimate.
            return_good_lr_image: include the clean low-resolution SPEN image.
            return_metadata: include truth maps and sampled artifact values.
        """
        if H.ndim == 2:
            H = H.unsqueeze(0)
        H = H.to(self.device)
        Hb = self._prepare_input(H)
        Hb = self._apply_intensity_model(Hb)

        encoded, good_lr_image, encode_meta = self._encode_clean_lr(Hb)
        good_lr_image = _normalize_abs(good_lr_image)

        encoded = self._apply_trajectory_model(encoded)
        phase_true, phase_estimate = self._build_even_odd_phase(H, encoded.shape[1] // 2, encoded.shape[2])
        if phase_true.numel():
            encoded[:, 1::2, :] = encoded[:, 1::2, :] * torch.exp(1j * phase_true)

        kspace = fft_xspace_to_kspace(encoded, dim=1)
        kspace = _normalize_abs(kspace)
        kspace, noise_meta = self._add_noise_and_kspace_events(kspace)
        final_rxyacq_rofft = fft_kspace_to_xspace(kspace, dim=1)

        metadata = {
            "config": self.sample_config(),
            "sampled": encode_meta["sampled"],
            "noise": noise_meta,
            "phase_map_true": phase_true,
            "phase_map_estimate": phase_estimate,
            "shot_phase_map": encode_meta["shot_phase_map"],
        }

        outputs: list[Any] = [final_rxyacq_rofft]
        if return_phase_map:
            outputs.append(phase_estimate)
        if return_good_lr_image:
            outputs.append(good_lr_image)
        if return_metadata:
            outputs.append(metadata)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


__all__ = ["DEFAULT_SIM_CONFIG", "SpenSim", "load_sim_config", "save_sim_config"]
