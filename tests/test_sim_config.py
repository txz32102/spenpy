"""Tests for the YAML-configurable SPEN simulator."""

from pathlib import Path

import torch
import pytest

from spenpy.sim import SpenSim, load_sim_config
from spenpy.spen import spen


def test_sim_config_dict_preserves_legacy_return_shapes():
    cfg = {
        "scanner": {"L": [2.0, 2.0], "acq_point": [32, 32], "oversample_pe": 4},
        "randomization": {"seed": 123},
        "artifacts": {
            "even_odd": {
                "constant_range_rad": [0.1, 0.1],
                "linear_range_rad_per_cm": [0.0, 0.0],
                "object_phase_scale_range_rad": [0.0, 0.0],
            },
            "noise": {"complex_std": [0.0, 0.0]},
        },
    }
    sim = SpenSim(config=cfg)
    H = torch.rand(2, 32, 32)

    corrupted, phase_map, good_lr = sim.sim(
        H,
        return_phase_map=True,
        return_good_lr_image=True,
    )

    assert corrupted.shape == (2, 32, 32)
    assert corrupted.dtype == torch.complex64
    assert phase_map.shape == (2, 16, 32)
    assert phase_map.dtype == torch.float32
    assert good_lr.shape == (2, 32, 32)
    assert good_lr.dtype == torch.complex64


def test_sim_can_load_yaml_and_return_metadata(tmp_path: Path):
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text(
        """
scanner:
  L: [2.0, 2.0]
  acq_point: [24, 24]
  oversample_pe: 4
randomization:
  seed: 5
artifacts:
  b0:
    enabled: true
    coef_ranges_cm:
      - [0.001, 0.001]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
  shot_phase:
    enabled: true
    poly_coeff_ranges_rad:
      - [0.2, 0.2]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
      - [0.0, 0.0]
  noise:
    complex_std: [0.0, 0.0]
""",
        encoding="utf-8",
    )

    sim = SpenSim.from_yaml(cfg_path)
    out, phase_map, meta = sim.sim(torch.rand(1, 24, 24), return_phase_map=True, return_metadata=True)

    assert out.shape == (1, 24, 24)
    assert phase_map.shape == (1, 12, 24)
    assert meta["phase_map_true"].shape == (1, 12, 24)
    assert meta["shot_phase_map"].shape == (1, 1, 24, 24)
    assert meta["sampled"]["b0_coeffs_cm"][0][0] == pytest.approx(0.001)


def test_legacy_spen_name_accepts_config():
    sim = spen(
        L=[2.0, 2.0],
        acq_point=[24, 24],
        config={"scanner": {"oversample_pe": 4}},
        seed=1,
    )
    out = sim.sim(torch.rand(1, 24, 24))
    assert out.shape == (1, 24, 24)


def test_packaged_scanner_like_config_loads():
    cfg = load_sim_config(Path("spenpy/configs/scanner_like.yaml"))
    assert cfg["scanner"]["acq_point"] == [96, 96]
    assert cfg["artifacts"]["b0"]["enabled"] is True
