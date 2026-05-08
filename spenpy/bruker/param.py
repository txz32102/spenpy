"""Bruker ParaVision parameter file reader.

Ported from ReadPVParam.m -- reads parameters from method/acqp/reco files.
"""

import re
from pathlib import Path

# Cache for parsed files
_cache: dict[str, dict[str, str | list]] = {}


def _parse_bruker_file(filepath: str) -> str:
    """Read and concatenate all lines from a Bruker parameter file."""
    try:
        with open(filepath, "r") as f:
            return f.read().lower()
    except FileNotFoundError:
        return ""


def _load_param_files(fid_path: str, two_dseq_path: str = "") -> str:
    """Load and concatenate method, acqp, and optionally reco files."""
    import os

    cache_key = fid_path + "|" + two_dseq_path
    if cache_key in _cache:
        return _cache[cache_key]

    dir_path = os.path.dirname(fid_path)
    if not dir_path:
        dir_path = fid_path

    content = ""
    for fname in ["method", "acqp"]:
        fpath = os.path.join(dir_path, fname)
        content += _parse_bruker_file(fpath)

    if two_dseq_path:
        reco_dir = os.path.dirname(two_dseq_path)
        if not reco_dir:
            reco_dir = two_dseq_path
        content += _parse_bruker_file(os.path.join(reco_dir, "reco"))

    _cache[cache_key] = content
    return content


def read_pv_param(fid_path: str, param_name: str, two_dseq_path: str = ""):
    """Read a parameter value from Bruker ParaVision files.

    Args:
        fid_path: path to the fid/rawdata directory
        param_name: name of the parameter (case-insensitive)
        two_dseq_path: optional path to 2dseq directory for reco params

    Returns:
        Numeric value/array, string, or None if not found.
    """
    data = _load_param_files(fid_path, two_dseq_path)

    p_name = "##$" + param_name.lower().strip() + "="

    ind = data.find(p_name)
    if ind == -1:
        return None

    rest = data[ind + len(p_name):]

    # Check for dimension info: (dim1, dim2, ...) -- handle spaces inside parens
    dim_match = re.match(r"\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)", rest)
    p_dim = None
    val_start = 0
    if dim_match:
        p_dim = [int(d.strip()) for d in dim_match.group(1).split(",")]
        val_start = dim_match.end()

    # Find end of value (next ##$ or end of string)
    end_marker = rest.find("##$", val_start)
    if end_marker == -1:
        end_marker = len(rest)
    val_str = rest[val_start:end_marker].strip()

    # Strip trailing $$ comments (Bruker JCAMP-DX format)
    if "$$" in val_str:
        val_str = val_str[:val_str.index("$$")].strip()

    # Try to parse as numeric
    try:
        # Handle space-separated values
        values = [float(v) for v in val_str.split()]
        if p_dim:
            # Reshape according to dimensions (reverse order like MATLAB)
            import numpy as np

            arr = np.array(values)
            arr = arr.reshape(tuple(p_dim[::-1]))
            # Squeeze singleton dimensions
            arr = np.squeeze(arr)
            if arr.ndim > 0:
                return arr.tolist() if arr.ndim == 1 else arr
            return float(arr)
        if len(values) == 1:
            val = values[0]
            return int(val) if val == int(val) else val
        return values
    except (ValueError, TypeError):
        return val_str
