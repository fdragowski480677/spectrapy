"""Chromatic adaptation transforms (von Kries type), vectorized over any batch shape."""
from __future__ import annotations

import numpy as np

from ._util import as_float_array

#: Cone-response-like matrices used by the von Kries family of transforms.
CAT_MATRICES = {
    "bradford": np.array([[0.8951, 0.2664, -0.1614],
                          [-0.7502, 1.7135, 0.0367],
                          [0.0389, -0.0685, 1.0296]]),
    "cat02": np.array([[0.7328, 0.4296, -0.1624],
                       [-0.7036, 1.6975, 0.0061],
                       [0.0030, 0.0136, 0.9834]]),
    "cat16": np.array([[0.401288, 0.650173, -0.051461],
                       [-0.250268, 1.204414, 0.045854],
                       [-0.002079, 0.048952, 0.953127]]),
    "von_kries": np.array([[0.40024, 0.70760, -0.08081],
                           [-0.22630, 1.16532, 0.04570],
                           [0.0, 0.0, 0.91822]]),
    "xyz_scaling": np.eye(3),
}


def white_to_xyz(white) -> np.ndarray:
    """Accept a white point as XYZ (3 values), chromaticity xy (2 values) or a name like ``"D65"``."""
    if isinstance(white, str):
        from .illuminants import WHITE_POINTS

        white = WHITE_POINTS[white.strip().upper()]
    w = as_float_array(white)
    if w.shape[-1] == 2:
        x, y = w[..., 0], w[..., 1]
        return np.stack([x / y, np.ones_like(x), (1.0 - x - y) / y], axis=-1)
    return w


def adaptation_matrix(src_white, dst_white, method: str = "bradford") -> np.ndarray:
    """3x3 matrix mapping XYZ seen under ``src_white`` to corresponding colours under ``dst_white``."""
    M = CAT_MATRICES[method.lower()]
    ws = M @ white_to_xyz(src_white)
    wd = M @ white_to_xyz(dst_white)
    return np.linalg.inv(M) @ np.diag(wd / ws) @ M


def adapt(xyz, src_white, dst_white, method: str = "bradford") -> np.ndarray:
    """Corresponding colours: ``xyz`` (shape ``(..., 3)``) adapted from ``src_white`` to ``dst_white``.

    White points can be XYZ, xy or names; only their chromaticity matters for the transform, so
    ``Y`` of the output follows the input scale.
    """
    ws = white_to_xyz(src_white)
    wd = white_to_xyz(dst_white)
    A = adaptation_matrix(ws / ws[1], wd / wd[1], method)
    return as_float_array(xyz) @ A.T
