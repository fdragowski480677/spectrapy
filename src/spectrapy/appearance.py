"""Colour appearance models CIECAM02 (CIE 159:2004) and CAM16 (Li et al. 2017), with their
uniform colour spaces CAM02-UCS and CAM16-UCS (Luo, Cui & Li 2006).

All functions are vectorized over any batch shape ``(..., 3)``. XYZ values are relative to the
adopted white ``xyz_w`` (any common scale - they are rescaled internally so that ``Y_w = 100``,
as the models expect). Viewing conditions: adapting luminance ``L_A`` [cd/m²], relative
background luminance ``Y_b`` (in % of the white), and the surround (``"average"``, ``"dim"`` or
``"dark"``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._util import as_float_array

M_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                    [-0.7036, 1.6975, 0.0061],
                    [0.0030, 0.0136, 0.9834]])
M_HPE = np.array([[0.38971, 0.68898, -0.07868],
                  [-0.22981, 1.18340, 0.04641],
                  [0.00000, 0.00000, 1.00000]])
M_16 = np.array([[0.401288, 0.650173, -0.051461],
                 [-0.250268, 1.204414, 0.045854],
                 [-0.002079, 0.048952, 0.953127]])
_M_CAT02_INV = np.linalg.inv(M_CAT02)
_M_16_INV = np.linalg.inv(M_16)

#: Surround parameters (F, c, N_c).
SURROUNDS = {"average": (1.0, 0.69, 1.0), "dim": (0.9, 0.59, 0.9), "dark": (0.8, 0.525, 0.8)}


@dataclass
class CAMSpecification:
    """Appearance correlates: lightness ``J``, chroma ``C``, hue angle ``h`` (degrees), colourfulness
    ``M``, saturation ``s``, brightness ``Q``. Each is an array of the input batch shape."""

    J: np.ndarray
    C: np.ndarray
    h: np.ndarray
    M: np.ndarray
    s: np.ndarray
    Q: np.ndarray

    @property
    def JMh(self) -> np.ndarray:
        return np.stack([self.J, self.M, self.h], axis=-1)

    @property
    def JCh(self) -> np.ndarray:
        return np.stack([self.J, self.C, self.h], axis=-1)


def _spow(x, p):
    return np.sign(x) * np.abs(x) ** p


def _compress(rgb, F_L):
    """Post-adaptation non-linear response compression (sign preserving)."""
    f = _spow(F_L * rgb / 100.0, 0.42)
    return 400.0 * f / (27.13 + np.abs(f)) + 0.1


def _decompress(rgb_a, F_L):
    x = rgb_a - 0.1
    return np.sign(x) * (100.0 / F_L) * ((27.13 * np.abs(x)) / (400.0 - np.abs(x))) ** (1.0 / 0.42)


def _conditions(L_A, Y_b, Y_w, surround):
    F, c, N_c = SURROUNDS[surround] if isinstance(surround, str) else surround
    k = 1.0 / (5.0 * L_A + 1.0)
    F_L = 0.2 * k**4 * (5.0 * L_A) + 0.1 * (1.0 - k**4) ** 2 * np.cbrt(5.0 * L_A)
    n = Y_b / Y_w
    N_bb = 0.725 * n**-0.2
    z = 1.48 + np.sqrt(n)
    return F, c, N_c, F_L, n, N_bb, z


def _degree_of_adaptation(F, L_A, discount_illuminant):
    if discount_illuminant:
        return 1.0
    return float(np.clip(F * (1.0 - (1.0 / 3.6) * np.exp((-L_A - 42.0) / 92.0)), 0.0, 1.0))


def _forward(xyz, xyz_w, L_A, Y_b, surround, discount_illuminant, model):
    xyz = as_float_array(xyz)
    xyz_w = as_float_array(xyz_w)
    scale = 100.0 / xyz_w[..., 1:2]
    xyz = xyz * scale
    xyz_w = xyz_w * scale
    Y_w = 100.0
    F, c, N_c, F_L, n, N_bb, z = _conditions(L_A, Y_b, Y_w, surround)
    D = _degree_of_adaptation(F, L_A, discount_illuminant)
    M_cat = M_CAT02 if model == "ciecam02" else M_16
    rgb = xyz @ M_cat.T
    rgb_w = xyz_w @ M_cat.T
    gain = D * Y_w / rgb_w + 1.0 - D
    rgb_c = gain * rgb
    rgb_cw = gain * rgb_w
    if model == "ciecam02":  # to Hunt-Pointer-Estevez cone space
        to_hpe = M_HPE @ _M_CAT02_INV
        rgb_c = rgb_c @ to_hpe.T
        rgb_cw = rgb_cw @ to_hpe.T
    rgb_a = _compress(rgb_c, F_L)
    rgb_aw = _compress(rgb_cw, F_L)
    Ra, Ga, Ba = np.moveaxis(rgb_a, -1, 0)
    a = Ra - 12.0 * Ga / 11.0 + Ba / 11.0
    b = (Ra + Ga - 2.0 * Ba) / 9.0
    h = np.degrees(np.arctan2(b, a)) % 360.0
    e_t = 0.25 * (np.cos(np.radians(h) + 2.0) + 3.8)
    A = (2.0 * Ra + Ga + Ba / 20.0 - 0.305) * N_bb
    A_w = (2.0 * rgb_aw[..., 0] + rgb_aw[..., 1] + rgb_aw[..., 2] / 20.0 - 0.305) * N_bb
    J = 100.0 * (np.maximum(A, 0.0) / A_w) ** (c * z)
    Q = (4.0 / c) * np.sqrt(J / 100.0) * (A_w + 4.0) * F_L**0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (50000.0 / 13.0 * N_c * N_bb) * e_t * np.hypot(a, b) / (Ra + Ga + 21.0 * Ba / 20.0)
    t = np.nan_to_num(t)
    C = np.abs(t) ** 0.9 * np.sqrt(J / 100.0) * (1.64 - 0.29**n) ** 0.73
    M = C * F_L**0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.nan_to_num(100.0 * np.sqrt(M / Q))
    return CAMSpecification(J=J, C=C, h=h, M=M, s=s, Q=Q)


def xyz_to_ciecam02(xyz, xyz_w, L_A: float = 64.0 / np.pi / 5.0, Y_b: float = 20.0, surround="average",
                    discount_illuminant: bool = False) -> CAMSpecification:
    """CIECAM02 appearance correlates of ``xyz`` viewed under white ``xyz_w``.

    The default ``L_A`` corresponds to a 64 lux environment (L_A = E / (5 π)).
    """
    return _forward(xyz, xyz_w, L_A, Y_b, surround, discount_illuminant, "ciecam02")


def xyz_to_cam16(xyz, xyz_w, L_A: float = 64.0 / np.pi / 5.0, Y_b: float = 20.0, surround="average",
                 discount_illuminant: bool = False) -> CAMSpecification:
    """CAM16 appearance correlates (the CIECAM02 successor with a single, better-behaved matrix)."""
    return _forward(xyz, xyz_w, L_A, Y_b, surround, discount_illuminant, "cam16")


def jmh_to_ucs(JMh, c1: float = 0.007, c2: float = 0.0228, K_L: float = 1.0) -> np.ndarray:
    """``(J, M, h)`` -> uniform colour space ``(J', a', b')`` (CAM02-UCS / CAM16-UCS coefficients)."""
    J, M, h = np.moveaxis(as_float_array(JMh), -1, 0)
    Jp = (1.0 + 100.0 * c1) * J / (1.0 + c1 * J) / K_L
    Mp = np.log1p(c2 * M) / c2
    hr = np.radians(h)
    return np.stack([Jp, Mp * np.cos(hr), Mp * np.sin(hr)], axis=-1)


def ucs_to_jmh(Jab, c1: float = 0.007, c2: float = 0.0228, K_L: float = 1.0) -> np.ndarray:
    """Inverse of :func:`jmh_to_ucs`."""
    Jp, ap, bp = np.moveaxis(as_float_array(Jab), -1, 0)
    Jp = Jp * K_L
    J = Jp / (1.0 + 100.0 * c1 - c1 * Jp)
    Mp = np.hypot(ap, bp)
    M = np.expm1(c2 * Mp) / c2
    h = np.degrees(np.arctan2(bp, ap)) % 360.0
    return np.stack([J, M, h], axis=-1)


def xyz_to_cam02ucs(xyz, xyz_w, **conditions) -> np.ndarray:
    """XYZ -> CAM02-UCS ``(J', a', b')``."""
    return jmh_to_ucs(xyz_to_ciecam02(xyz, xyz_w, **conditions).JMh)


def xyz_to_cam16ucs(xyz, xyz_w, **conditions) -> np.ndarray:
    """XYZ -> CAM16-UCS ``(J', a', b')``."""
    return jmh_to_ucs(xyz_to_cam16(xyz, xyz_w, **conditions).JMh)


def cam16_to_xyz(J, M, h, xyz_w, L_A: float = 64.0 / np.pi / 5.0, Y_b: float = 20.0, surround="average",
                 discount_illuminant: bool = False) -> np.ndarray:
    """Inverse CAM16: lightness ``J``, colourfulness ``M`` and hue ``h`` (degrees) -> XYZ, on the scale
    of ``xyz_w``. Useful e.g. to construct colours of given appearance or to gamut-map in CAM16-UCS."""
    xyz_w = as_float_array(xyz_w)
    scale = xyz_w[..., 1:2] / 100.0
    w100 = xyz_w / scale
    F, c, N_c, F_L, n, N_bb, z = _conditions(L_A, Y_b, 100.0, surround)
    D = _degree_of_adaptation(F, L_A, discount_illuminant)
    rgb_w = w100 @ M_16.T
    gain = D * 100.0 / rgb_w + 1.0 - D
    rgb_aw = _compress(gain * rgb_w, F_L)
    A_w = (2.0 * rgb_aw[..., 0] + rgb_aw[..., 1] + rgb_aw[..., 2] / 20.0 - 0.305) * N_bb
    J = as_float_array(J)
    M = as_float_array(M)
    h = as_float_array(h)
    C = M / F_L**0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(J > 0, (C / (np.sqrt(J / 100.0) * (1.64 - 0.29**n) ** 0.73)) ** (1.0 / 0.9), 0.0)
    hr = np.radians(h)
    e_t = 0.25 * (np.cos(hr + 2.0) + 3.8)
    A = A_w * (J / 100.0) ** (1.0 / (c * z))
    p1 = (50000.0 / 13.0) * N_c * N_bb * e_t
    p2 = A / N_bb + 0.305  # = 2 R'a + G'a + B'a / 20
    # from t = p1 |ab| / (R'a + G'a + 21 B'a / 20) with R'a, G'a, B'a linear in (p2, a, b):
    gamma = 23.0 * p2 * t / (23.0 * p1 + 11.0 * t * np.cos(hr) + 108.0 * t * np.sin(hr))
    a = gamma * np.cos(hr)
    b = gamma * np.sin(hr)
    Ra = (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0
    Ga = (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0
    Ba = (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0
    rgb_c = _decompress(np.stack([Ra, Ga, Ba], axis=-1), F_L)
    rgb = rgb_c / gain
    return (rgb @ _M_16_INV.T) * scale


__all__ = ["CAMSpecification", "SURROUNDS", "cam16_to_xyz", "jmh_to_ucs", "ucs_to_jmh", "xyz_to_cam02ucs",
           "xyz_to_cam16", "xyz_to_cam16ucs", "xyz_to_ciecam02"]
