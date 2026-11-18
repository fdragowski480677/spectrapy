"""Perceptually more uniform colour spaces and colour differences.

All functions accept arrays of shape ``(..., 3)``. XYZ is on the relative scale where the reference
white has ``Y = 1``.
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array
from .adaptation import white_to_xyz

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _f(t):
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _finv(f):
    f3 = f**3
    return np.where(f3 > _EPS, f3, (116.0 * f - 16.0) / _KAPPA)


def xyz_to_lab(xyz, white="D65") -> np.ndarray:
    """CIE 1976 L*a*b*."""
    w = white_to_xyz(white)
    w = w / w[1]
    fx, fy, fz = np.moveaxis(_f(as_float_array(xyz) / w), -1, 0)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def lab_to_xyz(lab, white="D65") -> np.ndarray:
    w = white_to_xyz(white)
    w = w / w[1]
    L, a, b = np.moveaxis(as_float_array(lab), -1, 0)
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    yr = np.where(L > _KAPPA * _EPS, fy**3, L / _KAPPA)
    return np.stack([_finv(fx), yr, _finv(fz)], axis=-1) * w


def _uv_prime(xyz):
    X, Y, Z = np.moveaxis(as_float_array(xyz), -1, 0)
    d = X + 15.0 * Y + 3.0 * Z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(d > 0, 4.0 * X / d, 0.0)
        v = np.where(d > 0, 9.0 * Y / d, 0.0)
    return u, v


def xyz_to_luv(xyz, white="D65") -> np.ndarray:
    """CIE 1976 L*u*v*."""
    w = white_to_xyz(white)
    w = w / w[1]
    xyz = as_float_array(xyz)
    yr = xyz[..., 1] / w[1]
    L = np.where(yr > _EPS, 116.0 * np.cbrt(yr) - 16.0, _KAPPA * yr)
    u, v = _uv_prime(xyz)
    un, vn = _uv_prime(w)
    return np.stack([L, 13.0 * L * (u - un), 13.0 * L * (v - vn)], axis=-1)


def luv_to_xyz(luv, white="D65") -> np.ndarray:
    w = white_to_xyz(white)
    w = w / w[1]
    un, vn = _uv_prime(w)
    L, u, v = np.moveaxis(as_float_array(luv), -1, 0)
    Y = np.where(L > _KAPPA * _EPS, ((L + 16.0) / 116.0) ** 3, L / _KAPPA) * w[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(L != 0, u / (13.0 * L) + un, un)
        vp = np.where(L != 0, v / (13.0 * L) + vn, vn)
        X = np.where(vp != 0, Y * 9.0 * up / (4.0 * vp), 0.0)
        Z = np.where(vp != 0, Y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp), 0.0)
    return np.stack([X, Y, Z], axis=-1)


def lab_to_lch(lab) -> np.ndarray:
    """Cartesian (L, a, b) -> cylindrical (L, C, h[degrees]); works for Lab, Luv and OKLab."""
    L, a, b = np.moveaxis(as_float_array(lab), -1, 0)
    return np.stack([L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360.0], axis=-1)


def lch_to_lab(lch) -> np.ndarray:
    L, C, h = np.moveaxis(as_float_array(lch), -1, 0)
    hr = np.radians(h)
    return np.stack([L, C * np.cos(hr), C * np.sin(hr)], axis=-1)


# OKLab (B. Ottosson 2020), defined for XYZ relative to D65 with Y = 1.
_OK_M1 = np.array([[0.8189330101, 0.3618667424, -0.1288597137],
                   [0.0329845436, 0.9293118715, 0.0361456387],
                   [0.0482003018, 0.2643662691, 0.6338517070]])
_OK_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
_OK_M1_INV = np.linalg.inv(_OK_M1)
_OK_M2_INV = np.linalg.inv(_OK_M2)


def xyz_to_oklab(xyz) -> np.ndarray:
    """OKLab from XYZ (D65 white, Y = 1)."""
    return np.cbrt(as_float_array(xyz) @ _OK_M1.T) @ _OK_M2.T


def oklab_to_xyz(lab) -> np.ndarray:
    return (as_float_array(lab) @ _OK_M2_INV.T) ** 3 @ _OK_M1_INV.T


def delta_e_76(lab1, lab2) -> np.ndarray:
    """Euclidean CIE 1976 colour difference."""
    return np.linalg.norm(as_float_array(lab1) - as_float_array(lab2), axis=-1)


def delta_e_2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), vectorized."""
    L1, a1, b1 = np.moveaxis(as_float_array(lab1), -1, 0)
    L2, a2, b2 = np.moveaxis(as_float_array(lab2), -1, 0)
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cb7 / (Cb7 + 25.0**7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360.0)
    zero = (C1p * C2p) == 0
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(zero, 0.0, dh)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2.0)
    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= 180.0, hsum / 2.0,
                   np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0))
    hbp = np.where(zero, hsum, hbp)
    T = (1.0 - 0.17 * np.cos(np.radians(hbp - 30.0)) + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0)) - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))
    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp**7
    RC = 2.0 * np.sqrt(Cbp7 / (Cbp7 + 25.0**7))
    SL = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC
    tL, tC, tH = dLp / (kL * SL), dCp / (kC * SC), dHp / (kH * SH)
    return np.sqrt(tL**2 + tC**2 + tH**2 + RT * tC * tH)
