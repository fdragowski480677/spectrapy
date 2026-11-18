"""Correlated colour temperature (CCT), Duv and the Planckian / daylight loci.

The Planckian locus is computed by integrating Planck's law against the chosen observer (CIE 015
convention, c2 = 1.4388e-2 m K), so CCT and Duv are consistent with any observer. CCT is found
with a table search plus Newton refinement in mired space - accurate to well below 0.1 K and
fully vectorized over many chromaticities.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._util import as_float_array, trapezoid_weights
from .colorimetry import uv1960_to_xy, xy_to_uv1960, xyz_to_uv1960
from .constants import C2_CIE
from .illuminants import daylight_xy, planck
from .observers import get_observer

T_MIN, T_MAX = 1000.0, 100_000.0


def _weighted_cmf(obs):
    return obs.cmf * trapezoid_weights(obs.wavelengths)[:, None]


def planckian_uv(T, observer="1931") -> np.ndarray:
    """CIE 1960 (u, v) of a Planckian radiator at temperature(s) ``T`` [K] (``np.inf`` allowed)."""
    obs = get_observer(observer)
    T = as_float_array(T)
    Tf = np.where(np.isinf(T), 1e12, T)
    B = planck(obs.wavelengths, Tf[..., None], c2=C2_CIE)
    B = B / np.max(B, axis=-1, keepdims=True)
    return xyz_to_uv1960(B @ _weighted_cmf(obs))


def planckian_xy(T, observer="1931") -> np.ndarray:
    """CIE xy of a Planckian radiator at temperature(s) ``T`` [K]."""
    return uv1960_to_xy(planckian_uv(T, observer))


def _make_table(obs):
    # a little beyond [T_MIN, T_MAX] so the cubic interpolation never extrapolates
    mired = np.linspace(1e6 / (1.25 * T_MAX), 1e6 / (0.9 * T_MIN), 4500)
    return mired, planckian_uv(1e6 / mired, obs)


@lru_cache(maxsize=4)
def _cached_table(key: str):
    return _make_table(get_observer(key))


def _table(obs):
    from .observers import CIE1931_2, CIE1964_10

    if obs is CIE1931_2:
        return _cached_table("1931")
    if obs is CIE1964_10:
        return _cached_table("1964")
    return _make_table(obs)


def _resolve_uv(xyz=None, xy=None, uv=None) -> np.ndarray:
    if uv is not None:
        return as_float_array(uv)
    if xy is not None:
        return xy_to_uv1960(xy)
    if xyz is not None:
        return xyz_to_uv1960(xyz)
    raise ValueError("give one of xyz=, xy= or uv=")


def cct(xyz=None, *, xy=None, uv=None, observer="1931", iterations: int = 2):
    """Correlated colour temperature [K] and Duv of colours given as XYZ, xy or CIE 1960 uv.

    Returns ``(T, duv)`` arrays. Duv is the signed distance from the Planckian locus in the
    CIE 1960 uv diagram (positive = above the locus, i.e. greenish). Colours outside
    1000-100000 K are clamped to the end of the range.

    Method: nearest node of a dense locus table (coarse-to-fine search), exact projection onto the
    adjacent polyline segments, then Newton steps on a cubic interpolation of the integrated
    Planckian locus (interpolation error ~1e-11 in uv: CCT to better than 1e-6 relative for lighting-like
    colours, ~1e-5 in the extreme corner of very hot colours far from the locus).
    """
    obs = get_observer(observer)
    uv_t = _resolve_uv(xyz, xy, uv)
    shape = uv_t.shape[:-1]
    pts = uv_t.reshape(-1, 2)
    mired_tab, uv_tab = _table(obs)
    lo, hi = 1e6 / T_MAX, 1e6 / T_MIN
    n_tab = len(mired_tab)
    # 1) nearest node: coarse table (every 10th node), then the fine nodes around it
    step = 10
    uvc = uv_tab[::step]
    idx = np.empty(len(pts), dtype=int)
    for a in range(0, len(pts), 8192):
        d2 = ((pts[a:a + 8192, None, :] - uvc[None]) ** 2).sum(-1)
        idx[a:a + 8192] = np.argmin(d2, axis=1) * step
    cand = np.clip(idx[:, None] + np.arange(-2 * step, 2 * step + 1), 0, n_tab - 1)
    d2 = ((uv_tab[cand] - pts[:, None, :]) ** 2).sum(-1)
    i = cand[np.arange(len(pts)), np.argmin(d2, axis=1)]
    # 2) project onto the two polyline segments next to that node; keep the closer foot point
    best_m = np.empty(len(pts))
    best_d = np.full(len(pts), np.inf)
    for a_idx, b_idx in ((np.maximum(i - 1, 0), i), (i, np.minimum(i + 1, n_tab - 1))):
        A, B = uv_tab[a_idx], uv_tab[b_idx]
        AB = B - A
        L2 = np.maximum((AB**2).sum(-1), 1e-300)
        t = np.clip(((pts - A) * AB).sum(-1) / L2, 0.0, 1.0)
        dist = ((A + t[:, None] * AB - pts) ** 2).sum(-1)
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_m = np.where(better, mired_tab[a_idx] + t * (mired_tab[b_idx] - mired_tab[a_idx]), best_m)
    m = best_m
    h_tab = mired_tab[1] - mired_tab[0]

    def locus(mm):  # Catmull-Rom interpolation of the dense table: |error| ~ 1e-11 in uv
        x = (mm - mired_tab[0]) / h_tab
        k = np.clip(np.floor(x).astype(int), 1, n_tab - 3)
        s = (x - k)[:, None]
        p0, p1, p2, p3 = uv_tab[k - 1], uv_tab[k], uv_tab[k + 1], uv_tab[k + 2]
        return 0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s**2
                      + (3 * p1 - p0 - 3 * p2 + p3) * s**3)

    def g_exact(mm):
        return ((locus(mm) - pts) ** 2).sum(-1)

    def newton(mm, step, g, max_move):
        g0, gp, gm = g(mm), g(mm + step), g(mm - step)
        curv = gp - 2.0 * g0 + gm
        with np.errstate(divide="ignore", invalid="ignore"):
            dm = np.where(curv > 0, -step * (gp - gm) / (2.0 * curv), 0.0)
        return np.clip(mm + np.clip(dm, -max_move, max_move), lo, hi)

    # 3) Newton steps on d/dm |uv(m) - uv|^2 = 0 on the smooth (cubic) locus remove the polyline error
    for _ in range(iterations):
        m = newton(m, 2e-3, g_exact, 0.25)
    T = 1e6 / m
    foot = locus(m)
    duv = np.sign(pts[:, 1] - foot[:, 1]) * np.linalg.norm(pts - foot, axis=-1)
    return T.reshape(shape), duv.reshape(shape)


def _unit_normal(T, observer):
    """Unit normal of the Planckian locus in uv (pointing to larger v)."""
    T = as_float_array(T)
    mired = 1e6 / T
    d = 1e-3 * np.maximum(mired, 1.0)
    t = planckian_uv(1e6 / (mired - d), observer) - planckian_uv(1e6 / (mired + d), observer)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    n = np.stack([-t[..., 1], t[..., 0]], axis=-1)
    return np.where(n[..., 1:2] < 0, -n, n)


def uv_from_cct(T, duv=0.0, observer="1931") -> np.ndarray:
    """CIE 1960 uv for a CCT [K] and Duv (inverse of :func:`cct`)."""
    T = as_float_array(T)
    duv = as_float_array(duv)
    return planckian_uv(T, observer) + duv[..., None] * _unit_normal(T, observer)


def xy_from_cct(T, duv=0.0, observer="1931") -> np.ndarray:
    """CIE xy for a CCT [K] and Duv."""
    return uv1960_to_xy(uv_from_cct(T, duv, observer))


def isotherms(T, duv: float = 0.02, observer="1931") -> np.ndarray:
    """End points (shape ``T.shape + (2, 2)``, xy) of isotemperature lines spanning ``±duv``."""
    T = as_float_array(T)
    a = xy_from_cct(T, -duv, observer)
    b = xy_from_cct(T, duv, observer)
    return np.stack([a, b], axis=-2)


def daylight_locus(T=None) -> np.ndarray:
    """xy of the CIE daylight locus (4000-25000 K)."""
    T = np.geomspace(4000.0, 25000.0, 200) if T is None else as_float_array(T)
    return daylight_xy(T)


def cct_mccamy(xy) -> np.ndarray:
    """McCamy's (1992) cubic approximation of CCT from xy (quick, ~±2 K near the locus, 2856-6504 K)."""
    x, y = np.moveaxis(as_float_array(xy), -1, 0)
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
