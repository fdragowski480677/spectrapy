"""Small numerical helpers shared across modules (vectorized, NumPy only)."""
from __future__ import annotations

from functools import cache
from importlib import resources

import numpy as np


def as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def interp(x_new, x, y, left=0.0, right=0.0):
    """Linear interpolation of ``y`` (shape ``(..., n)``) sampled at ascending ``x`` (shape ``(n,)``).

    Works on whole batches at once (the interpolation weights are computed a single time).
    ``left``/``right`` give the value outside ``[x[0], x[-1]]``: a number, or ``"hold"`` to keep
    the end values.
    """
    x = as_float_array(x)
    y = as_float_array(y)
    x_new = as_float_array(x_new)
    if x.size == 1:
        out = np.broadcast_to(y[..., :1], y.shape[:-1] + x_new.shape).copy()
    else:
        idx = np.clip(np.searchsorted(x, x_new, side="right") - 1, 0, x.size - 2)
        x0 = x[idx]
        t = (x_new - x0) / (x[idx + 1] - x0)
        y0 = y[..., idx]
        out = y0 + (y[..., idx + 1] - y0) * t
    lo = x_new < x[0]
    hi = x_new > x[-1]
    if np.any(lo):
        out[..., lo] = y[..., :1] if isinstance(left, str) else left
    if np.any(hi):
        out[..., hi] = y[..., -1:] if isinstance(right, str) else right
    return out


def trapezoid_weights(x) -> np.ndarray:
    """Weights ``w`` such that ``sum(w * f) == trapezoid(f, x)`` for samples ``f`` on grid ``x``."""
    x = as_float_array(x)
    w = np.zeros_like(x)
    if x.size < 2:
        return w
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def trapezoid(y, x, axis=-1):
    """Trapezoidal integral of ``y`` over ``x`` along ``axis`` (NumPy-version independent)."""
    y = np.moveaxis(as_float_array(y), axis, -1)
    return y @ trapezoid_weights(x)


def union_grid(*grids, lo=None, hi=None) -> np.ndarray:
    """Sorted union of wavelength grids restricted to ``[lo, hi]`` (end points included)."""
    g = np.unique(np.concatenate([as_float_array(a).ravel() for a in grids]))
    if lo is not None:
        g = g[g >= lo]
    if hi is not None:
        g = g[g <= hi]
    extra = [v for v in (lo, hi) if v is not None]
    if extra:
        g = np.unique(np.concatenate([g, extra]))
    return g


def erf(x):
    """Error function, vectorized (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)."""
    x = as_float_array(x)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return s * (1.0 - poly * np.exp(-a * a))


def sliding_min(a, radius: int, axis: int = -1):
    """Running minimum over a window of ``2 * radius + 1`` samples (edges padded with end values)."""
    if radius <= 0:
        return np.array(a, dtype=float)
    a = np.moveaxis(as_float_array(a), axis, -1)
    pad = np.concatenate([np.repeat(a[..., :1], radius, -1), a, np.repeat(a[..., -1:], radius, -1)], -1)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1, axis=-1)
    return np.moveaxis(win.min(-1), -1, axis)


def lower_envelope_smooth(a, radius: int, axis: int = -1):
    """Smooth ``a`` while never exceeding it: running minimum followed by a Gaussian blur of the
    same support. Used to remove kinks (Mach bands) from gamut-mapping factors."""
    if radius <= 0:
        return np.array(a, dtype=float)
    m = np.moveaxis(sliding_min(a, radius, axis), axis, -1)
    k = np.arange(-radius, radius + 1)
    g = np.exp(-0.5 * (k / max(radius / 2.0, 1e-9)) ** 2)
    g /= g.sum()
    pad = np.concatenate([np.repeat(m[..., :1], radius, -1), m, np.repeat(m[..., -1:], radius, -1)], -1)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1, axis=-1)
    return np.moveaxis(win @ g, -1, axis)


@cache
def load_table(name: str):
    """Load a bundled CSV table. Returns ``(wavelengths, {column: values})`` as read-only arrays."""
    res = resources.files("spectrapy").joinpath("data", name)
    if name.endswith(".gz"):
        import gzip

        text = gzip.decompress(res.read_bytes()).decode("utf-8")
    else:
        text = res.read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    header = [h.strip() for h in rows[0].split(",")]
    arr = np.array([[float(v) for v in ln.split(",")] for ln in rows[1:]])
    arr.setflags(write=False)
    cols = {h: arr[:, i] for i, h in enumerate(header[1:], start=1)}
    return arr[:, 0], cols
