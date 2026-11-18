"""Geometric optics of dispersion: prisms, diffraction gratings and rainbows (Descartes' theory).

The rainbow model is geometric optics only (no Airy supernumerary bows): for every wavelength the
ray density near the minimum-deviation angle θ_k of order ``k`` (number of internal reflections)
diverges as ``|θ_k - θ|^(-1/2)`` on the bright side, which is then smoothed by the finite size of
the solar disk (radius 0.266°). The secondary bow is weaker (factor ``secondary_gain``) and has
reversed colours; Alexander's dark band between the bows appears automatically.
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array
from .spectrum import Spectrum
from .spectrum import wavelengths as default_wavelengths
from .thinfilm import BK7, WATER, Material, as_material, sellmeier

SUN_RADIUS_DEG = 0.266

#: Schott N-SF11 dense flint glass (strong dispersion, good for showing a prism's colour fan).
SF11 = sellmeier([1.73759695, 0.313747346, 1.89878101], [0.013188707, 0.0623068142, 155.23629], "N-SF11")


# ------------------------------------------------------------------------- prisms
def _refract(d, normal, eta):
    """Vector Snell's law. ``d``: unit directions (..., 2); ``normal``: unit normal pointing against
    ``d``; ``eta = n1 / n2``. Returns refracted unit directions (NaN for total internal reflection)."""
    cos_i = -np.sum(d * normal, axis=-1, keepdims=True)
    sin2_t = eta**2 * (1.0 - cos_i**2)
    with np.errstate(invalid="ignore"):
        t = eta * d + (eta * cos_i - np.sqrt(1.0 - sin2_t)) * normal
    return np.where(sin2_t <= 1.0, t, np.nan)


def _hit(p, d, a, b):
    """Parameter ``s`` along rays ``p + s d`` where they cross the line through ``a`` and ``b``."""
    e = b - a
    den = d[..., 0] * e[1] - d[..., 1] * e[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((a[0] - p[..., 0]) * e[1] - (a[1] - p[..., 1]) * e[0]) / den


def prism_rays(wavelengths, material: Material = SF11, apex_deg: float = 60.0, incidence_deg: float | None = None,
               ref_wavelength: float = 550.0, side: float = 1.0, screen_x: float = 2.2, start_x: float = -1.6):
    """Trace rays of several wavelengths through a prism (exact vector Snell's law, in 2D).

    The prism has its apex at the top and its base on ``y = 0``; light enters through the left face.
    By default the incidence angle is chosen for minimum deviation at ``ref_wavelength``.

    Returns ``(prism_xy, paths)``: the triangle's three vertices, and ``paths`` of shape
    ``(n_wavelengths, 4, 2)`` - source point, entry point, exit point, and the point where each ray
    meets a vertical screen at ``x = screen_x``.
    """
    wl = as_float_array(wavelengths).ravel()
    n = as_material(material)(wl).real
    a = np.radians(apex_deg)
    height = side * np.cos(a / 2.0)
    half = side * np.sin(a / 2.0)
    A, B, C = np.array([0.0, height]), np.array([-half, 0.0]), np.array([half, 0.0])
    if incidence_deg is None:
        n_ref = as_material(material)(np.array([ref_wavelength])).real[0]
        incidence_deg = np.degrees(np.arcsin(np.clip(n_ref * np.sin(a / 2.0), -1, 1)))
    # outward unit normals of the left (BA) and right (AC) faces
    nl = np.array([-np.cos(a / 2.0), np.sin(a / 2.0)])
    nr = np.array([np.cos(a / 2.0), np.sin(a / 2.0)])
    i = np.radians(incidence_deg)
    # incoming direction: the inward normal (-nl) rotated counter-clockwise by the incidence angle,
    # i.e. the ray travels up and to the right (at minimum deviation it runs parallel to the base inside)
    rot = np.array([[np.cos(i), -np.sin(i)], [np.sin(i), np.cos(i)]])
    d0 = rot @ (-nl)
    entry = B + 0.55 * (A - B)  # common entry point on the left face
    src = entry - d0 * ((entry[0] - start_x) / d0[0])
    d0 = np.broadcast_to(d0, (wl.size, 2))
    d1 = _refract(d0, nl, 1.0 / n[:, None])
    p1 = np.broadcast_to(entry, (wl.size, 2))
    s1 = _hit(p1, d1, A, C)
    exit_ = p1 + s1[:, None] * d1
    d2 = _refract(d1, -nr, n[:, None])
    s2 = (screen_x - exit_[:, 0]) / d2[:, 0]
    screen = exit_ + s2[:, None] * d2
    paths = np.stack([np.broadcast_to(src, (wl.size, 2)), p1, exit_, screen], axis=1)
    return np.array([A, B, C]), paths


# ------------------------------------------------------------------------ gratings
def grating_angle(wl_nm, lines_per_mm: float = 600.0, order: int = 1, incidence_deg: float = 0.0) -> np.ndarray:
    """Diffraction angle (degrees) from the grating equation ``d (sin θ_m - sin θ_i) = m λ``;
    NaN where the order does not exist."""
    d = 1e6 / lines_per_mm  # groove spacing in nm
    s = order * as_float_array(wl_nm) / d + np.sin(np.radians(incidence_deg))
    with np.errstate(invalid="ignore"):
        return np.degrees(np.where(np.abs(s) <= 1.0, np.arcsin(s), np.nan))


def grating_screen_xyz(x, lines_per_mm: float = 600.0, orders=(-3, -2, -1, 1, 2, 3), source=None,
                       efficiency=None, observer="1931", line_width: float = 0.004) -> np.ndarray:
    """XYZ of the light from a diffraction grating landing on a flat screen at distance 1 (normal
    incidence), at screen positions ``x``: shape ``x.shape + (3,)``.

    Each order spreads the source spectrum (a :class:`Spectrum`, default D65) over the screen; the
    power per unit screen length includes the Jacobian ``|dλ/dx|``, and overlapping orders simply
    add up. For a :class:`~spectrapy.LineSpectrum` every line appears in every order as a narrow
    Gaussian of width ``line_width`` (screen units). ``efficiency`` maps the order number to a
    relative efficiency (default ``1/|m|``).
    """
    from .illuminants import illuminant
    from .observers import get_observer
    from .spectrum import LineSpectrum

    x = as_float_array(x)
    obs = get_observer(observer)
    src = illuminant("D65") if source is None else source
    d = 1e6 / lines_per_mm
    out = np.zeros(x.shape + (3,))
    if isinstance(src, LineSpectrum):
        cmf = obs.at(src.wavelengths)
        for m in orders:
            if m == 0:
                continue
            s_m = m * src.wavelengths / d
            ok = np.abs(s_m) < 1.0
            xm = s_m[ok] / np.sqrt(1.0 - s_m[ok] ** 2)  # tan(asin(.))
            eff = (1.0 / abs(m)) if efficiency is None else efficiency(m)
            prof = np.exp(-0.5 * ((x[..., None] - xm) / line_width) ** 2)
            out += eff * (prof * src.intensities[ok]) @ cmf[ok]
        return out
    sin_t = x / np.sqrt(1.0 + x**2)
    jac_base = (1.0 + x**2) ** -1.5  # d(sin θ)/dx
    for m in orders:
        if m == 0:
            continue
        lam = d * sin_t / m
        valid = (lam >= obs.wavelengths[0]) & (lam <= obs.wavelengths[-1])
        lam_c = np.where(valid, lam, obs.wavelengths[0])
        eff = (1.0 / abs(m)) if efficiency is None else efficiency(m)
        w = eff * src.at(lam_c) * (d / abs(m)) * jac_base * valid
        out += w[..., None] * obs.at(lam_c)
    return out


def rainbow_angle(n, order: int = 1) -> np.ndarray:
    """Rainbow angle (degrees from the antisolar point) for refractive index ``n`` and ``order``."""
    n = as_float_array(n)
    k = order
    cos_i = np.sqrt((n**2 - 1.0) / (k * k + 2.0 * k))
    i = np.arccos(cos_i)
    r = np.arcsin(np.sin(i) / n)
    D = np.degrees(2.0 * (i - r) + k * (np.pi - 2.0 * r))
    return np.abs(180.0 - np.mod(D, 360.0))


def prism_deviation(n, apex_deg: float = 60.0, incidence_deg=None) -> np.ndarray:
    """Deviation angle (degrees) of a ray through a prism; minimum deviation if ``incidence_deg`` is None."""
    n = as_float_array(n)
    A = np.radians(apex_deg)
    if incidence_deg is None:
        return np.degrees(2.0 * np.arcsin(n * np.sin(A / 2.0)) - A)
    i1 = np.radians(incidence_deg)
    r1 = np.arcsin(np.sin(i1) / n)
    r2 = A - r1
    i2 = np.arcsin(np.clip(n * np.sin(r2), -1.0, 1.0))
    return np.degrees(i1 + i2 - A)


def _disk_kernel(step_deg: float, radius_deg: float = SUN_RADIUS_DEG):
    r = int(np.ceil(radius_deg / step_deg))
    x = np.arange(-r, r + 1) * step_deg
    k = np.sqrt(np.clip(1.0 - (x / radius_deg) ** 2, 0.0, None))  # 1-D projection of a uniform disk
    return k / k.sum()


def rainbow_spectra(theta_deg, material=WATER, wavelengths=None, orders=(1, 2), secondary_gain: float = 0.43,
                    sun: Spectrum | None = None, sun_radius_deg: float = SUN_RADIUS_DEG) -> Spectrum:
    """Spectral radiance (relative) of rainbow light at angles ``theta_deg`` from the antisolar point.

    ``theta_deg`` must be a uniformly spaced 1-D array. Returns a batch :class:`Spectrum` with one
    spectrum per angle (sunlight defaults to a 5772 K blackbody).
    """
    from .illuminants import blackbody

    th = as_float_array(theta_deg).ravel()
    step = float(th[1] - th[0])
    wl = default_wavelengths(380, 780, 2) if wavelengths is None else as_float_array(wavelengths).ravel()
    n = as_material(material)(wl).real
    S = (blackbody(5772.0, wavelengths=wl) if sun is None else sun.resample(wl)).values
    out = np.zeros((th.size, wl.size))
    kern = _disk_kernel(step, sun_radius_deg)
    for k in orders:
        tk = rainbow_angle(n, k)  # (n_wl,)
        d = (tk[None, :] - th[:, None]) if k == 1 else (th[:, None] - tk[None, :])  # bright side > 0
        dens = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, step / 4)), 0.0)
        gain = 1.0 if k == 1 else secondary_gain
        out += gain * dens
    # smooth by the solar disk, column by column (wavelength)
    pad = len(kern) // 2
    padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, j], kern, mode="valid") for j in range(wl.size)], axis=1)
    return Spectrum(wl, out * S[None, :], "emission", name="rainbow")


__all__ = ["BK7", "SF11", "WATER", "grating_angle", "grating_screen_xyz", "prism_deviation", "prism_rays",
           "rainbow_angle", "rainbow_spectra"]
