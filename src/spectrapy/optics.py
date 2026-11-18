"""Geometric optics of dispersion: prisms and rainbows (Descartes' theory).

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
from .thinfilm import BK7, WATER, as_material

SUN_RADIUS_DEG = 0.266


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


__all__ = ["BK7", "WATER", "prism_deviation", "rainbow_angle", "rainbow_spectra"]
