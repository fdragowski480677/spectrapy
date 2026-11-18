"""Spectral lines: line profiles, built-in emission line lists and Fraunhofer absorption lines.

Wavelengths are standard air wavelengths (nm) as listed in the NIST Atomic Spectra Database.
**Relative intensities and the Fraunhofer line strengths are approximate** - they depend on the
source conditions and are meant for illustration (spectroscope views, teaching), not for
quantitative spectroscopy. Use :class:`spectrapy.LineSpectrum` with your own data for that.
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array, erf
from .spectrum import LineSpectrum, Spectrum

_SQRT2LN2 = np.sqrt(2.0 * np.log(2.0))


def line_profile(wl, center, fwhm: float, profile: str = "gaussian"):
    """Unit-area line profile (per nm) evaluated at ``wl``; broadcasting over ``wl`` and ``center``."""
    x = as_float_array(wl) - as_float_array(center)
    if profile == "gaussian":
        sigma = fwhm / (2.0 * _SQRT2LN2)
        return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    if profile == "lorentzian":
        g = 0.5 * fwhm
        return (g / np.pi) / (x**2 + g**2)
    raise ValueError("profile must be 'gaussian' or 'lorentzian'")


def line_profile_cdf(wl, center, fwhm: float, profile: str = "gaussian"):
    """Cumulative distribution of :func:`line_profile` (used for exact pixel-binned rendering)."""
    x = as_float_array(wl) - as_float_array(center)
    if profile == "gaussian":
        sigma = fwhm / (2.0 * _SQRT2LN2)
        return 0.5 * (1.0 + erf(x / (sigma * np.sqrt(2.0))))
    if profile == "lorentzian":
        return 0.5 + np.arctan(x / (0.5 * fwhm)) / np.pi
    raise ValueError("profile must be 'gaussian' or 'lorentzian'")


def binned_lines(lines: LineSpectrum, edges, fwhm: float, profile: str = "gaussian") -> np.ndarray:
    """Power of each line falling into wavelength bins ``edges`` (exact integration of the profile),
    summed over lines. Returns an array of ``len(edges) - 1`` values (power per bin)."""
    edges = as_float_array(edges)
    cdf = line_profile_cdf(edges[None, :], lines.wavelengths[:, None], fwhm, profile)
    return lines.intensities @ np.diff(cdf, axis=1)


# (air wavelength nm, approximate relative intensity)
_LINES = {
    "H": [(388.905, 0.04), (397.007, 0.06), (410.174, 0.10), (434.047, 0.18), (486.135, 0.36),
          (656.279, 1.00)],
    "He": [(388.865, 0.50), (396.473, 0.02), (402.619, 0.05), (438.793, 0.01), (447.148, 0.20),
           (471.315, 0.03), (492.193, 0.05), (501.568, 0.10), (504.774, 0.01), (587.562, 1.00),
           (667.815, 0.10), (706.519, 0.20), (728.135, 0.05)],
    "Ne": [(533.078, 0.03), (534.109, 0.04), (540.056, 0.06), (585.249, 0.40), (588.190, 0.20),
           (594.483, 0.50), (597.553, 0.10), (603.000, 0.20), (607.434, 0.20), (609.616, 0.30),
           (614.306, 0.60), (616.359, 0.20), (621.728, 0.20), (626.650, 0.60), (630.479, 0.20),
           (633.443, 0.60), (638.299, 0.60), (640.225, 1.00), (650.653, 0.80), (653.288, 0.30),
           (659.895, 0.60), (667.828, 0.50), (671.704, 0.20), (692.947, 0.60), (703.241, 0.50),
           (717.394, 0.20), (724.517, 0.60), (743.890, 0.30)],
    "Na": [(497.854, 0.003), (498.281, 0.006), (568.263, 0.01), (568.820, 0.02), (588.995, 1.00),
           (589.592, 0.50), (615.423, 0.008), (616.075, 0.015)],
    "Hg": [(404.656, 0.45), (407.783, 0.05), (435.833, 1.00), (491.607, 0.02), (546.074, 0.90),
           (576.960, 0.20), (579.066, 0.22), (623.437, 0.01), (690.752, 0.01)],
}
_NAMES = {"H": "Hydrogen", "He": "Helium", "Ne": "Neon", "Na": "Sodium", "Hg": "Mercury"}


def available_elements() -> list[str]:
    return list(_LINES)


def line_spectrum(element: str) -> LineSpectrum:
    """Approximate visible emission lines of a discharge lamp: ``"H"``, ``"He"``, ``"Ne"``, ``"Na"``
    or ``"Hg"`` (strongest line = 1)."""
    key = {v.lower(): k for k, v in _NAMES.items()}.get(element.strip().lower(), element.strip().capitalize())
    if key not in _LINES:
        raise ValueError(f"unknown element {element!r}; available: {', '.join(_LINES)}")
    wl, it = np.array(_LINES[key]).T
    return LineSpectrum(wl, it, name=_NAMES[key])


# Fraunhofer lines: (label, air wavelength nm, species, depth 0..1, Lorentzian FWHM nm)
FRAUNHOFER = [
    ("A", 759.370, "O₂", 0.90, 0.90),
    ("B", 686.719, "O₂", 0.75, 0.60),
    ("C", 656.281, "Hα", 0.85, 0.30),
    ("a", 627.661, "O₂", 0.40, 0.35),
    ("D₁", 589.592, "Na", 0.90, 0.04),
    ("D₂", 588.995, "Na", 0.93, 0.05),
    ("E₂", 527.039, "Fe", 0.70, 0.03),
    ("b₁", 518.362, "Mg", 0.85, 0.12),
    ("b₂", 517.270, "Mg", 0.85, 0.10),
    ("b₄", 516.733, "Mg", 0.70, 0.07),
    ("c", 495.761, "Fe", 0.55, 0.05),
    ("F", 486.134, "Hβ", 0.85, 0.28),
    ("d", 466.814, "Fe", 0.50, 0.04),
    ("e", 438.355, "Fe", 0.75, 0.08),
    ("G′", 434.047, "Hγ", 0.80, 0.22),
    ("G", 430.790, "Fe/CH", 0.75, 0.25),
    ("h", 410.175, "Hδ", 0.75, 0.24),
    ("H", 396.847, "Ca⁺", 0.95, 1.00),
    ("K", 393.366, "Ca⁺", 0.95, 1.30),
]


#: Short labels for annotating the classic Fraunhofer lines in plots (close lines grouped).
FRAUNHOFER_LABELS = [("K", 393.366), ("H", 396.847), ("h", 410.175), ("G", 430.790), ("F", 486.134),
                     ("b", 517.6), ("E", 527.039), ("D", 589.29), ("a", 627.661), ("C", 656.281),
                     ("B", 686.719), ("A", 759.370)]


def absorption(continuum: Spectrum, lines=None, wavelengths=None, step: float = 0.01,
               width_scale: float = 1.0) -> Spectrum:
    """Imprint absorption lines on a continuum: ``S(λ) = C(λ) Π (1 - dᵢ Lᵢ(λ))`` with Lorentzian
    profiles of unit peak. ``lines`` is a sequence of ``(label, wavelength, species, depth, fwhm)``
    (default :data:`FRAUNHOFER`); the result is sampled every ``step`` nm. ``width_scale`` widens all
    lines (e.g. to keep them visible in small figures)."""
    lines = FRAUNHOFER if lines is None else lines
    if wavelengths is None:
        lo, hi = continuum.wavelengths[0], continuum.wavelengths[-1]
        wavelengths = np.arange(lo, hi + step / 2, step)
    wl = as_float_array(wavelengths)
    trans = np.ones_like(wl)
    for _label, c, _species, depth, fwhm in lines:
        g = 0.5 * fwhm * width_scale
        trans *= 1.0 - depth * g**2 / ((wl - c) ** 2 + g**2)
    base = continuum.resample(wl)
    return base._with(base.values * trans, name=f"{continuum.name or 'continuum'} + absorption")


def solar_spectrum(T: float = 5772.0, lo: float = 380.0, hi: float = 780.0, step: float = 0.01,
                   width_scale: float = 1.0) -> Spectrum:
    """Illustrative solar spectrum: a blackbody photosphere with the classic Fraunhofer lines."""
    from .illuminants import blackbody

    wl = np.arange(lo, hi + step / 2, step)
    s = absorption(blackbody(T, wavelengths=wl), wavelengths=wl, width_scale=width_scale)
    s.name = "Sun (blackbody + Fraunhofer lines)"
    return s
