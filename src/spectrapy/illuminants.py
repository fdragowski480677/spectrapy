"""Light sources: Planckian radiators, CIE daylight (D series) and the CIE standard illuminants."""
from __future__ import annotations

import re

import numpy as np

from ._util import as_float_array, interp, load_table
from .constants import C1L, C2, C2_CIE
from .spectrum import Spectrum
from .spectrum import wavelengths as default_wavelengths

#: CIE 1931 2° chromaticities (x, y) of common white points.
WHITE_POINTS = {
    "A": (0.44757, 0.40745),
    "C": (0.31006, 0.31616),
    "D50": (0.3457, 0.3585),
    "D55": (0.33242, 0.34743),
    "D65": (0.3127, 0.3290),
    "D75": (0.29902, 0.31485),
    "E": (1.0 / 3.0, 1.0 / 3.0),
}


def planck(wl_nm, T, c2: float = C2):
    """Planck's law: spectral radiance [W sr^-1 m^-2 nm^-1] at wavelength ``wl_nm`` and temperature
    ``T`` [K]. Fully broadcasting; very short wavelengths underflow cleanly to 0."""
    wl = as_float_array(wl_nm) * 1e-9
    T = as_float_array(T)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        L = C1L / (wl**5 * np.expm1(c2 / (wl * T)))
    return np.where(np.isfinite(L), L, 0.0) * 1e-9


def blackbody(T, wavelengths=None, normalize: str | None = None, c2: float = C2) -> Spectrum:
    """Spectrum of a Planckian radiator (spectral radiance per nm).

    ``T`` may be an array: the result is then a batch of spectra with ``shape == T.shape``.
    ``normalize``: ``None`` (absolute radiance), ``"max"``, ``"560"`` (value at 560 nm = 100, the
    CIE convention for illuminants) or ``"luminance"``.
    """
    wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths).ravel()
    T = as_float_array(T)
    vals = planck(wl, T[..., None], c2=c2)
    s = Spectrum(wl, vals, "emission", name=f"Blackbody {T:g} K" if T.ndim == 0 else "Blackbody")
    if normalize == "560":
        return s._with(100.0 * vals / planck(560.0, T[..., None], c2=c2))
    if normalize is not None:
        return s.normalize(normalize)
    return s


def daylight_xy(cct):
    """Chromaticity (x, y) of CIE daylight at correlated colour temperature ``cct`` (4000-25000 K)."""
    T = as_float_array(cct)
    x_lo = -4.6070e9 / T**3 + 2.9678e6 / T**2 + 0.09911e3 / T + 0.244063
    x_hi = -2.0064e9 / T**3 + 1.9018e6 / T**2 + 0.24748e3 / T + 0.237040
    x = np.where(T <= 7000.0, x_lo, x_hi)
    y = -3.000 * x**2 + 2.870 * x - 0.275
    return np.stack([x, y], axis=-1)


def daylight_from_xy(xy, wavelengths=None, round_m: bool = True) -> Spectrum:
    """CIE daylight spectrum reconstructed from a chromaticity on the daylight locus."""
    xy = as_float_array(xy)
    x, y = xy[..., 0], xy[..., 1]
    M = 0.0241 + 0.2562 * x - 0.7341 * y
    M1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / M
    M2 = (0.0300 - 31.4424 * x + 30.0717 * y) / M
    if round_m:  # CIE 015: round M1, M2 to three decimals (reproduces the published tables)
        M1, M2 = np.round(M1, 3), np.round(M2, 3)
    wl_b, cols = load_table("daylight_basis.csv")
    vals = cols["S0"] + M1[..., None] * cols["S1"] + M2[..., None] * cols["S2"]
    s = Spectrum(wl_b, vals, "emission")
    return s if wavelengths is None else s.resample(wavelengths)


def daylight(cct, wavelengths=None, round_m: bool = True) -> Spectrum:
    """CIE daylight (D-series) illuminant for a correlated colour temperature [K] (array allowed).

    The spectrum comes from the S0/S1/S2 basis functions (5 nm) and is linearly interpolated to
    ``wavelengths`` if given. Values are relative, ~100 at 560 nm.
    """
    s = daylight_from_xy(daylight_xy(cct), wavelengths=wavelengths, round_m=round_m)
    T = as_float_array(cct)
    s.name = f"Daylight {T:g} K" if T.ndim == 0 else "Daylight"
    return s


def _nominal_d(cct_nominal: float) -> float:
    # D50, D55, D75 are defined at the nominal CCT scaled by the change of c2 (1.4380 -> 1.4388)
    return cct_nominal * 1.4388 / 1.4380


def _table(fname: str, column: str, name: str) -> Spectrum:
    wl, cols = load_table(fname)
    return Spectrum(wl, np.array(cols[column]), "emission", name=name)


def illuminant_a_formula(wavelengths=None) -> Spectrum:
    """CIE standard illuminant A from its defining formula (Planckian radiator, 2856 K)."""
    wl = default_wavelengths(300, 830, 1) if wavelengths is None else as_float_array(wavelengths)
    c2 = 1.435e7  # nm K, as in the CIE definition of illuminant A
    vals = 100.0 * (560.0 / wl) ** 5 * np.expm1(c2 / (2848.0 * 560.0)) / np.expm1(c2 / (2848.0 * wl))
    return Spectrum(wl, vals, "emission", name="A")


_FL = [f"FL{i}" for i in range(1, 13)]
_LED = ["LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5", "LED-BH1", "LED-RGB1", "LED-V1", "LED-V2"]


def available_illuminants() -> list[str]:
    """Names accepted by :func:`illuminant`."""
    return ["A", "D50", "D55", "D65", "D75", "E", *_FL, *_LED]


def illuminant(name: str, wavelengths=None) -> Spectrum:
    """A CIE illuminant by name: ``A``, ``D50``, ``D55``, ``D65``, ``D75``, ``E``, ``FL1``-``FL12``,
    ``LED-B1``... (see :func:`available_illuminants`). Optionally resampled to ``wavelengths``."""
    key = name.strip().upper()
    if re.fullmatch(r"F\d{1,2}", key):  # accept "F2" as well as "FL2"
        key = "FL" + key[1:]
    if key == "A":
        s = _table("illuminants_cie_1nm.csv", "A", "A")
    elif key == "D65":
        s = _table("illuminants_cie_1nm.csv", "D65", "D65")
    elif key in ("D50", "D55", "D75"):
        s = daylight(_nominal_d(float(key[1:]) * 100.0))
        s.name = key
    elif key == "E":
        wl = default_wavelengths(300, 830, 1)
        s = Spectrum(wl, np.full(wl.shape, 100.0), "emission", name="E")
    elif key in _FL:
        s = _table("illuminants_fl.csv", key, key)
    elif key in [n.upper() for n in _LED]:
        col = _LED[[n.upper() for n in _LED].index(key)]
        s = _table("illuminants_led.csv", col, col)
    else:
        raise ValueError(f"unknown illuminant {name!r}; available: {', '.join(available_illuminants())}")
    return s if wavelengths is None else Spectrum(as_float_array(wavelengths),
                                                  interp(wavelengths, s.wavelengths, s.values),
                                                  "emission", s.name)


def resolve_illuminant(spec) -> Spectrum:
    """Accept a :class:`Spectrum`, an illuminant name, or a number (blackbody temperature in K)."""
    if isinstance(spec, Spectrum):
        return spec
    if isinstance(spec, str):
        return illuminant(spec)
    if np.ndim(spec) == 0:
        return blackbody(float(spec), normalize="560", c2=C2_CIE)
    raise TypeError(f"cannot interpret {spec!r} as an illuminant")
