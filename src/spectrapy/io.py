"""Reading and writing spectra (CSV / whitespace-delimited text, e.g. spectrometer exports)."""
from __future__ import annotations

import pathlib

import numpy as np

from .constants import HC_EV_NM, C
from .spectrum import Spectrum

_UNITS = {
    "nm": ("length", 1.0), "um": ("length", 1e3), "µm": ("length", 1e3), "micron": ("length", 1e3),
    "a": ("length", 0.1), "angstrom": ("length", 0.1), "å": ("length", 0.1), "m": ("length", 1e9),
    "ev": ("energy", HC_EV_NM), "cm-1": ("energy", 1e7), "1/cm": ("energy", 1e7), "wavenumber": ("energy", 1e7),
    "thz": ("energy", C * 1e-3), "hz": ("energy", C * 1e9),
}


def _unit(unit: str):
    try:
        return _UNITS[unit.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; use one of {', '.join(_UNITS)}") from None


def to_nanometres(x, unit: str = "nm") -> np.ndarray:
    """Convert spectral coordinates (wavelength, photon energy, wavenumber or frequency) to nm."""
    kind, f = _unit(unit)
    x = np.asarray(x, dtype=float)
    return x * f if kind == "length" else f / x


def parse_spectrum(text: str, unit: str = "nm", kind: str = "emission", columns=(0, 1), delimiter=None,
                   density: bool = True, name: str | None = None, scale: float = 1.0) -> Spectrum:
    """Parse a two-column spectrum from text; see :func:`read_spectrum` for the parameters."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#%;":
            continue
        if delimiter is None:
            parts = s.replace(";", ",").split(",") if ("," in s or ";" in s) else s.split()
        else:
            parts = s.split(delimiter)
        try:
            rows.append([float(parts[c]) for c in columns])
        except (ValueError, IndexError):
            continue  # header or malformed line
    if not rows:
        raise ValueError("no numeric data found")
    data = np.array(rows)
    wl = to_nanometres(data[:, 0], unit)
    y = data[:, 1] * scale
    ukind, f = _unit(unit)
    if density and kind == "emission" and ukind == "energy":
        y = y * f / wl**2  # Jacobian |dx/dλ| for x = f / λ
    return Spectrum(wl, y, kind, name)


def read_spectrum(path, unit: str = "nm", kind: str = "emission", columns=(0, 1), delimiter=None,
                  density: bool = True, name: str | None = None, scale: float = 1.0) -> Spectrum:
    """Read a two-column spectrum from a text file (header and comment lines are skipped).

    Parameters
    ----------
    unit : str
        Unit of the spectral column: ``nm``, ``um``, ``angstrom``, ``m`` (wavelength), ``eV``,
        ``cm-1`` (wavenumber), ``THz`` or ``Hz``.
    kind : {"emission", "reflectance", "transmittance"}
    columns : (int, int)
        Zero-based indices of the spectral and value columns.
    density : bool
        If the values are a spectral *density* per energy/wavenumber/frequency unit, convert them to
        a density per nm with the Jacobian |dx/dλ|, so integrals and colours stay correct.
    scale : float
        Factor applied to the values (e.g. 0.01 for reflectance given in percent).
    """
    p = pathlib.Path(path)
    return parse_spectrum(p.read_text(encoding="utf-8", errors="replace"), unit=unit, kind=kind,
                          columns=columns, delimiter=delimiter, density=density, name=name or p.stem,
                          scale=scale)


def write_spectrum(spectrum: Spectrum, path) -> None:
    """Write a spectrum (or batch) as CSV with a ``wavelength_nm`` column."""
    spectrum.to_csv(path)


__all__ = ["parse_spectrum", "read_spectrum", "to_nanometres", "write_spectrum"]
