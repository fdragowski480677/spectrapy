"""Spectral distributions: sampled spectra (:class:`Spectrum`) and discrete lines (:class:`LineSpectrum`).

Conventions
-----------
* Wavelengths are in nanometres and must be ascending.
* ``Spectrum.values`` has shape ``(..., n)``: the *last* axis follows the wavelengths, any leading
  axes form a batch (e.g. 300 blackbodies, or a thickness x angle grid of thin films).
* ``kind`` says how the values are interpreted:

  ``"emission"``
      a spectral density of light per nanometre (radiance, irradiance, power, ...).
      Outside its sampled range the light is zero.
  ``"reflectance"`` / ``"transmittance"``
      a dimensionless factor that needs an illuminant to produce a colour. Outside its sampled
      range the end values are held (CIE 015 practice).
"""
from __future__ import annotations

import numbers

import numpy as np

from ._util import as_float_array, interp, trapezoid, union_grid

KINDS = ("emission", "reflectance", "transmittance")


def wavelengths(start: float = 360.0, stop: float = 830.0, step: float = 1.0) -> np.ndarray:
    """Inclusive wavelength grid in nm (default: the CIE 360-830 nm, 1 nm grid)."""
    n = int(round((stop - start) / step)) + 1
    return start + step * np.arange(n)


class Spectrum:
    """One spectrum or a batch of spectra sampled on a common wavelength grid.

    Parameters
    ----------
    wavelengths : array_like, shape (n,)
        Wavelengths in nm (sorted automatically if needed).
    values : array_like, shape (..., n)
        Spectral values; leading axes form a batch.
    kind : {"emission", "reflectance", "transmittance"}
    name : str, optional
    """

    __slots__ = ("wavelengths", "values", "kind", "name")

    def __init__(self, wavelengths, values, kind: str = "emission", name: str | None = None):
        wl = as_float_array(wavelengths).ravel()
        vals = as_float_array(values)
        if vals.ndim == 0:
            vals = np.full(wl.shape, float(vals))
        if vals.shape[-1] != wl.size:
            raise ValueError(f"last axis of values ({vals.shape[-1]}) must match wavelengths ({wl.size})")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if wl.size > 1 and np.any(np.diff(wl) <= 0):
            order = np.argsort(wl, kind="stable")
            wl = wl[order]
            vals = vals[..., order]
            if np.any(np.diff(wl) == 0):
                raise ValueError("wavelengths must be unique")
        self.wavelengths = wl
        self.values = vals
        self.kind = kind
        self.name = name

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple:
        """Batch shape (``values.shape[:-1]``); ``()`` for a single spectrum."""
        return self.values.shape[:-1]

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("a single Spectrum has no len(); use .wavelengths.size")
        return self.shape[0]

    def __getitem__(self, idx) -> Spectrum:
        if not self.shape:
            raise TypeError("cannot index a single spectrum")
        idx = idx if isinstance(idx, tuple) else (idx,)
        return Spectrum(self.wavelengths, self.values[idx + (Ellipsis,)], self.kind, self.name)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        wl = self.wavelengths
        batch = f", batch={self.shape}" if self.shape else ""
        name = f"{self.name!r}, " if self.name else ""
        return (f"Spectrum({name}{self.kind}, {wl[0]:g}-{wl[-1]:g} nm, "
                f"{wl.size} samples{batch})")

    def copy(self) -> Spectrum:
        return Spectrum(self.wavelengths.copy(), self.values.copy(), self.kind, self.name)

    def _with(self, values, kind=None, name=None) -> Spectrum:
        return Spectrum(self.wavelengths, values, kind or self.kind, self.name if name is None else name)

    # ------------------------------------------------------------- sampling
    def at(self, wl) -> np.ndarray:
        """Values at arbitrary wavelengths (linear interpolation; see module notes for outside)."""
        edge = "hold" if self.kind != "emission" else 0.0
        return interp(wl, self.wavelengths, self.values, edge, edge)

    __call__ = at

    def resample(self, wl) -> Spectrum:
        """Spectrum on a new wavelength grid."""
        wl = as_float_array(wl).ravel()
        return Spectrum(wl, self.at(wl), self.kind, self.name)

    def trim(self, lo: float, hi: float) -> Spectrum:
        """Keep only samples in ``[lo, hi]`` nm."""
        m = (self.wavelengths >= lo) & (self.wavelengths <= hi)
        return Spectrum(self.wavelengths[m], self.values[..., m], self.kind, self.name)

    # ------------------------------------------------------------ analysis
    def integrate(self, lo: float | None = None, hi: float | None = None) -> np.ndarray:
        """Trapezoidal integral over wavelength (optionally over ``[lo, hi]`` only)."""
        if lo is None and hi is None:
            return trapezoid(self.values, self.wavelengths)
        wl = self.wavelengths
        lo = wl[0] if lo is None else lo
        hi = wl[-1] if hi is None else hi
        g = union_grid(wl, lo=max(lo, wl[0]), hi=min(hi, wl[-1]))
        return trapezoid(self.at(g), g)

    @property
    def peak_wavelength(self) -> np.ndarray:
        """Wavelength of the maximum (per spectrum in a batch)."""
        return self.wavelengths[np.argmax(self.values, axis=-1)]

    def normalize(self, to="max", wavelength: float | None = None) -> Spectrum:
        """Scale the spectrum (each spectrum of a batch separately).

        ``to``: ``"max"`` (peak = 1), ``"area"`` (integral = 1), ``"at"`` (value at
        ``wavelength`` = 1, e.g. 560 nm as for CIE illuminants) or ``"luminance"`` (CIE Y = 1).
        """
        if to == "max":
            f = np.max(np.abs(self.values), axis=-1, keepdims=True)
        elif to == "area":
            f = self.integrate()[..., None]
        elif to == "at":
            if wavelength is None:
                raise ValueError("normalize(to='at') needs a wavelength")
            f = self.at(np.array([wavelength]))
        elif to == "luminance":
            from .colorimetry import xyz

            f = xyz(self)[..., 1:2]
        else:
            raise ValueError(f"unknown normalization {to!r}")
        f = np.where(f == 0, 1.0, f)
        return self._with(self.values / f)

    # ---------------------------------------------------------- arithmetic
    def _binary(self, other, op) -> Spectrum:
        if isinstance(other, Spectrum):
            lo = max(self.wavelengths[0], other.wavelengths[0])
            hi = min(self.wavelengths[-1], other.wavelengths[-1])
            if hi < lo:
                raise ValueError("spectra do not overlap in wavelength")
            if np.array_equal(self.wavelengths, other.wavelengths):
                g, a, b = self.wavelengths, self.values, other.values
            else:
                g = union_grid(self.wavelengths, other.wavelengths, lo=lo, hi=hi)
                a, b = self.at(g), other.at(g)
            kind = "emission" if "emission" in (self.kind, other.kind) else self.kind
            return Spectrum(g, op(a, b), kind, self.name)
        if isinstance(other, numbers.Number) or np.ndim(other) == 0:
            return self._with(op(self.values, float(other)))
        other = as_float_array(other)
        return self._with(op(self.values, other[..., None] if other.ndim else other))

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __add__(self, other):
        return self._binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    # ----------------------------------------------------------- shortcuts
    def xyz(self, observer="1931", illuminant=None, k=None) -> np.ndarray:
        """CIE XYZ tristimulus values; see :func:`spectrapy.xyz`."""
        from .colorimetry import xyz

        return xyz(self, observer=observer, illuminant=illuminant, k=k)

    def color(self, space="srgb", **kwargs) -> np.ndarray:
        """Displayable, encoded RGB in [0, 1]; see :func:`spectrapy.color`."""
        from .colorimetry import color

        return color(self, space=space, **kwargs)

    def hex(self, space="srgb", **kwargs):
        from .colorimetry import hex_color

        return hex_color(self, space=space, **kwargs)

    def plot(self, ax=None, **kwargs):
        """Plot with :func:`spectrapy.plot.spectrum`."""
        from .plot import spectrum

        return spectrum(self, ax=ax, **kwargs)

    def to_csv(self, path, header: bool = True) -> None:
        """Write ``wavelength_nm, value[, value, ...]`` columns to a CSV file."""
        vals = self.values.reshape(-1, self.wavelengths.size).T
        cols = ["wavelength_nm"] + [f"value{i}" if vals.shape[1] > 1 else "value" for i in range(vals.shape[1])]
        np.savetxt(path, np.column_stack([self.wavelengths, vals]), delimiter=",",
                   header=",".join(cols) if header else "", comments="", fmt="%.10g")


class LineSpectrum:
    """Discrete spectral lines: wavelengths (nm) and integrated line intensities.

    Lines are treated as exact delta functions for colour calculations, which is accurate even when
    the lines are far narrower than any wavelength grid (e.g. the sodium D doublet).
    """

    __slots__ = ("wavelengths", "intensities", "name", "labels")

    def __init__(self, wavelengths, intensities=1.0, name: str | None = None, labels=None):
        wl = as_float_array(wavelengths).ravel()
        it = np.broadcast_to(as_float_array(intensities), wl.shape).astype(float)
        order = np.argsort(wl, kind="stable")
        self.wavelengths = wl[order]
        self.intensities = it[order]
        self.name = name
        self.labels = None if labels is None else [list(labels)[i] for i in order]

    def __repr__(self) -> str:
        name = f"{self.name!r}, " if self.name else ""
        return f"LineSpectrum({name}{self.wavelengths.size} lines)"

    def __len__(self) -> int:
        return self.wavelengths.size

    def scaled(self, factor: float) -> LineSpectrum:
        return LineSpectrum(self.wavelengths, self.intensities * factor, self.name, self.labels)

    def normalize(self) -> LineSpectrum:
        """Strongest line = 1."""
        m = np.max(self.intensities) if self.intensities.size else 1.0
        return self.scaled(1.0 / m if m else 1.0)

    def __add__(self, other: LineSpectrum) -> LineSpectrum:
        labels = None
        if self.labels is not None and other.labels is not None:
            labels = list(self.labels) + list(other.labels)
        return LineSpectrum(np.concatenate([self.wavelengths, other.wavelengths]),
                            np.concatenate([self.intensities, other.intensities]),
                            self.name or other.name, labels)

    def broaden(self, fwhm: float = 0.5, profile: str = "gaussian", wavelengths=None) -> Spectrum:
        """Sampled emission spectrum with each line spread over a profile of unit area.

        The default grid spans the lines +- 10 FWHM with 20 samples per FWHM.
        """
        from .lines import line_profile

        if wavelengths is None:
            step = fwhm / 20.0
            lo = self.wavelengths.min() - 10 * fwhm
            hi = self.wavelengths.max() + 10 * fwhm
            wavelengths = np.arange(lo, hi + step / 2, step)
        wl = as_float_array(wavelengths)
        prof = line_profile(wl[None, :], self.wavelengths[:, None], fwhm, profile)
        return Spectrum(wl, self.intensities @ prof, "emission", self.name)

    def xyz(self, observer="1931", k=None) -> np.ndarray:
        from .colorimetry import xyz

        return xyz(self, observer=observer, k=k)

    def color(self, space="srgb", **kwargs) -> np.ndarray:
        from .colorimetry import color

        return color(self, space=space, **kwargs)
