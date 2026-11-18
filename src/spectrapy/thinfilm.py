"""Thin-film interference with the transfer-matrix method (multilayers, oblique incidence, s/p
polarisation, dispersive and absorbing materials), vectorized over wavelength, angle and thickness.

Conventions: time dependence exp(-iωt), complex refractive index ``n + iκ`` with ``κ >= 0`` for
absorbing media, angles in degrees, thicknesses in nm. All layers are treated coherently; for
layers much thicker than the coherence length of the light (> ~ 10 µm for sunlight) the fringes
become finer than the 1 nm colour-matching grid and should be averaged instead.
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array
from .spectrum import Spectrum
from .spectrum import wavelengths as default_wavelengths


class Material:
    """A (possibly dispersive, absorbing) optical material: ``material(wl_nm) -> complex n``."""

    def __init__(self, func, name: str = "material"):
        self._func = func
        self.name = name

    def n(self, wl_nm) -> np.ndarray:
        wl = as_float_array(wl_nm)
        return np.broadcast_to(np.asarray(self._func(wl), dtype=complex), wl.shape)

    __call__ = n

    def __repr__(self) -> str:
        return f"Material({self.name!r})"


def constant(n, name: str | None = None) -> Material:
    """Non-dispersive material with refractive index ``n`` (complex allowed)."""
    return Material(lambda wl: np.full(np.shape(wl), complex(n)), name or f"n = {n}")


def cauchy(A: float, B: float = 0.0, C: float = 0.0, name: str = "Cauchy") -> Material:
    """Cauchy dispersion ``n = A + B/λ² + C/λ⁴`` with λ in µm."""
    return Material(lambda wl: A + B / (wl * 1e-3) ** 2 + C / (wl * 1e-3) ** 4, name)


def sellmeier(B, C, name: str = "Sellmeier") -> Material:
    """Sellmeier dispersion ``n² = 1 + Σ Bᵢ λ² / (λ² - Cᵢ)`` with λ in µm and ``Cᵢ`` in µm²."""
    B = as_float_array(B)
    C = as_float_array(C)

    def f(wl):
        l2 = (as_float_array(wl) * 1e-3)[..., None] ** 2
        return np.sqrt(1.0 + np.sum(B * l2 / (l2 - C), axis=-1))

    return Material(f, name)


def tabulated(wl_nm, n, k=0.0, name: str = "tabulated") -> Material:
    """Material from tabulated ``n`` and ``k`` (linear interpolation, end values held outside)."""
    wl_t = as_float_array(wl_nm)
    nt = as_float_array(n)
    kt = np.broadcast_to(as_float_array(k), wl_t.shape)
    return Material(lambda wl: np.interp(wl, wl_t, nt) + 1j * np.interp(wl, wl_t, kt), name)


def as_material(m) -> Material:
    return m if isinstance(m, Material) else constant(m)


VACUUM = constant(1.0, "vacuum")
AIR = constant(1.000293, "air")
#: Water at 20 °C, Cauchy fit to n_C = 1.33114, n_D = 1.33299, n_F = 1.33712.
WATER = cauchy(1.32387, 0.003131, name="water")
#: Schott N-BK7 crown glass (Sellmeier coefficients from the Schott datasheet).
BK7 = sellmeier([1.03961212, 0.231792344, 1.01046945], [0.00600069867, 0.0200179144, 103.560653], "N-BK7")
#: Fused silica (Malitson 1965).
FUSED_SILICA = sellmeier([0.6961663, 0.4079426, 0.8974794],
                         [0.0684043**2, 0.1162414**2, 9.896161**2], "fused silica")
SIO2 = FUSED_SILICA
#: Magnesium fluoride as a typical thin-film coating (n ≈ 1.38).
MGF2 = constant(1.38, "MgF₂")
#: Titanium dioxide, rutile, ordinary ray (Devore 1951).
TIO2 = Material(lambda wl: np.sqrt(5.913 + 0.2441 / ((wl * 1e-3) ** 2 - 0.0803)), "TiO₂ (rutile)")
#: Light mineral oil (n ≈ 1.47).
OIL = constant(1.47, "oil")
#: Typical coating materials (approximate, non-dispersive).
AL2O3 = constant(1.63, "Al₂O₃")
ZRO2 = constant(2.05, "ZrO₂")
#: Crystalline silicon at 300 K - approximate n, k after Green (2008); held constant below 400 nm.
SI = tabulated([400, 450, 500, 550, 600, 650, 700, 750, 800, 850],
               [5.57, 4.68, 4.29, 4.08, 3.94, 3.84, 3.77, 3.72, 3.68, 3.65],
               [0.303, 0.0913, 0.0442, 0.0280, 0.0198, 0.0145, 0.0106, 0.0078, 0.0054, 0.0036],
               "silicon")


def tmm(indices, thicknesses, wl, angle=0.0, polarization: str = "unpolarized"):
    """Reflectance and transmittance of a layer stack (transfer-matrix / characteristic matrix method).

    Parameters
    ----------
    indices : sequence of array_like
        Complex refractive indices ``[n_ambient, n_1, ..., n_N, n_substrate]``, each broadcastable
        with ``wl``. The ambient medium must be lossless.
    thicknesses : sequence of array_like
        Thicknesses [nm] of the N inner layers (broadcastable, e.g. an array of shape ``(m, 1)``
        to scan a thickness for every wavelength at once).
    wl : array_like
        Vacuum wavelengths [nm].
    angle : array_like
        Angle of incidence in the ambient medium [degrees], broadcastable.
    polarization : {"s", "p", "unpolarized"}

    Returns
    -------
    R, T : ndarray
        Intensity reflectance and transmittance (broadcast shape of all inputs).
    """
    if polarization in ("u", "unpolarized", "unpolarised", "both"):
        Rs, Ts = tmm(indices, thicknesses, wl, angle, "s")
        Rp, Tp = tmm(indices, thicknesses, wl, angle, "p")
        return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
    if polarization not in ("s", "p"):
        raise ValueError("polarization must be 's', 'p' or 'unpolarized'")
    if len(thicknesses) != len(indices) - 2:
        raise ValueError("need one thickness per inner layer")
    wl = as_float_array(wl)
    n = [np.asarray(x, dtype=complex) for x in indices]
    if np.any(np.abs(n[0].imag) > 0):
        raise ValueError("the ambient medium must be lossless")
    s = n[0].real * np.sin(np.radians(as_float_array(angle)))  # invariant n sin(theta)

    def ncos(nj):
        v = np.sqrt(nj**2 - s**2 + 0j)
        return np.where(v.imag < 0, -v, v)  # forward-decaying / forward-propagating branch

    def admittance(nj, ncj):
        return ncj if polarization == "s" else nj**2 / ncj

    with np.errstate(divide="ignore", invalid="ignore"):
        e0 = admittance(n[0], ncos(n[0]))
        es = admittance(n[-1], ncos(n[-1]))
        m11, m12, m21, m22 = 1.0 + 0j, 0j, 0j, 1.0 + 0j
        for nj, dj in zip(n[1:-1], thicknesses):
            ncj = ncos(nj)
            ej = admittance(nj, ncj)
            delta = 2.0 * np.pi * as_float_array(dj) * ncj / wl
            c, sn = np.cos(delta), np.sin(delta)
            a12, a21 = -1j * sn / ej, -1j * ej * sn
            m11, m12, m21, m22 = m11 * c + m12 * a21, m11 * a12 + m12 * c, m21 * c + m22 * a21, m21 * a12 + m22 * c
        B = m11 + m12 * es
        C = m21 + m22 * es
        den = e0 * B + C
        r = (e0 * B - C) / den
        R = np.abs(r) ** 2
        T = 4.0 * e0.real * es.real / np.abs(den) ** 2
    return R, T


class Stack:
    """A multilayer: ``ambient | layer 1 | ... | layer N | substrate``.

    Example::

        soap = Stack([(WATER, 400)], ambient=AIR, substrate=AIR)   # 400 nm soap film
        R = soap.reflectance()                                      # Spectrum (reflectance)
        sp.color(R)                                                 # its colour under D65
    """

    def __init__(self, layers=(), ambient=AIR, substrate=AIR, name: str | None = None):
        self.layers = [(as_material(m), d) for m, d in layers]
        self.ambient = as_material(ambient)
        self.substrate = as_material(substrate)
        self.name = name

    def __repr__(self) -> str:
        inner = " | ".join(f"{m.name} {np.asarray(d).squeeze()} nm" if np.size(d) == 1 else f"{m.name} [scan]"
                           for m, d in self.layers)
        return f"Stack({self.ambient.name} | {inner} | {self.substrate.name})"

    def _indices(self, wl):
        return [self.ambient(wl), *[m(wl) for m, _ in self.layers], self.substrate(wl)]

    def rt(self, wavelengths=None, angle=0.0, polarization="unpolarized", thicknesses=None):
        """Raw ``(R, T)`` arrays; see :func:`tmm` for broadcasting rules."""
        wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths)
        ds = [d for _, d in self.layers] if thicknesses is None else thicknesses
        return tmm(self._indices(wl), ds, wl, angle, polarization)

    def _spectrum(self, which, wavelengths, angle, polarization):
        wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths).ravel()
        ang = as_float_array(angle)
        R, T = self.rt(wl, ang[..., None] if ang.ndim else ang, polarization)
        vals = R if which == "R" else T
        vals = np.broadcast_to(vals, ang.shape + wl.shape)
        return Spectrum(wl, vals, "reflectance" if which == "R" else "transmittance", self.name)

    def reflectance(self, wavelengths=None, angle=0.0, polarization="unpolarized") -> Spectrum:
        """Reflectance spectrum (a batch over ``angle`` if it is an array)."""
        return self._spectrum("R", wavelengths, angle, polarization)

    def transmittance(self, wavelengths=None, angle=0.0, polarization="unpolarized") -> Spectrum:
        """Transmittance into the substrate (a batch over ``angle`` if it is an array)."""
        return self._spectrum("T", wavelengths, angle, polarization)

    def scan(self, layer: int, thicknesses, wavelengths=None, angle=0.0, polarization="unpolarized",
             quantity: str = "R") -> Spectrum:
        """Spectra for a range of thicknesses of one layer: a batch of shape ``(len(thicknesses),)``."""
        wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths).ravel()
        d = as_float_array(thicknesses).ravel()
        ds = [dd for _, dd in self.layers]
        ds[layer] = d[:, None]
        R, T = self.rt(wl, angle, polarization, thicknesses=ds)
        vals = np.broadcast_to(R if quantity == "R" else T, d.shape + wl.shape)
        return Spectrum(wl, vals, "reflectance" if quantity == "R" else "transmittance", self.name)

    def xyz_map(self, layer: int, thicknesses, angles, illuminant="D65", observer="1931",
                polarization="unpolarized", wavelengths=None, quantity: str = "R", max_elements: float = 4e6):
        """XYZ (white ``Y = 1``) of the stack for every (angle, thickness): shape ``(n_angles, n_thick, 3)``.

        Works in chunks and reduces each chunk of spectra to XYZ immediately, so the full
        angle x thickness x wavelength cube is never stored.
        """
        from .colorimetry import xyz

        wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths).ravel()
        d = as_float_array(thicknesses).ravel()
        a = as_float_array(angles).ravel()
        out = np.empty((a.size, d.size, 3))
        chunk = max(1, int(max_elements // (a.size * wl.size)))
        for i in range(0, d.size, chunk):
            ds = [dd for _, dd in self.layers]
            ds[layer] = d[i:i + chunk][None, :, None]
            R, T = self.rt(wl, a[:, None, None], polarization, thicknesses=ds)
            vals = np.broadcast_to(R if quantity == "R" else T, (a.size, ds[layer].shape[1], wl.size))
            kind = "reflectance" if quantity == "R" else "transmittance"
            out[:, i:i + chunk] = xyz(Spectrum(wl, vals, kind), observer=observer, illuminant=illuminant)
        return out


def film(material, thickness, ambient=AIR, substrate=AIR) -> Stack:
    """A single film, e.g. ``film(WATER, 500)`` for a soap film in air."""
    return Stack([(material, thickness)], ambient=ambient, substrate=substrate)
