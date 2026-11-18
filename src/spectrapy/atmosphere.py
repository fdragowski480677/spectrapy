"""Colours of the sun and sky from Rayleigh scattering (single-scattering, plane-parallel model).

A deliberately simple, transparent model for illustration and teaching:

* Rayleigh optical depth of the whole atmosphere: Bodhaine et al. (1999), eq. 30.
* Relative air mass: Kasten & Young (1989).
* Optional ozone absorption in the Chappuis band, approximated by a Gaussian band
  (peak cross-section ~5e-21 cm² near 602 nm) for a layer at ~22 km. It is what keeps the
  twilight zenith sky blue (Hulburt 1953) - without it the model predicts a greyish-yellow sky.
* Aerosols, multiple scattering and the spherical geometry of Rayleigh scattering are ignored.
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array
from .illuminants import blackbody
from .spectrum import Spectrum
from .spectrum import wavelengths as default_wavelengths

SUN_TEMPERATURE = 5772.0  #: effective temperature of the Sun [K] (IAU 2015 nominal)
_R_EARTH_KM = 6371.0
_DU = 2.6867e16  # molecules per cm² per Dobson unit


def rayleigh_optical_depth(wl_nm, pressure_hpa: float = 1013.25) -> np.ndarray:
    """Vertical Rayleigh optical depth of the atmosphere (Bodhaine et al. 1999, eq. 30)."""
    lam = as_float_array(wl_nm) * 1e-3
    l2 = lam**2
    tau = 0.0021520 * (1.0455996 - 341.29061 / l2 - 0.90230850 * l2) / (1.0 + 0.0027059889 / l2 - 85.968563 * l2)
    return tau * pressure_hpa / 1013.25


def airmass(elevation_deg) -> np.ndarray:
    """Relative optical air mass for a solar elevation in degrees (Kasten & Young 1989)."""
    z = 90.0 - np.clip(as_float_array(elevation_deg), 0.0, 90.0)
    return 1.0 / (np.cos(np.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def ozone_airmass(elevation_deg, height_km: float = 22.0) -> np.ndarray:
    """Air mass of a thin absorbing shell at ``height_km`` (spherical geometry)."""
    z = np.radians(90.0 - np.clip(as_float_array(elevation_deg), 0.0, 90.0))
    k = _R_EARTH_KM / (_R_EARTH_KM + height_km)
    return 1.0 / np.sqrt(1.0 - (k * np.sin(z)) ** 2)


def ozone_optical_depth(wl_nm, column_du: float = 300.0) -> np.ndarray:
    """Approximate vertical ozone optical depth in the visible (Gaussian Chappuis band)."""
    wl = as_float_array(wl_nm)
    sigma = 5.0e-21 * np.exp(-0.5 * ((wl - 602.0) / 48.0) ** 2)  # cm²
    return sigma * column_du * _DU


def _sun(wl):
    return blackbody(SUN_TEMPERATURE, wavelengths=wl).values


def direct_sunlight(elevation_deg, wavelengths=None, ozone: bool = True, pressure_hpa: float = 1013.25,
                    column_du: float = 300.0) -> Spectrum:
    """Spectrum of the direct solar beam at the ground (batch over ``elevation_deg``)."""
    wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths).ravel()
    e = as_float_array(elevation_deg)[..., None]
    tau = rayleigh_optical_depth(wl, pressure_hpa) * airmass(e)
    if ozone:
        tau = tau + ozone_optical_depth(wl, column_du) * ozone_airmass(e)
    return Spectrum(wl, _sun(wl) * np.exp(-tau), "emission", name="direct sunlight")


def zenith_sky(elevation_deg, wavelengths=None, ozone: bool = True, pressure_hpa: float = 1013.25,
               column_du: float = 300.0) -> Spectrum:
    """Spectral radiance of the zenith sky (single Rayleigh scattering, relative units).

    For an exponential atmosphere the scattering integral has a closed form:
    ``I ∝ S P(Θ) (e^(-τ m) - e^(-τ)) / (1 - m)``, with Θ the scattering angle (= solar zenith
    angle), ``P`` the Rayleigh phase function and ``m`` the solar air mass. Ozone is treated as a
    layer above the scatterers.
    """
    wl = default_wavelengths() if wavelengths is None else as_float_array(wavelengths).ravel()
    e = as_float_array(elevation_deg)[..., None]
    m = airmass(e)
    tau = rayleigh_optical_depth(wl, pressure_hpa)
    z = np.radians(90.0 - np.clip(e, 0.0, 90.0))
    phase = 3.0 / (16.0 * np.pi) * (1.0 + np.cos(z) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(m - 1.0) > 1e-6, (np.exp(-tau * m) - np.exp(-tau)) / (1.0 - m), tau * np.exp(-tau))
    I = _sun(wl) * phase * ratio
    if ozone:
        I = I * np.exp(-ozone_optical_depth(wl, column_du) * (ozone_airmass(e) + 1.0))
    return Spectrum(wl, I, "emission", name="zenith sky")
