"""spectrapy - physically based spectral colour calculations and spectral plots.

Quick start::

    import spectrapy as sp

    sun = sp.blackbody(5772)                # Planck spectrum, W sr^-1 m^-2 nm^-1
    sp.xyz(sun)                             # CIE XYZ (Y = luminance in cd/m^2)
    sp.hex_color(sun)                       # '#ffeee3'-ish: the colour of the spectrum
    sp.cct(sp.xyz(sun))                     # correlated colour temperature and Duv

    ax = sp.plot.spectrum(sun)              # matplotlib Axes, nothing saved behind your back
    sp.style.save(ax.figure, "sun.png")

Plotting lives in :mod:`spectrapy.plot` (imported on first use, so the numerical core does not
import matplotlib).
"""
from __future__ import annotations

import importlib

from . import (
    adaptation,
    atmosphere,
    colorimetry,
    colorspaces,
    io,
    lines,
    optics,
    quality,
    render,
    temperature,
    thinfilm,
    uniform,
)
from .colorimetry import (
    color,
    dominant_wavelength,
    hex_color,
    illuminant_white,
    uv1960_to_xy,
    uv1976_to_xy,
    xy_to_uv1960,
    xy_to_uv1976,
    xy_to_xyz,
    xyz,
    xyz_to_uv1960,
    xyz_to_uv1976,
    xyz_to_xy,
)
from .colorspaces import ADOBE_RGB, DISPLAY_P3, LINEAR_SRGB, REC2020, SRGB, RGBColorSpace, get_colorspace
from .constants import K_M
from .illuminants import (
    WHITE_POINTS,
    available_illuminants,
    blackbody,
    daylight,
    illuminant,
    planck,
)
from .lines import line_spectrum, solar_spectrum
from .observers import CIE1931_2, CIE1964_10, Observer, get_observer
from .quality import cri
from .spectrum import LineSpectrum, Spectrum, wavelengths
from .temperature import cct, planckian_xy, xy_from_cct
from .thinfilm import Stack

__version__ = "0.1.0"

_LAZY = {"plot", "style", "cmap", "gallery"}


def __getattr__(name):  # PEP 562: import the matplotlib-based modules on first access
    if name in _LAZY:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'spectrapy' has no attribute {name!r}")


__all__ = [
    "ADOBE_RGB", "CIE1931_2", "CIE1964_10", "DISPLAY_P3", "K_M", "LINEAR_SRGB", "REC2020", "SRGB",
    "WHITE_POINTS", "LineSpectrum", "Observer", "RGBColorSpace", "Spectrum", "Stack", "adaptation",
    "atmosphere", "available_illuminants", "blackbody", "cct", "color", "colorimetry", "colorspaces",
    "cri", "daylight", "dominant_wavelength", "get_colorspace", "get_observer", "hex_color",
    "illuminant", "illuminant_white", "io", "line_spectrum", "lines", "optics", "planck", "planckian_xy",
    "quality", "render", "solar_spectrum", "temperature", "thinfilm", "uniform", "uv1960_to_xy", "uv1976_to_xy",
    "wavelengths", "xy_from_cct", "xy_to_uv1960", "xy_to_uv1976", "xy_to_xyz", "xyz", "xyz_to_uv1960",
    "xyz_to_uv1976", "xyz_to_xy", "plot", "style", "cmap",
]
