"""Plotting with matplotlib. Every function draws into ``ax`` (or a new one) and returns it -
nothing is saved or shown behind your back, so panels compose freely into your own figures.

Use :mod:`spectrapy.style` for the light/dark themes and the ``save`` helper.
"""
from .chromaticity import chromaticity_diagram, chromaticity_points
from .patches import color_map, color_strip, cri_bars, swatches
from .spectra import (
    absorption_spectrum,
    cmf,
    emission_atlas,
    emission_spectrum,
    fill_colors,
    spectrum,
    visible_band,
    visible_spectrum,
    wavelength_bar,
)
from .tm30 import tm30_bars, tm30_cvg, tm30_report

__all__ = [
    "absorption_spectrum", "chromaticity_diagram", "chromaticity_points", "cmf", "color_map", "color_strip",
    "cri_bars", "emission_atlas", "emission_spectrum", "fill_colors", "spectrum", "swatches",
    "tm30_bars", "tm30_cvg", "tm30_report", "visible_band", "visible_spectrum",
    "wavelength_bar",
]
