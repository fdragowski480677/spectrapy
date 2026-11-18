"""A wavelength colormap: colour-code anything by wavelength in your own matplotlib plots.

Example::

    sc = ax.scatter(x, y, c=wavelength_nm, cmap=sp.cmap.wavelength_cmap(), norm=sp.cmap.wavelength_norm())
    fig.colorbar(sc, label="Wavelength (nm)")
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

from .render import spectral_colors


def wavelength_cmap(lo: float = 380.0, hi: float = 750.0, brightness: str = "full", n: int = 512,
                    name: str | None = None) -> ListedColormap:
    """Colormap whose 0..1 range spans ``lo``-``hi`` nm (use with :func:`wavelength_norm`)."""
    rgb = spectral_colors(np.linspace(lo, hi, n), brightness=brightness)
    return ListedColormap(rgb, name=name or f"spectrapy_{brightness}_{lo:g}_{hi:g}")


def wavelength_norm(lo: float = 380.0, hi: float = 750.0) -> Normalize:
    return Normalize(vmin=lo, vmax=hi, clip=True)


def register(name: str = "spectrapy_wavelength") -> str:
    """Register the default wavelength colormap with matplotlib under ``name`` (idempotent)."""
    if name not in mpl.colormaps:
        mpl.colormaps.register(wavelength_cmap(name=name))
    return name
