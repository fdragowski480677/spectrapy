"""Figure styles (light / dark) and a small save helper.

Plot functions never change global settings; they read the active matplotlib style, so these
themes apply to everything inside ``with sp.style.context("dark"):`` or after ``sp.style.use()``.
"""
from __future__ import annotations

import pathlib
import warnings
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from .colorspaces import get_colorspace, get_output_space, output_space, set_output_space  # noqa: F401

_COMMON = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Source Sans 3", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 9.0,
    "axes.labelsize": 9.5,
    "axes.labelpad": 5.0,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "grid.linewidth": 0.6,
    "figure.dpi": 110,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}

LIGHT = {
    **_COMMON,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "axes.edgecolor": "#5b6170",
    "axes.labelcolor": "#262a33",
    "axes.titlecolor": "#161920",
    "text.color": "#262a33",
    "xtick.color": "#5b6170",
    "ytick.color": "#5b6170",
    "xtick.labelcolor": "#4a4f5c",
    "ytick.labelcolor": "#4a4f5c",
    "grid.color": "#e7e9ef",
    "legend.labelcolor": "#262a33",
    "axes.prop_cycle": cycler(color=["#2f6db5", "#d1495b", "#2a9d8f", "#e09f3e", "#7b5ea7", "#3d405b"]),
}

DARK = {
    **_COMMON,
    "figure.facecolor": "#0d1017",
    "axes.facecolor": "#0d1017",
    "savefig.facecolor": "#0d1017",
    "axes.edgecolor": "#8a92a3",
    "axes.labelcolor": "#d7dbe4",
    "axes.titlecolor": "#eef0f5",
    "text.color": "#d7dbe4",
    "xtick.color": "#8a92a3",
    "ytick.color": "#8a92a3",
    "xtick.labelcolor": "#b7bdc9",
    "ytick.labelcolor": "#b7bdc9",
    "grid.color": "#232836",
    "legend.labelcolor": "#d7dbe4",
    "axes.prop_cycle": cycler(color=["#6ea8fe", "#ff7b8a", "#4fd1c5", "#f6c26b", "#b69cff", "#e2e4ea"]),
}

THEMES = {"light": LIGHT, "dark": DARK}


def theme(name: str = "light") -> dict:
    """The rcParams dictionary of a theme (``"light"`` or ``"dark"``)."""
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"unknown theme {name!r}; use 'light' or 'dark'") from None


def use(name: str = "light") -> None:
    """Apply a theme globally (like ``plt.style.use``)."""
    mpl.rcParams.update(theme(name))


@contextmanager
def context(name: str = "light"):
    """Apply a theme temporarily: ``with sp.style.context("dark"): ...``."""
    with mpl.rc_context(theme(name)):
        yield


def is_dark() -> bool:
    """True if the active axes background is dark."""
    r, g, b, _ = mpl.colors.to_rgba(mpl.rcParams["axes.facecolor"])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4


_RASTER = {"png", "jpg", "jpeg", "tif", "tiff", "webp"}


def save(fig, path, dpi: float | None = None, colorspace=None, **kwargs) -> pathlib.Path:
    """Save a figure, creating parent folders; the format follows the file extension.

    If the colours were produced in a wide-gamut space (``colorspace``, default: the active
    :func:`output_space`, e.g. ``"p3"``), raster formats get the matching ICC profile embedded so
    colour-managed viewers show the full gamut. Vector formats cannot carry the profile; a warning
    is issued because viewers will then interpret the colours as sRGB.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cs = get_colorspace(colorspace)
    fmt = str(kwargs.get("format") or p.suffix.lstrip(".") or mpl.rcParams["savefig.format"]).lower()
    if cs.name != "sRGB":
        if fmt in _RASTER:
            from .icc import icc_profile

            pil = dict(kwargs.pop("pil_kwargs", None) or {})
            pil.setdefault("icc_profile", icc_profile(cs))
            kwargs["pil_kwargs"] = pil
        else:
            warnings.warn(f"{fmt.upper()} files cannot carry an ICC profile; the {cs.name} colours will be "
                          "shown as if they were sRGB", stacklevel=2)
    fig.savefig(p, dpi=dpi or mpl.rcParams["savefig.dpi"], **kwargs)
    return p


def close(fig) -> None:
    plt.close(fig)
