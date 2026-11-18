"""Shared plotting helpers: axes handling, theme colours and image-based colour fills."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path


def get_ax(ax=None, figsize=(6.4, 3.4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def fg() -> str:
    return mpl.rcParams["text.color"]


def edge() -> str:
    return mpl.rcParams["axes.edgecolor"]


def is_dark(ax=None) -> bool:
    r, g, b, _ = mpl.colors.to_rgba(ax.get_facecolor() if ax is not None else mpl.rcParams["axes.facecolor"])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4


def neutral(ax=None) -> np.ndarray:
    """A quiet grey used for 'invisible' parts of spectra, adapted to the background."""
    return np.array(mpl.colors.to_rgb("#2b303c" if is_dark(ax) else "#d9dce3"))


def uniform_colors(x, colors):
    """Resample per-sample colours onto a uniform grid so they can be drawn as an image."""
    x = np.asarray(x, dtype=float)
    colors = np.asarray(colors, dtype=float)
    if x.size < 2:
        return x, colors
    dx = np.diff(x)
    if np.allclose(dx, dx[0], rtol=1e-6, atol=1e-9):
        return x, colors
    n = int(min(6000, max(x.size, np.ceil((x[-1] - x[0]) / dx.min()) + 1)))
    xu = np.linspace(x[0], x[-1], n)
    return xu, np.stack([np.interp(xu, x, colors[:, c]) for c in range(colors.shape[1])], axis=-1)


def _extent(xu):
    dx = (xu[-1] - xu[0]) / (xu.size - 1) if xu.size > 1 else 1.0
    return xu[0] - dx / 2.0, xu[-1] + dx / 2.0


def image_fill(ax, x, y, colors, baseline=0.0, alpha=1.0, zorder=1.5):
    """Fill between ``baseline`` and the curve ``y(x)`` with a colour per ``x`` (one image, clipped to
    the area under the curve): fast, seamless and small in vector output."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xu, cu = uniform_colors(x, colors)
    x0, x1 = _extent(xu)
    ymin, ymax = min(np.nanmin(y), baseline), max(np.nanmax(y), baseline)
    if ymax == ymin:
        ymax = ymin + 1.0
    poly = ax.fill_between(x, baseline, y, facecolor="none", edgecolor="none", linewidth=0, zorder=zorder)
    im = ax.imshow(cu[None, :, :], extent=(x0, x1, ymin, ymax), aspect="auto", origin="lower",
                   interpolation="bilinear", alpha=alpha, zorder=zorder)
    im.set_clip_path(Path.make_compound_path(*poly.get_paths()), transform=ax.transData)
    return im


def image_strip(ax, x, colors, y0=0.0, y1=1.0, zorder=1.0, transform=None, interpolation="bilinear"):
    """Draw per-``x`` colours as a horizontal strip from ``y0`` to ``y1``."""
    xu, cu = uniform_colors(np.asarray(x, dtype=float), colors)
    x0, x1 = _extent(xu)
    kw = {} if transform is None else {"transform": transform}
    return ax.imshow(cu[None, :, :], extent=(x0, x1, y0, y1), aspect="auto", origin="lower",
                     interpolation=interpolation, zorder=zorder, **kw)


def swatch(ax, rgb, xy, width, height, transform=None, rounding=0.12, edgecolor=None, lw=0.6, zorder=3):
    """A rounded colour patch (``FancyBboxPatch``); returns the patch."""
    rs = rounding * min(width, height)
    p = FancyBboxPatch(xy, width, height, boxstyle=f"round,pad=0,rounding_size={rs}",
                       facecolor=tuple(np.clip(rgb, 0, 1)), edgecolor=edgecolor or "none", linewidth=lw,
                       transform=transform if transform is not None else ax.transData, zorder=zorder,
                       clip_on=False)
    ax.add_patch(p)
    return p


def legible(rgb, ax=None, min_contrast: float = 2.2) -> np.ndarray:
    """Mix a line colour towards the text colour until it has at least ``min_contrast`` (WCAG ratio)
    against the axes background - keeps near-white colours visible on white and dark ones on black."""

    def lum(c):
        c = np.asarray(c, dtype=float)[:3]
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    bgc = np.array(mpl.colors.to_rgb(ax.get_facecolor() if ax is not None else mpl.rcParams["axes.facecolor"]))
    target = np.array(mpl.colors.to_rgb(fg()))
    c = np.clip(np.asarray(rgb, dtype=float)[:3], 0, 1)
    lb = lum(bgc)
    for t in np.linspace(0.0, 1.0, 21):
        m = (1 - t) * c + t * target
        lm = lum(m)
        if (max(lm, lb) + 0.05) / (min(lm, lb) + 0.05) >= min_contrast:
            return m
    return target


def text_color_for(rgb) -> str:
    """Black or white, whichever reads better on the (encoded) colour ``rgb``."""
    r, g, b = np.asarray(rgb, dtype=float)[:3]
    lin = np.where(np.array([r, g, b]) <= 0.04045, np.array([r, g, b]) / 12.92,
                   ((np.array([r, g, b]) + 0.055) / 1.055) ** 2.4)
    return "#111111" if (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]) > 0.22 else "#ffffff"


def strip_axes(ax, keep_x: bool = True):
    """Remove frame and y axis (for colour strips). Safe for axes with a shared x axis."""
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(keep_x)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    if not keep_x:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
