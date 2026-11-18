"""Colour patches, colour-vs-parameter strips, 2D colour maps and CRI bar charts."""
from __future__ import annotations

import numpy as np

from ..colorspaces import rgb_to_hex
from ._core import edge, fg, get_ax, image_strip, strip_axes, swatch, text_color_for


def _as_rgb(colors):
    if isinstance(colors, (list, tuple)) and colors and isinstance(colors[0], str):
        from matplotlib.colors import to_rgb

        return np.array([to_rgb(c) for c in colors])
    return np.asarray(colors, dtype=float).reshape(-1, 3)


def swatches(colors, labels=None, ax=None, ncols: int | None = None, sublabels="hex", title: str | None = None,
             gap: float = 0.14, rounding: float = 0.14, label_inside: bool = False, fontsize: float = 8.0):
    """A grid of rounded colour patches with labels.

    ``colors``: encoded RGB in [0, 1] (shape ``(n, 3)``) or hex strings - e.g. ``sp.color(spectra)``.
    ``sublabels``: ``"hex"`` (default), a list of strings, or ``None``.
    """
    rgb = _as_rgb(colors)
    n = len(rgb)
    ncols = ncols or min(n, 6)
    nrows = int(np.ceil(n / ncols))
    ax = get_ax(ax, figsize=(1.25 * ncols + 0.3, 1.45 * nrows + (0.35 if title else 0.1)))
    subs = rgb_to_hex(rgb) if sublabels == "hex" else sublabels
    if isinstance(subs, str):
        subs = [subs]
    row_h = 1.0 + (0.0 if label_inside else 0.42)
    for i, c in enumerate(rgb):
        r, k = divmod(i, ncols)
        x, y = k, -r * row_h
        swatch(ax, c, (x + gap / 2, y), 1.0 - gap, 1.0 - gap, rounding=rounding, edgecolor=edge(), lw=0.4)
        lab = labels[i] if labels is not None else None
        sub = subs[i] if subs is not None else None
        if label_inside:
            tc = text_color_for(c)
            if lab:
                ax.text(x + 0.5, y + 0.55, lab, ha="center", va="center", fontsize=fontsize, color=tc,
                        fontweight="bold")
            if sub:
                ax.text(x + 0.5, y + 0.3, sub, ha="center", va="center", fontsize=fontsize - 1.5, color=tc)
        else:
            if lab:
                ax.text(x + 0.5, y - 0.06, lab, ha="center", va="top", fontsize=fontsize, color=fg())
            if sub:
                ax.text(x + 0.5, y - (0.24 if lab else 0.06), sub, ha="center", va="top", fontsize=fontsize - 1.5,
                        color=edge())
    ax.set_xlim(-0.05, ncols + 0.05)
    ax.set_ylim(-(nrows - 1) * row_h - (0.0 if label_inside else 0.45) - 0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def color_strip(x, colors, ax=None, xlabel: str | None = None, ticks: bool = True, label: str | None = None):
    """Colour as a function of a parameter (``colors`` encoded RGB, one row per ``x``)."""
    ax = get_ax(ax, figsize=(7.5, 0.9))
    x = np.asarray(x, dtype=float)
    image_strip(ax, x, _as_rgb(colors), interpolation="antialiased")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    strip_axes(ax, keep_x=ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
    return ax


def color_map(x, y, colors, ax=None, xlabel: str | None = None, ylabel: str | None = None):
    """2D colour image: ``colors`` of shape ``(len(y), len(x), 3)``, encoded RGB."""
    ax = get_ax(ax, figsize=(6.4, 3.6))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    dy = (y[-1] - y[0]) / max(len(y) - 1, 1)
    ax.imshow(np.asarray(colors), origin="lower", aspect="auto", interpolation="antialiased",
              extent=(x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def cri_bars(result, ax=None, show_values: bool = True, title: str | None = None):
    """Bar chart of the special colour rendering indices R1-R14 of a :func:`spectrapy.cri` result,
    each bar in the colour of its test sample (under the reference illuminant)."""
    from ..colorimetry import color
    from ..quality import tcs_spectra

    ax = get_ax(ax, figsize=(6.4, 3.0))
    cols = color(tcs_spectra(), illuminant=result.reference, adapt=True)
    idx = np.arange(1, 15)
    bars = ax.bar(idx, result.R, color=cols, edgecolor=edge(), linewidth=0.4, width=0.72, zorder=3)
    ax.axhline(100, color=edge(), lw=0.6, ls=(0, (3, 3)), zorder=2)
    ax.axhline(0, color=edge(), lw=0.8, zorder=2)
    ax.set_xticks(idx, [f"R{i}" for i in idx])
    lo = min(0.0, float(np.min(result.R)) - 8.0)
    ax.set_ylim(lo, 110)
    ax.set_ylabel("Special CRI  Rᵢ")
    if show_values:
        for b, v in zip(bars, result.R):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, max(v, 0)), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=fg())
    ax.set_title(title or f"Ra = {result.Ra:.1f}   ·   CCT {result.cct:,.0f} K   ·   R9 = {result.R[8]:.0f}")
    return ax
