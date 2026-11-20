"""ANSI/IES TM-30 graphics: colour vector graphic, hue-bin bar charts and a one-page report."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

from ..colorimetry import color
from ..quality import ces_spectra
from ..spectrum import Spectrum
from ._core import edge, fg, get_ax, is_dark

_THETA = np.radians(22.5 * np.arange(16) + 11.25)  # bin centre angles
_TEST_COLOR = "#d62f3c"


def bin_colors(result, pastel: float = 0.0) -> np.ndarray:
    """Display colour of each of the 16 hue bins: the mean reflectance of its samples, seen under the
    reference illuminant by an adapted observer. ``pastel`` (0-1) mixes the colours towards white."""
    ces = ces_spectra()
    mean = np.stack([ces.values[result.bins == j].mean(axis=0) for j in range(16)])
    cols = color(Spectrum(ces.wavelengths, mean, "reflectance"), illuminant=result.reference, adapt=True,
                 brightness=0.42)
    return (1.0 - pastel) * cols + pastel


def tm30_cvg(result, ax=None, background: bool = True, arrows: bool = True, annotate="corners",
             labels: bool = True, name: str | None = None):
    """TM-30 colour vector graphic: the reference (circle) and the test source's average hue-bin
    chromaticities (polygon), normalised so that each reference bin lies on the unit circle.
    Points outside the circle mean increased saturation, inside decreased.

    ``annotate``: ``"corners"`` (Rf, Rg, CCT, Duv in the corners), ``"title"`` (one title line,
    better for side-by-side panels; ``name`` is prepended) or ``False``.
    """
    ax = get_ax(ax, figsize=(4.8, 4.8))
    ref_pts = np.column_stack([np.cos(_THETA), np.sin(_THETA)])
    d = (result.bin_ab_test - result.bin_ab_ref) / np.linalg.norm(result.bin_ab_ref, axis=1)[:, None]
    test_pts = ref_pts + d
    if background:
        cols = bin_colors(result, pastel=0.35 if not is_dark(ax) else 0.0)
        for j, c in enumerate(cols):
            ax.add_patch(Wedge((0, 0), 1.48, 22.5 * j, 22.5 * (j + 1), facecolor=tuple(c), edgecolor="none",
                               alpha=0.85, zorder=0))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=fg(), lw=1.3, zorder=2))
    for r in (0.8, 1.2):
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=fg(), lw=0.5, ls=(0, (2, 3)), alpha=0.6, zorder=2))
    closed = np.vstack([test_pts, test_pts[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=_TEST_COLOR, lw=2.0, zorder=4, solid_joinstyle="round")
    ax.scatter(test_pts[:, 0], test_pts[:, 1], s=14, color=_TEST_COLOR, zorder=5)
    if arrows:
        for p, q in zip(ref_pts, test_pts):
            if np.linalg.norm(q - p) > 0.02:
                ax.annotate("", xy=q, xytext=p, zorder=5,
                            arrowprops={"arrowstyle": "-|>", "color": _TEST_COLOR, "lw": 1.0, "shrinkA": 0,
                                        "shrinkB": 0, "mutation_scale": 8})
    if labels:
        for j, t in enumerate(_THETA):
            ax.text(1.38 * np.cos(t), 1.38 * np.sin(t), str(j + 1), ha="center", va="center", fontsize=7,
                    color=fg(), zorder=6)
    if annotate == "title":
        head = f"{name}:  " if name else ""
        ax.set_title(f"{head}Rf {result.Rf:.0f}  ·  Rg {result.Rg:.0f}\n{result.cct:,.0f} K, Duv {result.duv:+.4f}",
                     loc="center", fontsize=9.5)
    elif annotate:
        kw = {"transform": ax.transAxes, "fontsize": 9, "color": fg()}
        ax.text(0.0, 1.0, f"Rf  {result.Rf:.0f}", ha="left", va="top", fontweight="bold", **kw)
        ax.text(1.0, 1.0, f"Rg  {result.Rg:.0f}", ha="right", va="top", fontweight="bold", **kw)
        ax.text(0.0, 0.0, f"CCT  {result.cct:,.0f} K", ha="left", va="bottom", **kw)
        ax.text(1.0, 0.0, f"Duv  {result.duv:+.4f}", ha="right", va="bottom", **kw)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def tm30_bars(result, metric: str = "Rcs", ax=None):
    """Hue-bin bar chart of ``metric``: ``"Rcs"`` (local chroma shift, %), ``"Rhs"`` (local hue shift,
    rad) or ``"Rf"`` (local fidelity), each bar in its bin's colour."""
    ax = get_ax(ax, figsize=(6.0, 2.4))
    values = {"Rcs": result.Rcs, "Rhs": result.Rhs, "Rf": result.Rf_hues}[metric]
    x = np.arange(1, 17)
    ax.bar(x, values, color=bin_colors(result), edgecolor=edge(), linewidth=0.4, width=0.78, zorder=3)
    ax.axhline(0.0, color=edge(), lw=0.8, zorder=2)
    if metric == "Rcs":
        lim = max(30.0, float(np.nanmax(np.abs(values))) + 6.0)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Chroma shift (%)")
        fmt = "{:+.0f}"
    elif metric == "Rhs":
        lim = max(0.2, float(np.nanmax(np.abs(values))) + 0.05)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Hue shift (rad)")
        fmt = "{:+.2f}"
    else:
        ax.set_ylim(0, 110)
        ax.set_ylabel("Local fidelity")
        fmt = "{:.0f}"
    for xi, v in zip(x, values):
        ax.annotate(fmt.format(v), (xi, v), xytext=(0, 2 if v >= 0 else -2), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, color=fg())
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlim(0.4, 16.6)
    return ax


def tm30_report(result, title: str | None = None, fig=None):
    """One-page TM-30 summary: colour vector graphic plus the three hue-bin charts. Returns the Figure."""
    fig = fig or plt.figure(figsize=(10.4, 5.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.55])
    tm30_cvg(result, ax=fig.add_subplot(gs[:, 0]))
    a1 = fig.add_subplot(gs[0, 1])
    tm30_bars(result, "Rcs", ax=a1)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    tm30_bars(result, "Rhs", ax=a2)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    tm30_bars(result, "Rf", ax=a3)
    for a in (a1, a2):
        a.tick_params(labelbottom=False)
    a3.set_xlabel("Hue-angle bin")
    fig.suptitle(title or f"ANSI/IES TM-30:  Rf {result.Rf:.0f}   Rg {result.Rg:.0f}   CCT {result.cct:,.0f} K",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig
