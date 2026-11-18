"""Spectrum plots: coloured spectra, the visible spectrum, spectroscope views and CMFs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from ..colorimetry import color as _color
from ..colorimetry import xyz as _xyz
from ..lines import FRAUNHOFER_LABELS, line_spectrum
from ..observers import get_observer
from ..render import spectral_colors, strip_colors
from ..spectrum import LineSpectrum, Spectrum
from ._core import edge, fg, get_ax, image_fill, image_strip, is_dark, legible, neutral, strip_axes
from ._core import swatch as _draw_swatch


def visibility(wl, observer="1931", gamma: float = 0.35) -> np.ndarray:
    """0..1 weight of how visible each wavelength is (max of the CMFs, compressed by ``gamma``)."""
    obs = get_observer(observer)
    c = obs.at(np.asarray(wl, dtype=float)).max(-1)
    return np.clip(c / obs.cmf.max(), 0.0, 1.0) ** gamma


def fill_colors(wl, brightness="full", fade_invisible=True, ax=None, space=None) -> np.ndarray:
    """Spectral colours for filling under spectra; invisible wavelengths fade to a quiet grey."""
    rgb = spectral_colors(wl, brightness=brightness, space=space)
    if fade_invisible and brightness == "full":
        v = visibility(wl)[..., None]
        rgb = v * rgb + (1.0 - v) * neutral(ax)
    return rgb


def _swatch_label(s, rgb_hex):
    lines = [rgb_hex]
    if isinstance(s, Spectrum) and s.kind == "emission" and not s.shape:
        from ..temperature import cct

        X = _xyz(s)
        if X[1] > 0:
            T, duv = cct(X)
            if abs(float(duv)) < 0.02 and 1000.0 < float(T) < 99_000.0:
                lines.append(f"{float(T):,.0f} K")
    return "\n".join(lines)


def spectrum(s, ax=None, *, fill: bool = True, brightness: str = "full", fade_invisible: bool = True,
             alpha: float = 1.0, color=None, lw: float | None = None, label=None, swatch: bool = True,
             normalize: str | None = None, xlim=None, xlabel: str = "Wavelength (nm)", ylabel: str | None = None,
             illuminant=None, legend: bool = True, space=None):
    """Plot a spectrum with the area under the curve filled by the colours of its wavelengths.

    * single :class:`Spectrum`: coloured fill, outline, and (``swatch=True``) a patch on the
      right showing the colour of the whole spectrum with its hex code and CCT.
    * batch of spectra: one line per spectrum, drawn in that spectrum's own colour.
    * :class:`LineSpectrum`: coloured stems.

    Returns the matplotlib ``Axes``.
    """
    ax = get_ax(ax)
    lw = 1.3 if lw is None else lw
    if isinstance(s, LineSpectrum):
        return _line_stems(s, ax, normalize, xlim, xlabel, ylabel, label)
    if normalize:
        s = s.normalize(normalize)
    wl = s.wavelengths
    ylabel = ylabel or {"emission": "Spectral power (relative)" if normalize else "Spectral power density",
                        "reflectance": "Reflectance", "transmittance": "Transmittance"}[s.kind]
    if s.shape:
        vals = s.values.reshape(-1, wl.size)
        kw = {"illuminant": illuminant} if s.kind != "emission" else {}
        cols = _color(Spectrum(wl, vals, s.kind), space=space, brightness="max", **kw)
        labels = list(label) if isinstance(label, (list, tuple)) else [None] * len(vals)
        for v, c, lab in zip(vals, cols, labels):
            ax.plot(wl, v, color=legible(c, ax), lw=lw + 0.6, label=lab)
        if legend and any(lab is not None for lab in labels):
            ax.legend(loc="best")
    else:
        y = s.values
        if fill:
            image_fill(ax, wl, y, fill_colors(wl, brightness, fade_invisible, ax, space), alpha=alpha)
        ax.plot(wl, y, color=color or fg(), lw=lw, label=label, zorder=3)
        if swatch:
            kw = {"illuminant": illuminant} if s.kind != "emission" else {}
            c = _color(s, space=space, **kw)
            from ..colorspaces import rgb_to_hex

            sax = ax.inset_axes([1.025, 0.52, 0.13, 0.48])
            sax.set_axis_off()
            sax.set_xlim(0, 1)
            sax.set_ylim(0, 1)
            _draw_swatch(sax, c, (0.0, 0.32), 1.0, 0.68, rounding=0.1, edgecolor=edge(), lw=0.5)
            sax.text(0.5, 0.26, _swatch_label(s, rgb_to_hex(c)), ha="center", va="top", fontsize=7.5,
                     color=fg(), linespacing=1.3)
    lo, hi = (wl[0], wl[-1]) if xlim is None else xlim
    ax.set_xlim(lo, hi)
    vmax = np.nanmax(s.values) if s.values.size else 1.0
    ax.set_ylim(min(0.0, np.nanmin(s.values)), vmax * 1.06 if vmax > 0 else 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def _line_stems(lines: LineSpectrum, ax, normalize, xlim, xlabel, ylabel, label):
    it = lines.intensities / (lines.intensities.max() if normalize else 1.0)
    cols = spectral_colors(lines.wavelengths)
    ax.vlines(lines.wavelengths, 0, it, colors=cols, lw=2.2, zorder=3, label=label)
    ax.scatter(lines.wavelengths, it, s=10, c=cols, zorder=4, edgecolors="none")
    lo, hi = (min(380.0, lines.wavelengths.min() - 5), max(750.0, lines.wavelengths.max() + 5)) \
        if xlim is None else xlim
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, it.max() * 1.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or "Relative intensity")
    return ax


def visible_spectrum(ax=None, lo: float = 380.0, hi: float = 750.0, brightness: str = "full",
                     blur_nm: float | None = None, n: int = 1500, ticks: bool = True, xlabel: str | None = None):
    """The visible spectrum as a colour strip (see :func:`spectrapy.render.spectral_colors`)."""
    ax = get_ax(ax, figsize=(7.5, 1.1))
    wl = np.linspace(lo, hi, n)
    image_strip(ax, wl, spectral_colors(wl, brightness=brightness, blur_nm=blur_nm))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    strip_axes(ax, keep_x=ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def wavelength_bar(ax, lo: float | None = None, hi: float | None = None, height: float = 0.045,
                   brightness: str = "full", position: str = "bottom", pad: bool = True):
    """Add a thin visible-spectrum bar along the bottom (or top) edge of a wavelength axis.

    With ``pad=True`` the y range is extended so the bar sits outside the existing data.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if pad:
        span = (ylim[1] - ylim[0]) * height / (1.0 - height) * 1.25
        ylim = (ylim[0] - span, ylim[1]) if position == "bottom" else (ylim[0], ylim[1] + span)
    lo = xlim[0] if lo is None else lo
    hi = xlim[1] if hi is None else hi
    wl = np.linspace(lo, hi, 1000)
    y0, y1 = (0.0, height) if position == "bottom" else (1.0 - height, 1.0)
    im = image_strip(ax, wl, fill_colors(wl, brightness, True, ax), y0, y1, zorder=1.2,
                     transform=ax.get_xaxis_transform())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if position == "bottom":
        ax.spines["bottom"].set_visible(False)
    return im


def visible_band(ax, lo: float = 380.0, hi: float = 750.0, alpha: float = 0.16, zorder: float = 0.5):
    """Tint the visible part of a wide wavelength axis with the spectral colours (full height)."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    wl = np.linspace(lo, hi, 800)
    im = image_strip(ax, wl, spectral_colors(wl), 0.0, 1.0, zorder=zorder, transform=ax.get_xaxis_transform())
    im.set_alpha(alpha)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im


def emission_spectrum(obj, ax=None, lo: float = 380.0, hi: float = 720.0, n: int = 1400, fwhm: float = 0.8,
                      exposure: float = 1.6, tone: float = 0.5, label: str | None = None, annotate: bool = False,
                      min_label: float = 0.12, ticks: bool = True, background: str = "#000000"):
    """Spectroscope view of an emission spectrum or line list: bright lines on black.

    ``obj`` is a :class:`LineSpectrum`, an element symbol (``"H"``, ``"He"``, ``"Ne"``, ``"Na"``,
    ``"Hg"``) or an emission :class:`Spectrum`. ``exposure`` > 1 over-exposes bright lines, whose
    cores then turn white as on a camera sensor. ``tone`` < 1 compresses the brightness range so weak
    lines stay visible (``tone=1``: true relative luminance); see :func:`spectrapy.render.strip_colors`.
    """
    ax = get_ax(ax, figsize=(7.5, 0.9))
    if isinstance(obj, str):
        obj = line_spectrum(obj)
    rgb = strip_colors(obj, lo, hi, n, exposure=exposure, fwhm=fwhm, tone=tone)
    ax.set_facecolor(background)
    wl = np.linspace(lo, hi, n)
    image_strip(ax, wl, rgb, interpolation="antialiased")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    strip_axes(ax, keep_x=ticks)
    name = label if label is not None else getattr(obj, "name", None)
    if name:
        ax.text(0.006, 0.5, name, transform=ax.transAxes, ha="left", va="center", fontsize=8.5,
                color="#f2f2f2", fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#000000", alpha=0.8)])
    if annotate and isinstance(obj, LineSpectrum):
        keep = obj.intensities >= min_label * obj.intensities.max()
        for w in obj.wavelengths[keep & (obj.wavelengths >= lo) & (obj.wavelengths <= hi)]:
            ax.annotate(f"{w:.1f}", (w, 1.0), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=6.5, color=fg(), rotation=90)
    return ax


def absorption_spectrum(spectrum_: Spectrum, ax=None, lo: float = 380.0, hi: float = 720.0, n: int = 1400,
                        continuum: Spectrum | None = None, lines=FRAUNHOFER_LABELS, annotate: bool = True,
                        exposure: float = 1.0, ticks: bool = True, label: str | None = None,
                        min_sep: float = 0.012):
    """Spectroscope view of an absorption spectrum (e.g. :func:`spectrapy.solar_spectrum`).

    With a ``continuum`` the strip is shown relative to it - a uniform rainbow crossed by dark lines,
    as in textbook pictures of the Fraunhofer lines; without, the true brightness is kept.
    ``lines`` (``(label, wavelength, ...)`` tuples) are labelled above the strip; labels closer than
    ``min_sep`` (fraction of the width) to the previous one are skipped.
    """
    ax = get_ax(ax, figsize=(7.5, 1.0))
    rgb = strip_colors(spectrum_, lo, hi, n, exposure=exposure, flatten=continuum, highlights="scale")
    ax.set_facecolor("#000000")
    image_strip(ax, np.linspace(lo, hi, n), rgb, interpolation="antialiased")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    strip_axes(ax, keep_x=ticks)
    if annotate and lines:
        last = -np.inf
        for item in sorted(lines, key=lambda t: t[1]):
            lab, w = item[0], item[1]
            if lo <= w <= hi and (w - last) >= min_sep * (hi - lo):
                ax.annotate(lab, (w, 1.0), xytext=(0, 2.5), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=fg())
                last = w
    if label:
        ax.text(0.006, 0.5, label, transform=ax.transAxes, ha="left", va="center", fontsize=8.5,
                color="#f2f2f2", fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#000000", alpha=0.8)])
    return ax


def emission_atlas(elements=("H", "He", "Ne", "Na", "Hg"), lo: float = 380.0, hi: float = 720.0,
                   continuum: bool = True, sun: bool = True, width: float = 8.0, strip_height: float = 0.42,
                   exposure: float = 1.6, tone: float = 0.5, fig=None):
    """A stacked spectroscope figure: continuous spectrum, the Sun with Fraunhofer lines (widths
    exaggerated 3x for visibility), and the emission lines of several elements. Returns the ``Figure``."""
    from ..illuminants import blackbody
    from ..lines import solar_spectrum

    rows = (["continuum"] if continuum else []) + (["sun"] if sun else []) + list(elements)
    fig = fig or plt.figure(figsize=(width, strip_height * len(rows) + 0.75))
    axes = fig.subplots(len(rows), 1, sharex=True, gridspec_kw={"hspace": 0.12})
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, rows):
        last = ax is axes[-1]
        if row == "continuum":
            visible_spectrum(ax, lo, hi, ticks=last)
            ax.text(0.006, 0.5, "Continuous", transform=ax.transAxes, ha="left", va="center", fontsize=8.5,
                    color="#ffffff", fontweight="bold",
                    path_effects=[patheffects.withStroke(linewidth=2.0, foreground="#000000", alpha=0.8)])
        elif row == "sun":
            s = solar_spectrum(lo=lo - 2, hi=hi + 2, step=0.02, width_scale=3.0)
            absorption_spectrum(s, ax, lo, hi, continuum=blackbody(5772.0, wavelengths=s.wavelengths),
                                annotate=True, ticks=last, label="Sun")
        else:
            emission_spectrum(row, ax, lo, hi, exposure=exposure, tone=tone, ticks=last)
        if not last:
            ax.tick_params(axis="x", length=0, labelbottom=False)
    axes[-1].set_xlabel("Wavelength (nm)")
    return fig


def cmf(observer="1931", ax=None, compare=None, bar: bool = True, fill: bool = True):
    """Colour-matching functions x̄, ȳ, z̄ (optionally compared with a second observer, dashed)."""
    ax = get_ax(ax)
    dark = is_dark(ax)
    cols = ["#ff6b6b", "#5ee08a", "#6ea8fe"] if dark else ["#d1495b", "#2a9d6f", "#2f6db5"]
    names = ["x̄(λ)", "ȳ(λ)", "z̄(λ)"]
    obs = get_observer(observer)
    for i in range(3):
        ax.plot(obs.wavelengths, obs.cmf[:, i], color=cols[i], lw=1.8, label=f"{names[i]}  {obs.name}")
        if fill:
            ax.fill_between(obs.wavelengths, 0, obs.cmf[:, i], color=cols[i], alpha=0.10, lw=0)
    if compare is not None:
        o2 = get_observer(compare)
        for i in range(3):
            ax.plot(o2.wavelengths, o2.cmf[:, i], color=cols[i], lw=1.2, ls=(0, (4, 2.5)), alpha=0.9,
                    label=f"{names[i]}  {o2.name}")
    ax.set_xlim(360, 830)
    top = max(obs.cmf.max(), get_observer(compare).cmf.max() if compare is not None else 0.0)
    ax.set_ylim(0, top * 1.08)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Tristimulus value")
    ax.legend(loc="upper right", ncol=2 if compare is not None else 1)
    if bar:
        wavelength_bar(ax, height=0.05)
    return ax
