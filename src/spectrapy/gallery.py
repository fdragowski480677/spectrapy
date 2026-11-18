"""Example gallery: ``python -m spectrapy.gallery --out gallery`` (or ``spectrapy-gallery``).

Each ``fig_*`` function builds one figure with the public API and returns it - read them as
recipes. ``--list`` shows the available figures, ``--only a,b`` renders a subset.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib
import numpy as np

FIGURES: dict = {}


def figure(name: str, theme: str = "light", description: str = ""):
    def deco(func):
        FIGURES[name] = (func, theme, description)
        return func

    return deco


def _plt():
    import matplotlib.pyplot as plt

    return plt


@figure("hero", "dark", "Spectroscope atlas: continuous spectrum, the Sun and five discharge lamps")
def fig_hero():
    from . import plot

    fig = plot.emission_atlas(("H", "He", "Ne", "Na", "Hg"), width=8.4, strip_height=0.44)
    fig.suptitle("Spectra of light: continuum, the Sun's Fraunhofer lines and emission lines",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    return fig


@figure("spectrum_rendering", "dark", "How spectral colours are brought into the display gamut")
def fig_spectrum_rendering():
    from . import CIE1931_2, SRGB, plot
    from .render import spectral_colors

    plt = _plt()
    wl = np.linspace(380, 750, 1500)
    X = CIE1931_2.at(wl)
    rows = [
        ("clip each channel", SRGB.to_display(X, brightness="max", gamut="clip")),
        ("add white (desaturate)", SRGB.to_display(X, brightness="max", gamut="desaturate")),
        ("OKLCh chroma reduction", SRGB.to_display(X, brightness="max", gamut="oklch")),
        ("OKLCh + 8 nm blur  (default)", spectral_colors(wl)),
        ("true relative luminance", spectral_colors(wl, "luminance")),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.4, 0.46 * len(rows) + 0.7), sharex=True)
    for ax, (label, rgb) in zip(axes, rows):
        plot.color_strip(wl, rgb, ax=ax, ticks=ax is axes[-1], label=label)
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle("Monochromatic light is outside every display gamut - five ways to show it",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    return fig


@figure("blackbody", "light", "Planck spectra and the colour of blackbodies")
def fig_blackbody():
    from . import blackbody, color, planck, plot

    plt = _plt()
    fig = plt.figure(figsize=(8.4, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1])
    ax = fig.add_subplot(gs[0])
    T = np.array([1500, 2500, 3500, 5772, 8000, 12000])
    wl = np.geomspace(150, 5000, 900)
    s = blackbody(T, wavelengths=wl)
    plot.spectrum(s, ax=ax, label=[f"{t:,} K" for t in T], xlabel="Wavelength (nm)",
                  ylabel="Spectral radiance (W sr⁻¹ m⁻² nm⁻¹)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150, 5000)
    ax.set_ylim(1e-3, 3e5)
    peak = 2.897771955e6 / T  # Wien's displacement law, nm
    ax.plot(peak, planck(peak, T), ls=(0, (2, 2)), color="#777b86", lw=0.9, marker="o", ms=2.5, zorder=4)
    ax.annotate("Wien peaks", (peak[2], planck(peak[2], T[2])), xytext=(10, 10), textcoords="offset points",
                fontsize=7.5, color="#5b6170")
    ticks = [200, 300, 500, 1000, 2000, 5000]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plot.visible_band(ax)
    ax.set_title("Planck's law: hotter is brighter, bluer, and peaks at shorter wavelengths")
    ax.legend(loc="lower right", ncol=2)
    ax2 = fig.add_subplot(gs[1])
    Ts = np.linspace(1000, 12000, 800)
    plot.color_strip(Ts, color(blackbody(Ts)), ax=ax2, xlabel="Temperature (K)")
    ax2.set_title("Colour of a blackbody (brightness normalised)", fontsize=9.5)
    return fig


@figure("sun", "light", "Solar spectrum with Fraunhofer lines")
def fig_sun():
    from . import blackbody, plot, solar_spectrum
    from .lines import FRAUNHOFER_LABELS

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    s = solar_spectrum(lo=380, hi=760, step=0.02, width_scale=2.0)
    s = s / blackbody(5772, wavelengths=[560]).values[0]
    plot.spectrum(s, ax=ax, lw=0.7, ylabel="Relative spectral radiance")
    for lab, w in FRAUNHOFER_LABELS:
        if 380 <= w <= 760:
            ax.annotate(lab, (w, 1.08), ha="center", va="bottom", fontsize=7.5, color="#4a4f5c",
                        annotation_clip=False)
    ax.set_ylim(0, 1.18)
    ax.set_title("The Sun: a 5772 K blackbody with the classic Fraunhofer absorption lines")
    return fig


@figure("cmf", "light", "CIE colour-matching functions, 1931 2° vs 1964 10°")
def fig_cmf():
    from . import plot

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    plot.cmf("1931", ax=ax, compare="1964")
    ax.set_title("CIE colour-matching functions: 2° (1931, solid) and 10° (1964, dashed) observers")
    return fig


@figure("chromaticity", "light", "CIE 1931 xy and 1976 u'v' diagrams with gamuts and light sources")
def fig_chromaticity():
    from . import illuminant, line_spectrum, plot

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4))
    sources = [illuminant(n) for n in ("A", "D65", "FL11", "LED-RGB1")] + [line_spectrum(e) for e in ("Na", "Hg")]
    names = ["A", "D65", "FL11", "LED-RGB1", "Na lamp", "Hg lamp"]
    offsets = [(-4, 9), (-30, -2), (-14, 9), (12, -16), (8, 3), (-50, -4)]
    for ax, kind in zip(axes, ("xy", "uv")):
        plot.chromaticity_diagram(ax=ax, kind=kind, gamuts=("sRGB", "Display P3", "Rec. 2020"),
                                  temperatures=(1500, 2500, 4000, 10000))
        for s, n, off in zip(sources, names, offsets):
            plot.chromaticity_points(ax, s, [n], kind=kind, size=26, offsets=[off])
    axes[0].set_title("CIE 1931 (x, y)")
    axes[1].set_title("CIE 1976 UCS (u′, v′)")
    return fig


@figure("light_sources", "light", "Colours of common light sources")
def fig_light_sources():
    from . import blackbody, cct, color, illuminant, line_spectrum, plot, xyz

    srcs = [("Candle", blackbody(1850)), ("Incandescent (A)", illuminant("A")),
            ("Halogen", blackbody(3200)), ("D50", illuminant("D50")), ("Daylight D65", illuminant("D65")),
            ("Overcast D75", illuminant("D75")), ("Fluorescent FL2", illuminant("FL2")),
            ("Tri-band FL11", illuminant("FL11")), ("LED B3", illuminant("LED-B3")),
            ("LED RGB1", illuminant("LED-RGB1")), ("Sodium lamp", line_spectrum("Na")),
            ("Mercury lamp", line_spectrum("Hg"))]
    cols, subs = [], []
    for _, s in srcs:
        cols.append(color(s))
        T, duv = cct(xyz(s))
        subs.append(f"{float(T):,.0f} K" if abs(float(duv)) < 0.02 else f"Duv {float(duv):+.3f}")
    ax = plot.swatches(np.array(cols), [n for n, _ in srcs], ncols=6, sublabels=subs)
    ax.set_title("The colour of light sources (correlated colour temperature below)")
    return ax.figure


@figure("illuminants", "light", "Spectra of CIE illuminants with colour swatches")
def fig_illuminants():
    from . import cri, illuminant, plot

    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6))
    for ax, name in zip(axes.ravel(), ("A", "D65", "FL11", "LED-B3")):
        s = illuminant(name).trim(360, 780).normalize("max")
        plot.spectrum(s, ax=ax, ylabel="Relative power")
        r = cri(illuminant(name))
        ax.set_title(f"{name}   ·   Ra {r.Ra:.0f}", fontsize=10)
        ax.set_xlim(360, 780)
    return fig


@figure("cri", "light", "Colour rendering index of a fluorescent and an LED source")
def fig_cri():
    from . import cri, illuminant, plot

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4))
    for ax, name in zip(axes, ("FL2", "LED-B5")):
        r = cri(illuminant(name))
        plot.cri_bars(r, ax=ax, title=f"{name}:  Ra = {r.Ra:.1f},  R9 = {r.R[8]:.0f},  CCT {r.cct:,.0f} K")
    return fig


@figure("thin_films", "light", "Interference colours of thin films")
def fig_thin_films():
    from . import color, plot
    from . import thinfilm as tf

    plt = _plt()
    fig = plt.figure(figsize=(8.4, 6.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 0.55, 0.55, 2.2])
    d = np.linspace(0, 1500, 1200)
    cases = [("Soap film in air", tf.Stack([(tf.WATER, 0)], tf.AIR, tf.AIR), d),
             ("Oil on water", tf.Stack([(tf.OIL, 0)], tf.AIR, tf.WATER), d),
             ("SiO₂ on silicon", tf.Stack([(tf.SIO2, 0)], tf.AIR, tf.SI), d)]
    for i, (label, stack, dd) in enumerate(cases):
        ax = fig.add_subplot(gs[i])
        plot.color_strip(dd, color(stack.scan(0, dd), brightness="global"), ax=ax, ticks=(i == 2), label=label)
        if i == 2:
            ax.set_xlabel("Film thickness (nm)")
    ax = fig.add_subplot(gs[3])
    soap = tf.Stack([(tf.WATER, 0)], tf.AIR, tf.AIR)
    spectra = soap.scan(0, [120, 300, 450, 900])
    plot.spectrum(spectra, ax=ax, label=["120 nm", "300 nm", "450 nm", "900 nm"], ylabel="Reflectance")
    ax.set_title("Soap-film reflectance for four thicknesses (lines in the film's colour)", fontsize=9.5)
    ax.set_xlim(380, 780)
    fig.suptitle("Thin-film interference (transfer-matrix method, D65 light, normal incidence)",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    return fig


@figure("iridescence", "dark", "Soap-film colour vs thickness and viewing angle")
def fig_iridescence():
    from . import SRGB, plot
    from . import thinfilm as tf

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    d = np.linspace(0, 1200, 480)
    a = np.linspace(0, 85, 170)
    wl = np.arange(380.0, 781.0, 5.0)  # 5 nm is plenty: the fringes are >= 90 nm apart up to 1.2 µm
    X = tf.Stack([(tf.WATER, 0)], tf.AIR, tf.AIR).xyz_map(0, d, a, wavelengths=wl)
    plot.color_map(d, a, SRGB.to_display(X, brightness="global"), ax=ax, xlabel="Film thickness (nm)",
                   ylabel="Angle of incidence (°)")
    ax.set_title("Iridescence of a soap film: colour vs thickness and viewing angle (unpolarised D65)")
    return fig


@figure("ar_coating", "light", "Anti-reflection coatings on glass")
def fig_ar_coating():
    from . import color, plot
    from . import thinfilm as tf

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    q = lambda n, lam=550.0: lam / (4.0 * n)  # noqa: E731 - quarter-wave thickness
    stacks = {
        "uncoated BK7": tf.Stack([], tf.AIR, tf.BK7),
        "MgF₂ λ/4": tf.Stack([(tf.MGF2, q(1.38))], tf.AIR, tf.BK7),
        "MgF₂ λ/4 · ZrO₂ λ/2 · Al₂O₃ λ/4": tf.Stack([(tf.MGF2, q(1.38)), (tf.ZRO2, 2 * q(2.05)),
                                                   (tf.AL2O3, q(1.63))], tf.AIR, tf.BK7),
    }
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch

    wl = np.linspace(380, 780, 401)
    handles, labels = [], []
    for (name, st), c in zip(stacks.items(), ("#5b6170", "#7b5ea7", "#2a9d8f")):
        R = st.reflectance(wl)
        (line,) = ax.plot(wl, 100 * R.values, color=c, lw=1.8)
        tint = Patch(facecolor=tuple(color(R, brightness="max")), edgecolor="#5b6170", linewidth=0.5)
        handles.append((line, tint))
        labels.append(name)
    ax.set_xlim(380, 780)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance (%)")
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, loc="center right",
              handlelength=3.4, handleheight=1.4, title="coating  ·  colour of the reflection", title_fontsize=8)
    plot.wavelength_bar(ax, height=0.04)
    ax.set_title("Anti-reflection coatings on BK7 glass (designed for 550 nm)")
    return fig


@figure("sky", "light", "Colours of the sun and the zenith sky vs solar elevation")
def fig_sky():
    from . import atmosphere as atm
    from . import color, plot

    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 2.6), sharex=True)
    e = np.linspace(0, 60, 600)
    rows = [("Sun disk", atm.direct_sunlight(e)), ("Zenith sky", atm.zenith_sky(e)),
            ("Zenith sky, no ozone", atm.zenith_sky(e, ozone=False))]
    for ax, (label, s) in zip(axes, rows):
        plot.color_strip(e, color(s), ax=ax, ticks=ax is axes[-1], label=label)
    axes[-1].set_xlabel("Solar elevation (°)")
    fig.suptitle("Rayleigh scattering: the setting sun reddens; ozone keeps the twilight sky blue",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    return fig


@figure("rainbow", "dark", "A rainbow computed from water dispersion (Descartes' theory)")
def fig_rainbow():
    from . import SRGB, xyz
    from . import atmosphere as atm
    from .optics import rainbow_spectra

    plt = _plt()
    th = np.arange(0.0, 75.0, 0.02)
    X = xyz(rainbow_spectra(th))
    X = X / X[:, 1].max()
    sun_elev = 8.0
    sky = xyz(atm.zenith_sky(35.0))
    sky = sky / sky[1]
    W, H = 1400, 620
    x = np.linspace(-62, 62, W)
    y = np.linspace(-4, 52, H)
    gx, gy = np.meshgrid(x, y)
    theta = np.hypot(gx, gy + sun_elev)  # angle from the antisolar point (below the horizon)
    Xr = np.stack([np.interp(theta, th, X[:, k]) for k in range(3)], axis=-1)
    grad = (0.55 + 0.45 * np.clip(gy / 52.0, 0, 1))[..., None]  # sky darker towards the zenith
    img = SRGB.to_display(0.2 * sky * grad + 0.62 * Xr, highlights="whiten")
    ground = gy < 0
    img[ground] = np.array([0.07, 0.09, 0.08]) + 0.04 * np.random.default_rng(0).random((ground.sum(), 1))
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.imshow(img, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), interpolation="antialiased")
    ax.set_xlabel("Azimuth from the antisolar direction (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(f"Primary and secondary rainbow with Alexander's dark band (sun {sun_elev:g}° high)")
    return fig


def render(names=None, out="gallery", fmt: str = "png", dpi: float = 150, verbose: bool = True):
    """Render gallery figures to ``out``; returns the list of written paths."""
    matplotlib.use("Agg")
    from . import style

    plt = _plt()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names or list(FIGURES):
        func, theme, _ = FIGURES[name]
        t0 = time.perf_counter()
        with style.context(theme):
            fig = func()
            path = style.save(fig, out / f"{name}.{fmt}", dpi=dpi)
        plt.close(fig)
        written.append(path)
        if verbose:
            print(f"{name:20s} {time.perf_counter() - t0:6.2f} s  ->  {path}")
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Render the spectrapy example gallery.")
    ap.add_argument("--out", default="gallery", help="output folder (default: ./gallery)")
    ap.add_argument("--format", default="png", help="png, svg, pdf, ...")
    ap.add_argument("--dpi", type=float, default=150)
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    args = ap.parse_args(argv)
    if args.list:
        for name, (_, theme, desc) in FIGURES.items():
            print(f"{name:20s} [{theme}]  {desc}")
        return
    names = [n.strip() for n in args.only.split(",")] if args.only else None
    unknown = [n for n in names or [] if n not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}; see --list")
    render(names, args.out, args.format, args.dpi)


if __name__ == "__main__":
    main()
