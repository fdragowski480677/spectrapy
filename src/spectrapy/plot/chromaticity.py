"""Chromaticity diagrams (CIE 1931 xy and CIE 1976 u'v') with gamuts and the Planckian locus."""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path

from ..colorimetry import uv1976_to_xy, xy_to_uv1976, xy_to_xyz, xyz_to_xy
from ..colorimetry import xyz as _xyz
from ..colorspaces import get_colorspace, map_gamut
from ..illuminants import WHITE_POINTS
from ..observers import get_observer
from ..render import _gaussian_blur
from ..spectrum import LineSpectrum, Spectrum
from ..temperature import isotherms, planckian_xy
from ._core import fg, get_ax, is_dark

LIMITS = {"xy": ((-0.01, 0.80), (-0.01, 0.90)), "uv": ((-0.01, 0.64), (-0.01, 0.61))}
LABEL_WL = {"xy": [460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700],
            "uv": [420, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]}
_DASHES = ["-", (0, (5, 2.5)), (0, (1.5, 2)), (0, (6, 2, 1.5, 2))]


def _to(p, kind):
    return p if kind == "xy" else xy_to_uv1976(p)


def _from(p, kind):
    return p if kind == "xy" else uv1976_to_xy(p)


def _fill_image(kind, resolution, observer, poly, blur: float):
    (x0, x1), (y0, y1) = LIMITS[kind]
    gx, gy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    pts = np.stack([gx, gy], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.nan_to_num(xy_to_xyz(_from(pts, kind)), nan=0.0, posinf=0.0, neginf=0.0)
    cs = get_colorspace(None)
    rgb = map_gamut(cs.from_xyz(X), cs.luminance_weights, "desaturate")
    if blur > 0:  # soften the creases of the gamut mapping (same idea as for spectrum strips)
        rgb = _gaussian_blur(_gaussian_blur(rgb, blur, axis=0), blur, axis=1)
    m = rgb.max(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(m > 0, rgb / m, 0.0)
    rgb = np.clip(cs.encode(np.clip(rgb, 0, 1)), 0, 1)
    inside = Path(poly).contains_points(pts.reshape(-1, 2)).reshape(gx.shape)
    return np.concatenate([rgb, inside[..., None].astype(float)], axis=-1), (x0, x1, y0, y1)


def chromaticity_diagram(ax=None, kind: str = "xy", observer="1931", fill: bool = True, resolution: int = 700,
                         gamuts=("sRGB",), planckian: bool = True, temperatures=(1500, 2000, 3000, 4000, 6500,
                                                                                    10000),
                         daylight: bool = False, wavelength_labels: bool = True, white_points=(),
                         grid: bool = True, blur: float = 2.0, locus_lw: float = 1.0, legend_loc: str | None = None):
    """CIE chromaticity diagram (``kind="xy"`` or ``"uv"`` for CIE 1976 u'v').

    The horseshoe is filled with the brightest displayable colour of every chromaticity (colours
    outside sRGB are desaturated towards white, so the fill is an honest *approximation* there).
    ``gamuts`` draws RGB colour-space triangles, ``planckian`` the blackbody locus with
    isotemperature lines at ``temperatures``, ``daylight`` the CIE daylight locus.
    Returns the ``Axes``; overlay sources with :func:`chromaticity_points`.
    """
    if kind not in LIMITS:
        raise ValueError("kind must be 'xy' or 'uv'")
    ax = get_ax(ax, figsize=(5.6, 5.6))
    obs = get_observer(observer)
    loc = _to(obs.spectral_locus(np.arange(380.0, 700.5, 0.5)), kind)
    poly = np.vstack([loc, loc[:1]])
    if fill:
        img, ext = _fill_image(kind, resolution, observer, poly, blur)
        ax.imshow(img, origin="lower", extent=ext, interpolation="bilinear", zorder=1)
    ax.plot(poly[:, 0], poly[:, 1], color=fg(), lw=locus_lw, zorder=4, solid_joinstyle="round")
    if wavelength_labels:
        _label_locus(ax, obs, kind)
    if planckian:
        T = np.geomspace(1000.0, 100_000.0, 300)
        pl = _to(planckian_xy(T, obs), kind)
        ax.plot(pl[:, 0], pl[:, 1], color=fg(), lw=0.9, zorder=5)
        iso = _to(isotherms(np.asarray(temperatures, dtype=float), duv=0.012, observer=obs), kind)
        for Ti, seg in zip(temperatures, iso):
            ax.plot(seg[:, 0], seg[:, 1], color=fg(), lw=0.7, zorder=5)
            ax.annotate(f"{Ti:,.0f} K", seg[0], xytext=(0, -2), textcoords="offset points", ha="center",
                        va="top", fontsize=6, color=fg(), zorder=6)
    if daylight:
        from ..temperature import daylight_locus

        dl = _to(daylight_locus(), kind)
        ax.plot(dl[:, 0], dl[:, 1], color=fg(), lw=0.9, ls=(0, (1.2, 1.6)), zorder=5, label="daylight locus")
    for i, name in enumerate(gamuts or ()):
        cs = get_colorspace(name)
        tri = _to(np.vstack([cs.primaries, cs.primaries[:1]]), kind)
        ax.plot(tri[:, 0], tri[:, 1], color=fg(), lw=1.0, ls=_DASHES[i % len(_DASHES)], zorder=5, label=cs.name)
    for wp in white_points or ():
        p = _to(np.asarray(WHITE_POINTS[wp]), kind)
        ax.plot(*p, marker="o", ms=3.5, mfc="none", mec=fg(), mew=0.9, zorder=7)
        ax.annotate(wp, p, xytext=(4, 3), textcoords="offset points", fontsize=7, color=fg(), zorder=7)
    (x0, x1), (y0, y1) = LIMITS[kind]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x" if kind == "xy" else "u′")
    ax.set_ylabel("y" if kind == "xy" else "v′")
    if grid:
        ax.grid(True, zorder=-1, alpha=0.6 if is_dark(ax) else 1.0)
    if gamuts or daylight:
        ax.legend(loc=legend_loc or ("upper right" if kind == "xy" else "lower right"), handlelength=2.6)
    return ax


def _label_locus(ax, obs, kind):
    white = _to(np.asarray(WHITE_POINTS["E"]), kind)
    for w in LABEL_WL[kind]:
        p0, p, p1 = _to(obs.spectral_locus(np.array([w - 0.5, w, w + 0.5])), kind)
        t = p1 - p0
        n = np.array([t[1], -t[0]])
        n /= np.linalg.norm(n)
        if np.dot(n, p - white) < 0:
            n = -n
        ax.plot([p[0], p[0] + 0.012 * n[0]], [p[1], p[1] + 0.012 * n[1]], color=fg(), lw=0.8, zorder=5)
        q = p + 0.022 * n
        ha = "left" if n[0] > 0.35 else ("right" if n[0] < -0.35 else "center")
        va = "bottom" if n[1] > 0.35 else ("top" if n[1] < -0.35 else "center")
        ax.text(q[0], q[1], f"{w}", fontsize=6.5, ha=ha, va=va, color=fg(), zorder=6)


def chromaticity_points(ax, values, labels=None, kind: str = "xy", observer="1931", size: float = 30,
                        marker: str = "o", annotate: bool = True, offsets=None, edgecolor=None, **kwargs):
    """Overlay colours on a chromaticity diagram.

    ``values``: a :class:`Spectrum` (batch allowed) / :class:`LineSpectrum`, XYZ (``(..., 3)``) or
    xy (``(..., 2)``). Points are filled with their own colour.
    """
    from ..colorimetry import color

    if isinstance(values, (Spectrum, LineSpectrum)):
        X = _xyz(values, observer=observer).reshape(-1, 3)
        xy = xyz_to_xy(X)
    else:
        v = np.asarray(values, dtype=float)
        if v.shape[-1] == 3:
            X = v.reshape(-1, 3)
            xy = xyz_to_xy(X)
        else:
            xy = v.reshape(-1, 2)
            X = xy_to_xyz(xy)
    face = color(X, brightness="max")
    p = _to(xy, kind)
    ax.scatter(p[:, 0], p[:, 1], s=size, c=face, marker=marker, edgecolors=edgecolor or fg(), linewidths=0.8,
               zorder=8, **kwargs)
    if annotate and labels is not None:
        labels = [labels] if isinstance(labels, str) else list(labels)
        offsets = offsets or [(6, 4)] * len(labels)
        for (x, y), lab, off in zip(p, labels, offsets):
            ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=fg(), zorder=9)
    return ax
