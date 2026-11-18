"""Turning spectra into arrays of display colours (no matplotlib needed).

These helpers produce the colours behind the plots - use them directly to build your own figures.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._util import as_float_array
from .colorspaces import get_colorspace, map_gamut
from .lines import binned_lines
from .observers import get_observer
from .spectrum import LineSpectrum, Spectrum


def _gaussian_blur(a, sigma_samples: float, axis: int = 0):
    if sigma_samples <= 0:
        return a
    r = max(1, int(np.ceil(4 * sigma_samples)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    a = np.moveaxis(a, axis, -1)
    pad = np.concatenate([np.repeat(a[..., :1], r, -1), a, np.repeat(a[..., -1:], r, -1)], -1)
    out = np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1, axis=-1) @ k
    return np.moveaxis(out, -1, axis)


@lru_cache(maxsize=16)
def _monochromatic_linear(space_name: str, observer_name: str, blur_nm: float, gamut: str, step: float = 0.25):
    """Linear RGB of equal-power monochromatic light on a fine grid (mapped into gamut, then
    blurred by a Gaussian 'spectrograph resolution' of ``blur_nm``)."""
    cs = get_colorspace(space_name)
    obs = get_observer(observer_name)
    g = np.arange(300.0, 900.0 + step / 2, step)
    rgb = map_gamut(cs.from_xyz(obs.at(g)), cs.luminance_weights, gamut, space=cs)
    rgb = _gaussian_blur(rgb, blur_nm / step, axis=0)
    return g, rgb, float(rgb.max())


def spectral_colors(wl, brightness: str = "full", blur_nm: float | None = None, space=None,
                    observer="1931", softness: float = 6.0, gamut: str = "oklch") -> np.ndarray:
    """Encoded RGB (shape ``wl.shape + (3,)``) of monochromatic light at wavelengths ``wl`` [nm].

    ``brightness="full"`` shows every wavelength as bright as possible (the classic rainbow);
    ``"luminance"`` keeps equal-power light at its true relative brightness (dim violet and red ends).

    Monochromatic light lies outside every display gamut. It is first brought into gamut (``gamut``:
    ``"oklch"`` reduces chroma at constant perceptual hue and lightness; ``"desaturate"`` mixes in
    white light of equal luminance), then blurred along the wavelength axis by ``blur_nm`` (default
    8 nm for ``"full"``, 6 nm for ``"luminance"``) - physically, a spectroscope of that resolution.
    The blur removes the bright/dark bands that per-colour gamut mapping produces at the display
    primaries; since the RGB cube is convex, blurred colours stay displayable. Full brightness uses a
    soft maximum (p-norm, ``softness``) instead of max() for the same reason.
    """
    wl = as_float_array(wl)
    cs = get_colorspace(space)
    obs = get_observer(observer)
    if blur_nm is None:
        blur_nm = 8.0 if brightness == "full" else 6.0
    obs_key = "1964" if obs.name == "CIE 1964 10°" else "1931"
    g, rgb_g, gmax = _monochromatic_linear(cs.name, obs_key, float(blur_nm), gamut)
    rgb = np.stack([np.interp(wl, g, rgb_g[:, c], left=0.0, right=0.0) for c in range(3)], axis=-1)
    if brightness == "full":
        norm = np.sum(np.maximum(rgb, 0.0) ** softness, axis=-1, keepdims=True) ** (1.0 / softness)
        with np.errstate(divide="ignore", invalid="ignore"):
            rgb = np.where(norm > 0, rgb / norm, 0.0)
    elif brightness == "luminance":
        rgb = rgb / gmax
    else:
        raise ValueError("brightness must be 'full' or 'luminance'")
    return np.clip(cs.encode(np.clip(rgb, 0.0, 1.0)), 0.0, 1.0)


def compress_luminance(xyz_, tone: float = 0.5) -> np.ndarray:
    """Compress the brightness range of an array of colours: ``Y -> Ymax (Y / Ymax)^tone`` with the
    chromaticity of every colour unchanged. A display aid (like a camera's tone curve) that keeps dim
    parts of a spectrum visible next to bright ones; ``tone=1`` returns the input unchanged."""
    X = np.asarray(xyz_, dtype=float)
    if tone == 1.0:
        return X
    Y = X[..., 1]
    ymax = np.max(Y) if np.max(Y) > 0 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(Y > 0, (np.clip(Y, 0, None) / ymax) ** (tone - 1.0), 0.0)
    return X * f[..., None]


def bin_edges(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, n + 1)


def binned_xyz(obj, edges, observer="1931", fwhm: float = 0.5, profile: str = "gaussian") -> np.ndarray:
    """XYZ of the light falling in each wavelength bin (``len(edges) - 1`` bins) - what one pixel
    column of a spectroscope image would record.

    ``obj`` is an emission :class:`Spectrum` (integrated exactly over each bin on a fine grid) or a
    :class:`LineSpectrum` (lines broadened to ``fwhm`` and integrated analytically, so even lines
    much narrower than a pixel keep their full power).
    """
    edges = as_float_array(edges)
    obs = get_observer(observer)
    centers = 0.5 * (edges[1:] + edges[:-1])
    cmf_c = obs.at(centers)
    if isinstance(obj, LineSpectrum):
        power = binned_lines(obj, edges, fwhm, profile)
        return power[:, None] * cmf_c
    if not isinstance(obj, Spectrum) or obj.kind != "emission":
        raise TypeError("binned_xyz expects an emission Spectrum or a LineSpectrum")
    # integrate S * cmf over each bin with a sub-grid of at most 0.05 nm
    sub = max(2, int(np.ceil((edges[1] - edges[0]) / 0.05)) + 1)
    t = np.linspace(0.0, 1.0, sub)
    w = np.full(sub, 1.0 / (sub - 1))
    w[0] = w[-1] = 0.5 / (sub - 1)
    g = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * t[None, :]  # (nbins, sub)
    vals = obj.at(g.ravel()).reshape(obj.shape + g.shape)
    cm = obs.at(g)  # (nbins, sub, 3)
    widths = (edges[1:] - edges[:-1])[:, None]
    return np.einsum("...bs,bsk,s,bz->...bk", vals, cm, w, widths)


def strip_colors(obj, lo: float = 380.0, hi: float = 750.0, n: int = 1200, space=None,
                 exposure: float = 1.0, highlights: str = "whiten", gamut: str = "oklch",
                 fwhm: float = 0.5, profile: str = "gaussian", flatten=None, tone: float = 1.0) -> np.ndarray:
    """Encoded RGB of ``n`` pixel columns spanning ``lo``-``hi`` nm for an emission spectrum or line
    list, with one exposure for the whole strip (the brightest column at 1 x ``exposure``).

    ``tone`` < 1 compresses the brightness range (column luminance ``Y -> Ymax (Y/Ymax)^tone``,
    chromaticity unchanged) - a display aid so that weak lines, e.g. violet ones the eye adapts to,
    stay visible next to bright ones. ``tone=1`` keeps true relative luminance.
    ``flatten``: an optional continuum :class:`Spectrum`; columns are then shown relative to it
    (e.g. a solar spectrum as a uniform rainbow crossed by dark Fraunhofer lines).
    """
    edges = bin_edges(lo, hi, n)
    X = binned_xyz(obj, edges, fwhm=fwhm, profile=profile)
    if flatten is None:
        X = compress_luminance(X, tone)
    if flatten is not None:
        Xc = binned_xyz(flatten, edges)
        ratio = X[..., 1] / np.where(Xc[..., 1] > 0, Xc[..., 1], 1.0)
        centers = 0.5 * (edges[1:] + edges[:-1])
        rgb = spectral_colors(centers, brightness="full", space=space)
        lin = get_colorspace(space).decode(rgb) * np.clip(ratio, 0.0, None)[..., None] * exposure
        return np.clip(get_colorspace(space).encode(np.clip(lin, 0, 1)), 0.0, 1.0)
    return get_colorspace(space).to_display(X, brightness="global", exposure=exposure,
                                            highlights=highlights, gamut=gamut)


def spectrum_colors(spectra: Spectrum, **kwargs) -> np.ndarray:
    """Colour of each spectrum in a batch (alias of :func:`spectrapy.color`)."""
    from .colorimetry import color

    return color(spectra, **kwargs)


__all__ = ["bin_edges", "binned_xyz", "compress_luminance", "spectral_colors", "spectrum_colors",
           "strip_colors"]
