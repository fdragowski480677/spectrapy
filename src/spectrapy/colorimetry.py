"""From spectra to CIE XYZ and chromaticity coordinates (vectorized over batches of spectra)."""
from __future__ import annotations

import numpy as np

from ._util import as_float_array, trapezoid_weights, union_grid
from .colorspaces import get_colorspace, rgb_to_hex
from .constants import K_M
from .illuminants import resolve_illuminant
from .observers import get_observer
from .spectrum import LineSpectrum, Spectrum


def _weighted_cmf(grid, obs) -> np.ndarray:
    return obs.at(grid) * trapezoid_weights(grid)[:, None]


def xyz(obj, observer="1931", illuminant=None, k: float | None = None) -> np.ndarray:
    """CIE XYZ tristimulus values, shape ``batch + (3,)``.

    * **emission** spectra (and :class:`LineSpectrum`): ``XYZ = k ∫ S(λ) cmf(λ) dλ`` with
      ``k = K_m = 683.002 lm/W`` by default, so ``Y`` is luminance [cd/m²] when ``S`` is spectral
      radiance [W sr⁻¹ m⁻² nm⁻¹] (illuminance [lx] for irradiance). The integral is evaluated with
      the trapezoid rule on the union of the spectrum's and the observer's grids, so irregular
      or coarse sampling gives the right answer (a plain sum would not).
    * **reflectance / transmittance**: relative values under ``illuminant`` (default D65),
      normalised so that a perfect white has ``Y = 1``.
    * Lines are exact delta functions: ``XYZ = k Σ Pᵢ cmf(λᵢ)``.
    """
    obs = get_observer(observer)
    if isinstance(obj, LineSpectrum):
        return (K_M if k is None else k) * (obj.intensities @ obs.at(obj.wavelengths))
    if not isinstance(obj, Spectrum):
        raise TypeError("xyz() expects a Spectrum or LineSpectrum")
    owl = obs.wavelengths
    if obj.kind == "emission":
        if illuminant is not None:
            raise ValueError("an illuminant only applies to reflectance/transmittance spectra")
        lo, hi = max(obj.wavelengths[0], owl[0]), min(obj.wavelengths[-1], owl[-1])
        if hi <= lo:
            return np.zeros(obj.shape + (3,))
        if np.array_equal(obj.wavelengths, owl):
            g, vals = owl, obj.values
        else:
            g = union_grid(obj.wavelengths, owl, lo=lo, hi=hi)
            vals = obj.at(g)
        return (K_M if k is None else k) * (vals @ _weighted_cmf(g, obs))
    illum = resolve_illuminant("D65" if illuminant is None else illuminant)
    if illum.shape:
        raise ValueError("use a single illuminant spectrum")
    lo, hi = max(illum.wavelengths[0], owl[0]), min(illum.wavelengths[-1], owl[-1])
    g = union_grid(obj.wavelengths, illum.wavelengths, owl, lo=lo, hi=hi)
    W = _weighted_cmf(g, obs) * illum.at(g)[:, None]
    scale = 1.0 / W[:, 1].sum() if k is None else k
    return (obj.at(g) @ W) * scale


def illuminant_white(illuminant="D65", observer="1931") -> np.ndarray:
    """XYZ of a perfect white under ``illuminant`` (``Y = 1``)."""
    w = xyz(resolve_illuminant(illuminant), observer)
    return w / w[1]


# ----------------------------------------------------------- chromaticity coordinates
def xyz_to_xy(xyz_) -> np.ndarray:
    xyz_ = as_float_array(xyz_)
    s = xyz_.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return xyz_[..., :2] / s


def xy_to_xyz(xy, Y=1.0) -> np.ndarray:
    xy = as_float_array(xy)
    x, y = xy[..., 0], xy[..., 1]
    Y = np.broadcast_to(as_float_array(Y), x.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([x * Y / y, Y, (1.0 - x - y) * Y / y], axis=-1)


def xyz_to_uv1960(xyz_) -> np.ndarray:
    X, Y, Z = np.moveaxis(as_float_array(xyz_), -1, 0)
    d = X + 15.0 * Y + 3.0 * Z
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([4.0 * X / d, 6.0 * Y / d], axis=-1)


def xyz_to_uv1976(xyz_) -> np.ndarray:
    """CIE 1976 UCS chromaticity (u', v')."""
    X, Y, Z = np.moveaxis(as_float_array(xyz_), -1, 0)
    d = X + 15.0 * Y + 3.0 * Z
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([4.0 * X / d, 9.0 * Y / d], axis=-1)


def xy_to_uv1960(xy) -> np.ndarray:
    x, y = np.moveaxis(as_float_array(xy), -1, 0)
    d = -2.0 * x + 12.0 * y + 3.0
    return np.stack([4.0 * x / d, 6.0 * y / d], axis=-1)


def uv1960_to_xy(uv) -> np.ndarray:
    u, v = np.moveaxis(as_float_array(uv), -1, 0)
    d = 2.0 * u - 8.0 * v + 4.0
    return np.stack([3.0 * u / d, 2.0 * v / d], axis=-1)


def xy_to_uv1976(xy) -> np.ndarray:
    x, y = np.moveaxis(as_float_array(xy), -1, 0)
    d = -2.0 * x + 12.0 * y + 3.0
    return np.stack([4.0 * x / d, 9.0 * y / d], axis=-1)


def uv1976_to_xy(uvp) -> np.ndarray:
    u, v = np.moveaxis(as_float_array(uvp), -1, 0)
    d = 6.0 * u - 16.0 * v + 12.0
    return np.stack([9.0 * u / d, 4.0 * v / d], axis=-1)


def dominant_wavelength(xy, white="D65", observer="1931"):
    """Dominant wavelength [nm] and excitation purity of chromaticities ``xy``.

    For colours in the purple region the *complementary* wavelength is returned as a negative
    number (CIE convention). Returns ``(wavelength, purity)`` arrays.
    """
    from .adaptation import white_to_xyz

    obs = get_observer(observer)
    wl = np.arange(380.0, 700.05, 0.1)
    loc = obs.spectral_locus(wl)
    w = xyz_to_xy(white_to_xyz(white))
    ang = np.unwrap(np.arctan2(loc[:, 1] - w[1], loc[:, 0] - w[0]))  # decreasing with wavelength
    xy = as_float_array(xy)
    d = xy - w
    a = np.arctan2(d[..., 1], d[..., 0])

    def lookup(theta):
        theta = (theta - ang[-1]) % (2 * np.pi) + ang[-1]  # bring into [ang[-1], ang[-1] + 2 pi)
        inside = theta <= ang[0]
        lam = np.interp(-theta, -ang, wl)
        return lam, inside

    lam, inside = lookup(a)
    lam_c, _ = lookup(a + np.pi)
    # boundary point: spectral locus for dominant, purple line for complementary
    b_loc = obs.spectral_locus(lam)
    p0, p1 = loc[0], loc[-1]
    e = p1 - p0
    denom = d[..., 0] * e[1] - d[..., 1] * e[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p0[0] - w[0]) * e[1] - (p0[1] - w[1]) * e[0]) / denom
    b_purple = w + t[..., None] * d
    b = np.where(inside[..., None], b_loc, b_purple)
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.linalg.norm(d, axis=-1) / np.linalg.norm(b - w, axis=-1)
    return np.where(inside, lam, -lam_c), purity


# ------------------------------------------------------------------ display shortcuts
def color(obj, space=None, observer="1931", illuminant=None, brightness="auto",
          gamut: str = "oklch", highlights: str = "scale", exposure: float = 1.0,
          adapt: bool = False) -> np.ndarray:
    """Displayable, encoded RGB in ``[0, 1]`` (shape ``batch + (3,)``).

    ``obj`` is a :class:`Spectrum`, a :class:`LineSpectrum` or XYZ values (relative, white ``Y = 1``).
    ``brightness="auto"`` shows light sources as bright as possible (``"max"``) and keeps the
    relative lightness of reflecting/transmitting objects and XYZ input (``None``).
    With ``adapt=True`` object colours under a non-D65 ``illuminant`` are shown as a viewer adapted
    to that illuminant would see them (Bradford transform); otherwise they look as in a photo
    taken with daylight white balance.
    """
    cs = get_colorspace(space)
    white = None
    if isinstance(obj, (Spectrum, LineSpectrum)):
        X = xyz(obj, observer, illuminant if isinstance(obj, Spectrum) and obj.kind != "emission" else None)
        emissive = isinstance(obj, LineSpectrum) or obj.kind == "emission"
        if adapt and not emissive and illuminant is not None:
            white = illuminant_white(illuminant, observer)
    else:
        X = as_float_array(obj)
        emissive = False
    if brightness == "auto":
        brightness = "max" if emissive else None
    return cs.to_display(X, brightness=brightness, gamut=gamut, highlights=highlights,
                         exposure=exposure, white=white)


def hex_color(obj, **kwargs):
    """Like :func:`color` but returns ``'#rrggbb'`` string(s)."""
    return rgb_to_hex(color(obj, **kwargs))
