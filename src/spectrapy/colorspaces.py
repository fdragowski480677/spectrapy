"""RGB colour spaces and the XYZ -> displayable RGB pipeline (exposure, gamut mapping, encoding).

Real spectra are often outside every display gamut (monochromatic light always is). How such
colours are brought into gamut matters for how plots look, so the pipeline is explicit:

1. ``XYZ -> linear RGB`` with the space's matrix (optionally after chromatic adaptation).
2. **gamut** (negative components):
   ``"oklch"`` (default) reduces chroma at constant OKLab lightness and hue - the perceptual
   approach also used by CSS Color 4 - so out-of-gamut colours keep their perceived hue.
   ``"desaturate"`` mixes each colour with white of the *same luminance* until it fits - the
   colorimetric equivalent of adding white light (keeps dominant wavelength and luminance).
   ``"uniform"`` uses one common amount of desaturation for all colours of the call (smooth,
   but paler). ``"smooth"`` lets the desaturation vary along the last-but-one axis without kinks,
   removing the Mach bands ``"desaturate"`` shows at the primaries in spectrum strips.
   ``"clip"`` simply clamps negative components (shifts hue).
3. **brightness**: ``None`` keeps the absolute scale (``Y = 1`` is display white), ``"max"`` makes
   every colour as bright as possible, ``"global"`` scales all colours together so the brightest
   one just fits, a number sets the luminance ``Y`` of every colour.
4. **highlights** (components above 1 after ``exposure``): ``"scale"`` darkens the colour keeping its
   chromaticity, ``"whiten"`` desaturates towards white keeping luminance (bright cores of
   over-exposed lines turn white, as on a camera sensor), ``"clip"`` clamps.
5. The transfer function (e.g. the sRGB curve) encodes the result for display.
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array, lower_envelope_smooth
from .adaptation import adapt, white_to_xyz


# --------------------------------------------------------------- transfer functions
def srgb_encode(x):
    x = as_float_array(x)
    a = np.abs(x)
    return np.copysign(np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055), x)


def srgb_decode(v):
    v = as_float_array(v)
    a = np.abs(v)
    return np.copysign(np.where(a <= 0.04045, a / 12.92, np.power((a + 0.055) / 1.055, 2.4)), v)


def _gamma_pair(g: float):
    def enc(x):
        x = as_float_array(x)
        return np.copysign(np.power(np.abs(x), 1.0 / g), x)

    def dec(v):
        v = as_float_array(v)
        return np.copysign(np.power(np.abs(v), g), v)

    return enc, dec


_A, _B = 1.09929682680944, 0.018053968510807  # ITU-R BT.2020 / BT.709 OETF constants


def bt2020_encode(x):
    x = as_float_array(x)
    a = np.abs(x)
    return np.copysign(np.where(a < _B, 4.5 * a, _A * np.power(a, 0.45) - (_A - 1.0)), x)


def bt2020_decode(v):
    v = as_float_array(v)
    a = np.abs(v)
    return np.copysign(np.where(a < 4.5 * _B, a / 4.5, np.power((a + _A - 1.0) / _A, 1.0 / 0.45)), v)


def _identity(x):
    return as_float_array(x)


def rgb_to_xyz_matrix(primaries, white) -> np.ndarray:
    """Normalised primary matrix from primaries ``[[xr, yr], [xg, yg], [xb, yb]]`` and white xy."""
    p = as_float_array(primaries)
    P = np.stack([p[:, 0] / p[:, 1], np.ones(3), (1.0 - p[:, 0] - p[:, 1]) / p[:, 1]])
    S = np.linalg.solve(P, white_to_xyz(white))
    return P * S


class RGBColorSpace:
    """An additive RGB colour space defined by primaries, white point and transfer function."""

    def __init__(self, name, primaries, white, encode, decode, aliases=()):
        self.name = name
        self.primaries = as_float_array(primaries)
        self.white = as_float_array(white)
        self.white_xyz = white_to_xyz(self.white)
        self.encode = encode
        self.decode = decode
        self.aliases = tuple(aliases)
        self.matrix_to_xyz = rgb_to_xyz_matrix(self.primaries, self.white)
        self.matrix_from_xyz = np.linalg.inv(self.matrix_to_xyz)
        self.luminance_weights = self.matrix_to_xyz[1].copy()

    def __repr__(self) -> str:
        return f"RGBColorSpace({self.name!r})"

    # ----------------------------------------------------------------- linear
    def from_xyz(self, xyz, white=None, method: str = "bradford") -> np.ndarray:
        """Linear RGB from XYZ. If ``white`` (the XYZ's adopted white) differs from the space's white,
        the colours are chromatically adapted first."""
        xyz = as_float_array(xyz)
        if white is not None:
            xyz = adapt(xyz, white, self.white_xyz, method)
        return xyz @ self.matrix_from_xyz.T

    def to_xyz(self, rgb) -> np.ndarray:
        return as_float_array(rgb) @ self.matrix_to_xyz.T

    def in_gamut(self, rgb, tol: float = 1e-9) -> np.ndarray:
        rgb = as_float_array(rgb)
        return np.all((rgb >= -tol) & (rgb <= 1.0 + tol), axis=-1)

    # --------------------------------------------------------------- display
    def to_display(self, xyz, *, brightness=None, gamut: str = "oklch", highlights: str = "scale",
                   exposure: float = 1.0, smooth_radius: int = 12, white=None, encode: bool = True):
        """XYZ (shape ``(..., 3)``) -> RGB in ``[0, 1]`` ready for display. See the module docs."""
        rgb = self.from_xyz(xyz, white=white)
        rgb = map_gamut(rgb, self.luminance_weights, gamut, smooth_radius, space=self)
        if brightness is not None:
            if isinstance(brightness, str):
                m = rgb.max(-1, keepdims=True)
                if brightness == "global":
                    m = np.max(m)
                elif brightness != "max":
                    raise ValueError(f"unknown brightness {brightness!r}")
            else:
                m = (rgb @ self.luminance_weights)[..., None] / float(brightness)
            with np.errstate(divide="ignore", invalid="ignore"):
                rgb = np.where(m > 0, rgb / m, 0.0)
        if exposure != 1.0:
            rgb = rgb * exposure
        rgb = map_highlights(rgb, self.luminance_weights, highlights)
        rgb = np.clip(rgb, 0.0, 1.0)
        return np.clip(self.encode(rgb), 0.0, 1.0) if encode else rgb

    def to_hex(self, xyz, **kwargs):
        """``'#rrggbb'`` string(s) for XYZ colour(s); keyword arguments go to :meth:`to_display`."""
        return rgb_to_hex(self.to_display(xyz, **kwargs))


def _oklch_map(rgb, space, iterations: int = 22) -> np.ndarray:
    """Reduce OKLab chroma at constant OKLab lightness and hue until all components are >= 0.
    OKLab hue is invariant to scaling, so this composes with any brightness normalisation."""
    from .uniform import oklab_to_xyz, xyz_to_oklab

    lab = xyz_to_oklab(space.to_xyz(rgb))
    L, ab = lab[..., :1], lab[..., 1:]
    neg = np.any(rgb < 0.0, axis=-1)
    lo = np.zeros(rgb.shape[:-1])
    hi = np.ones(rgb.shape[:-1])
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        trial = space.from_xyz(oklab_to_xyz(np.concatenate([L, ab * mid[..., None]], axis=-1)))
        ok = np.all(trial >= -1e-12, axis=-1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    s = np.where(neg, lo, 1.0)
    out = space.from_xyz(oklab_to_xyz(np.concatenate([L, ab * s[..., None]], axis=-1)))
    return np.maximum(out, 0.0)


def map_gamut(rgb, weights, method: str = "desaturate", smooth_radius: int = 12, space=None) -> np.ndarray:
    """Bring negative linear-RGB components to zero; see the module docs for ``method``.
    ``"oklch"`` (perceptual hue preserving chroma reduction) needs the colour ``space``."""
    rgb = as_float_array(rgb)
    if method in (None, "none"):
        return rgb
    if method == "clip":
        return np.maximum(rgb, 0.0)
    if method == "oklch":
        if space is None:
            raise ValueError("gamut='oklch' needs the colour space")
        return _oklch_map(rgb, space)
    Y = (rgb @ weights)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = np.where(rgb < 0.0, Y / (Y - rgb), 1.0)
    t = np.clip(np.nan_to_num(tc.min(-1), nan=0.0), 0.0, 1.0)
    t = np.where(Y[..., 0] > 0, t, 0.0)
    if method == "uniform":
        valid = Y[..., 0] > 0
        t = np.full_like(t, t[valid].min() if np.any(valid) else 1.0)
    elif method == "smooth":
        if t.ndim == 0:
            pass
        else:
            t = lower_envelope_smooth(t, int(smooth_radius), axis=-1)
    elif method != "desaturate":
        raise ValueError(f"unknown gamut mapping {method!r}")
    out = Y + t[..., None] * (rgb - Y)
    return np.maximum(out, 0.0)


def map_highlights(rgb, weights, method: str = "scale") -> np.ndarray:
    """Handle components above 1; see the module docs for ``method``."""
    rgb = as_float_array(rgb)
    m = rgb.max(-1, keepdims=True)
    if method == "clip":
        return np.minimum(rgb, 1.0)
    if method == "scale":
        return np.where(m > 1.0, rgb / np.where(m > 0, m, 1.0), rgb)
    if method == "whiten":
        Y = (rgb @ weights)[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(m > 1.0, np.clip((1.0 - Y) / (m - Y), 0.0, 1.0), 1.0)
        return np.minimum(Y + np.nan_to_num(t) * (rgb - Y), 1.0)
    raise ValueError(f"unknown highlight handling {method!r}")


def rgb_to_hex(rgb):
    """Encoded RGB in [0, 1] (shape ``(..., 3)``) -> ``'#rrggbb'`` (str or nested list)."""
    rgb = np.clip(np.round(as_float_array(rgb) * 255.0), 0, 255).astype(int)
    flat = ["#{:02x}{:02x}{:02x}".format(*c) for c in rgb.reshape(-1, 3)]
    if rgb.ndim == 1:
        return flat[0]
    return np.array(flat, dtype=object).reshape(rgb.shape[:-1]).tolist()


_D65 = (0.3127, 0.3290)
SRGB =RGBColorSpace("sRGB", [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], _D65,
                     srgb_encode, srgb_decode, aliases=("srgb", "rec709", "bt709"))
LINEAR_SRGB = RGBColorSpace("linear sRGB", SRGB.primaries, _D65, _identity, _identity,
                            aliases=("linear", "linear_srgb", "linear srgb"))
DISPLAY_P3 = RGBColorSpace("Display P3", [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]], _D65,
                           srgb_encode, srgb_decode, aliases=("p3", "display-p3", "display_p3", "displayp3"))
REC2020 = RGBColorSpace("Rec. 2020", [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], _D65,
                        bt2020_encode, bt2020_decode, aliases=("rec2020", "bt2020", "rec.2020"))
_adobe_enc, _adobe_dec = _gamma_pair(563.0 / 256.0)
ADOBE_RGB = RGBColorSpace("Adobe RGB (1998)", [[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]], _D65,
                          _adobe_enc, _adobe_dec, aliases=("adobe", "adobergb", "adobe rgb", "adobe_rgb"))

COLORSPACES = {cs.name: cs for cs in (SRGB, LINEAR_SRGB, DISPLAY_P3, REC2020, ADOBE_RGB)}


def get_colorspace(space="srgb") -> RGBColorSpace:
    """Look up a colour space by name or alias (``"srgb"``, ``"p3"``, ``"rec2020"``, ``"adobe"``...)."""
    if isinstance(space, RGBColorSpace):
        return space
    key = str(space).strip().lower()
    for cs in COLORSPACES.values():
        if key == cs.name.lower() or key in cs.aliases:
            return cs
    raise ValueError(f"unknown colour space {space!r}; available: {', '.join(COLORSPACES)}")
