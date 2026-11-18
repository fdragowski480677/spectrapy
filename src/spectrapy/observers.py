"""CIE standard colorimetric observers (colour-matching functions)."""
from __future__ import annotations

import numpy as np

from ._util import as_float_array, interp, load_table


class Observer:
    """A set of colour-matching functions x̄(λ), ȳ(λ), z̄(λ) sampled at 1 nm."""

    __slots__ = ("name", "wavelengths", "cmf")

    def __init__(self, name: str, wavelengths, cmf):
        self.name = name
        self.wavelengths = as_float_array(wavelengths)
        self.cmf = as_float_array(cmf)
        self.cmf.setflags(write=False)

    def __repr__(self) -> str:
        return f"Observer({self.name!r})"

    def at(self, wl) -> np.ndarray:
        """Colour-matching functions at wavelengths ``wl`` (nm), shape ``wl.shape + (3,)``.
        Zero outside the tabulated range."""
        wl = as_float_array(wl)
        return np.moveaxis(interp(wl, self.wavelengths, self.cmf.T, 0.0, 0.0), 0, -1)

    @property
    def xbar(self):
        return self.cmf[:, 0]

    @property
    def ybar(self):
        return self.cmf[:, 1]

    @property
    def zbar(self):
        return self.cmf[:, 2]

    def spectral_locus(self, wl=None) -> np.ndarray:
        """Chromaticities ``(x, y)`` of monochromatic light (default 380-780 nm, 1 nm)."""
        wl = np.arange(380.0, 781.0) if wl is None else as_float_array(wl)
        c = self.at(wl)
        return c[..., :2] / c.sum(-1, keepdims=True)


def _load(fname: str, name: str) -> Observer:
    wl, cols = load_table(fname)
    return Observer(name, wl, np.column_stack([cols["xbar"], cols["ybar"], cols["zbar"]]))


CIE1931_2 = _load("cmf_cie1931_2.csv", "CIE 1931 2°")
CIE1964_10 = _load("cmf_cie1964_10.csv", "CIE 1964 10°")

_ALIASES = {
    "1931": CIE1931_2, "2": CIE1931_2, "2deg": CIE1931_2, "cie1931": CIE1931_2, "cie 1931 2°": CIE1931_2,
    "1964": CIE1964_10, "10": CIE1964_10, "10deg": CIE1964_10, "cie1964": CIE1964_10,
    "cie 1964 10°": CIE1964_10,
}


def get_observer(observer="1931") -> Observer:
    """Return an :class:`Observer` from a name (``"1931"``/``"2"``, ``"1964"``/``"10"``) or pass one through."""
    if isinstance(observer, Observer):
        return observer
    try:
        return _ALIASES[str(observer).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown observer {observer!r}; use '1931' or '1964'") from None
