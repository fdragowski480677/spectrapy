"""Colour rendering of light sources: CIE 13.3-1995 colour rendering index (CRI)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._util import load_table
from .colorimetry import xyz, xyz_to_uv1960
from .constants import C2_CIE
from .illuminants import blackbody, daylight
from .spectrum import Spectrum
from .temperature import cct as _cct

TCS_NAMES = [f"TCS{i:02d}" for i in range(1, 15)]
#: Approximate Munsell notation of the CRI test colour samples.
TCS_MUNSELL = ["7.5 R 6/4", "5 Y 6/4", "5 GY 6/8", "2.5 G 6/6", "10 BG 6/4", "5 PB 6/8", "2.5 P 6/8",
               "10 P 6/8", "4.5 R 4/13", "5 Y 8/10", "4.5 G 5/8", "3 PB 3/11", "5 YR 8/4", "5 GY 4/4"]
TCS_DESCRIPTIONS = ["light greyish red", "dark greyish yellow", "strong yellow green",
                    "moderate yellowish green", "light bluish green", "light blue", "light violet",
                    "light reddish purple", "strong red", "strong yellow", "strong green", "strong blue",
                    "light yellowish pink (skin)", "moderate olive green (leaf)"]


def tcs_spectra() -> Spectrum:
    """The 14 CIE 13.3 test colour samples as a batch of reflectance spectra (5 nm)."""
    wl, cols = load_table("cri_tcs.csv")
    return Spectrum(wl, np.stack([cols[n] for n in TCS_NAMES]), "reflectance", name="CIE 13.3 TCS")


@dataclass
class CRIResult:
    """Outcome of :func:`cri`."""

    Ra: float                 #: general colour rendering index (mean of R1-R8)
    R: np.ndarray             #: special indices R1-R14
    cct: float                #: correlated colour temperature of the test source [K]
    duv: float                #: distance from the Planckian locus (CIE 1960 uv)
    reference: Spectrum       #: reference illuminant (Planckian below 5000 K, CIE daylight above)
    delta_e: np.ndarray       #: colour differences ΔE (CIE 1964 U*V*W*) of the 14 samples

    @property
    def valid(self) -> bool:
        """CIE 13.3 considers the index meaningful only for |Duv| < 5.4e-3."""
        return abs(self.duv) < 5.4e-3

    def __repr__(self) -> str:
        return f"CRIResult(Ra={self.Ra:.1f}, R9={self.R[8]:.1f}, CCT={self.cct:.0f} K, Duv={self.duv:+.4f})"


def _uvw(XYZ, uv_white):
    u, v = np.moveaxis(xyz_to_uv1960(XYZ), -1, 0)
    W = 25.0 * np.cbrt(XYZ[..., 1]) - 17.0
    return np.stack([13.0 * W * (u - uv_white[0]), 13.0 * W * (v - uv_white[1]), W], axis=-1)


def _cd(uv):
    u, v = uv[..., 0], uv[..., 1]
    return (4.0 - u - 10.0 * v) / v, (1.708 * v + 0.404 - 1.481 * u) / v


def cri(source: Spectrum) -> CRIResult:
    """CIE 13.3-1995 colour rendering index of an emission spectrum (CIE 1931 2° observer)."""
    if source.kind != "emission" or source.shape:
        raise ValueError("cri() expects a single emission spectrum")
    T, duv = _cct(xyz(source))
    T, duv = float(T), float(duv)
    wl = np.arange(360.0, 831.0)
    if T < 5000.0:
        ref = blackbody(T, wavelengths=wl, normalize="560", c2=C2_CIE)
    else:
        ref = daylight(T, wavelengths=wl, round_m=False)
    ref.name = f"reference {T:.0f} K"
    tcs = tcs_spectra()

    def tristimulus(illum):
        white = xyz(illum)
        k = 100.0 / white[1]
        return white * k, xyz(tcs, illuminant=illum) * 100.0

    w_t, s_t = tristimulus(source)
    w_r, s_r = tristimulus(ref)
    uv_wt, uv_wr = xyz_to_uv1960(w_t), xyz_to_uv1960(w_r)
    ct, dt = _cd(uv_wt)
    cr, dr = _cd(uv_wr)
    cti, dti = _cd(xyz_to_uv1960(s_t))
    denom = 16.518 + 1.481 * (cr / ct) * cti - (dr / dt) * dti
    u_a = (10.872 + 0.404 * (cr / ct) * cti - 4.0 * (dr / dt) * dti) / denom
    v_a = 5.520 / denom
    # adapted test-sample colours in U*V*W* relative to the reference white
    W_t = 25.0 * np.cbrt(s_t[:, 1]) - 17.0
    uvw_t = np.stack([13.0 * W_t * (u_a - uv_wr[0]), 13.0 * W_t * (v_a - uv_wr[1]), W_t], axis=-1)
    uvw_r = _uvw(s_r, uv_wr)
    dE = np.linalg.norm(uvw_t - uvw_r, axis=-1)
    R = 100.0 - 4.6 * dE
    return CRIResult(Ra=float(R[:8].mean()), R=R, cct=T, duv=duv, reference=ref, delta_e=dE)
