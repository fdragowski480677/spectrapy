"""Colour rendering of light sources: the CIE 13.3-1995 colour rendering index (CRI) and the
ANSI/IES TM-30 method (fidelity Rf - identical to CIE 224:2017 - gamut Rg and hue-bin metrics)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._util import load_table, trapezoid_weights
from .appearance import jmh_to_ucs, xyz_to_ciecam02
from .colorimetry import xyz, xyz_to_uv1960
from .constants import C2_CIE
from .illuminants import blackbody, daylight, daylight_from_xy, daylight_xy, planck
from .observers import CIE1931_2, CIE1964_10
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


# ---------------------------------------------------------------------------------------- TM-30
CES_NAMES = [f"CES{i:02d}" for i in range(1, 100)]


def ces_spectra() -> Spectrum:
    """The 99 colour evaluation samples of CIE 224:2017 / ANSI/IES TM-30 (reflectance, 1 nm, 380-780 nm)."""
    wl, cols = load_table("tm30_ces.csv.gz")
    return Spectrum(wl, np.stack([cols[n] for n in CES_NAMES]), "reflectance", name="CIE 224 / TM-30 CES")


@dataclass
class TM30Result:
    """Outcome of :func:`tm30`."""

    Rf: float                 #: fidelity index (0-100; = CIE 224:2017 Rf)
    Rg: float                 #: gamut index (100 = same gamut area as the reference)
    cct: float                #: correlated colour temperature of the test source [K]
    duv: float                #: distance from the Planckian locus
    Rf_hues: np.ndarray       #: local fidelity Rf,hj of the 16 hue bins
    Rcs: np.ndarray           #: local chroma shift Rcs,hj of the 16 hue bins [%]
    Rhs: np.ndarray           #: local hue shift Rhs,hj of the 16 hue bins [rad]
    Rf_samples: np.ndarray    #: fidelity of each of the 99 samples
    reference: Spectrum       #: reference illuminant (CIE 224:2017 blend of Planckian and daylight)
    jab_test: np.ndarray      #: CAM02-UCS (J', a', b') of the samples under the test source
    jab_ref: np.ndarray       #: CAM02-UCS (J', a', b') under the reference
    bins: np.ndarray          #: hue bin (0-15) of each sample, from its hue under the reference
    bin_ab_test: np.ndarray   #: mean (a', b') per hue bin, test source
    bin_ab_ref: np.ndarray    #: mean (a', b') per hue bin, reference

    def __repr__(self) -> str:
        return f"TM30Result(Rf={self.Rf:.1f}, Rg={self.Rg:.1f}, CCT={self.cct:.0f} K, Duv={self.duv:+.4f})"


def _rf(delta_e):
    return 10.0 * np.log1p(np.exp((100.0 - 6.73 * np.asarray(delta_e)) / 10.0))


def cie224_reference(T: float, wavelengths) -> np.ndarray:
    """CIE 224:2017 reference illuminant at CCT ``T`` on ``wavelengths``: Planckian below 4000 K,
    CIE daylight above 5000 K and a luminance-normalised linear blend in between."""
    wl = np.asarray(wavelengths, dtype=float)
    w = trapezoid_weights(wl)
    ybar = CIE1931_2.at(wl)[:, 1] * w
    P = planck(wl, T, c2=C2_CIE) if T <= 5000 else None
    D = daylight_from_xy(daylight_xy(T), wavelengths=wl).values if T >= 4000 else None
    if T < 4000:
        return P
    if T > 5000:
        return D
    m = (T - 4000.0) / 1000.0
    return (1.0 - m) * P / (P @ ybar) + m * D / (D @ ybar)


def _polygon_area(p) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def tm30(source: Spectrum) -> TM30Result:
    """ANSI/IES TM-30 colour rendition of an emission spectrum.

    Follows CIE 224:2017 for the fidelity index Rf (99 samples, CIE 1964 10° observer, CIECAM02 with
    complete adaptation, L_A = 100 cd/m², Y_b = 20, CAM02-UCS colour differences) and TM-30 for the
    gamut index Rg and the 16 hue-bin metrics. Spectra sampled at 5 nm are evaluated at 5 nm,
    anything else at 1 nm over 380-780 nm (missing values are zero, as the standard specifies).
    """
    if source.kind != "emission" or source.shape:
        raise ValueError("tm30() expects a single emission spectrum")
    wl_s = source.wavelengths
    step = 5.0 if wl_s.size > 1 and np.allclose(np.diff(wl_s), 5.0) else 1.0
    g = np.arange(380.0, 780.0 + step / 2.0, step)
    S = source.at(g)
    w = trapezoid_weights(g)
    T, duv = _cct(S @ (CIE1931_2.at(g) * w[:, None]))
    T, duv = float(T), float(duv)
    ref = cie224_reference(T, g)
    cmf10 = CIE1964_10.at(g) * w[:, None]
    ces = ces_spectra().resample(g).values  # exact subsample of the 1 nm table on a 5 nm grid

    def colorimetry(spd):
        white = spd @ cmf10
        k = 100.0 / white[1]
        XYZ = (ces * spd) @ cmf10 * k
        cam = xyz_to_ciecam02(XYZ, white * k, L_A=100.0, Y_b=20.0, surround="average", discount_illuminant=True)
        return jmh_to_ucs(cam.JMh), cam.h

    jab_t, _ = colorimetry(S)
    jab_r, h_r = colorimetry(ref)
    dE = np.linalg.norm(jab_t - jab_r, axis=-1)
    bins = np.minimum(np.floor(h_r / 22.5).astype(int), 15)
    onehot = bins[None, :] == np.arange(16)[:, None]  # (16, 99)
    counts = onehot.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ab_t = (onehot @ jab_t[:, 1:]) / counts[:, None]
        ab_r = (onehot @ jab_r[:, 1:]) / counts[:, None]
        dE_bins = (onehot @ dE) / counts
    theta = np.radians(22.5 * np.arange(16) + 11.25)
    d = ab_t - ab_r
    norm = np.linalg.norm(ab_r, axis=1)
    Rcs = 100.0 * (d[:, 0] * np.cos(theta) + d[:, 1] * np.sin(theta)) / norm
    Rhs = (-d[:, 0] * np.sin(theta) + d[:, 1] * np.cos(theta)) / norm
    return TM30Result(Rf=float(_rf(dE.mean())), Rg=100.0 * _polygon_area(ab_t) / _polygon_area(ab_r), cct=T,
                      duv=duv, Rf_hues=_rf(dE_bins), Rcs=Rcs, Rhs=Rhs, Rf_samples=_rf(dE),
                      reference=Spectrum(g, ref, "emission", name=f"CIE 224 reference {T:.0f} K"),
                      jab_test=jab_t, jab_ref=jab_r, bins=bins, bin_ab_test=ab_t, bin_ab_ref=ab_r)
