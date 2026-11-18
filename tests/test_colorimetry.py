"""Colorimetry: data integrity and XYZ integration, validated against colour-science."""
import numpy as np
import pytest

import spectrapy as sp
from spectrapy import LineSpectrum, Spectrum


def test_cmf_tables_match_colour(colour):
    for ours, key in ((sp.CIE1931_2, "CIE 1931 2 Degree Standard Observer"),
                      (sp.CIE1964_10, "CIE 1964 10 Degree Standard Observer")):
        ref = colour.MSDS_CMFS[key]
        assert np.array_equal(ours.wavelengths, ref.wavelengths)
        np.testing.assert_allclose(ours.cmf, ref.values, atol=1e-9)


@pytest.mark.parametrize("name", ["D65", "D50", "FL2", "FL11", "LED-B3", "LED-RGB1"])
def test_illuminant_xyz_matches_colour(colour, name):
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ref = colour.sd_to_XYZ(colour.SDS_ILLUMINANTS[name], cmfs, method="Integration")
    ours = sp.xyz(sp.illuminant(name))
    np.testing.assert_allclose(ours / ours[1], ref / ref[1], rtol=3e-4)


def test_sampling_invariance_of_xyz():
    """The integral, not a plain sum: 1 nm, 5 nm and irregular sampling give the same colour."""
    fine = sp.blackbody(4000, wavelengths=sp.wavelengths(360, 830, 1))
    coarse = sp.blackbody(4000, wavelengths=sp.wavelengths(360, 830, 5))
    irregular = sp.blackbody(4000, wavelengths=np.sort(np.r_[360, 830, np.random.default_rng(0).uniform(360, 830,
                                                                                                         300)]))
    X = sp.xyz(fine)
    np.testing.assert_allclose(sp.xyz(coarse), X, rtol=2e-3)
    np.testing.assert_allclose(sp.xyz(irregular), X, rtol=5e-3)


def test_luminance_of_monochromatic_watt():
    # 1 W/sr/m² at 555 nm (as a 1 nm wide box) -> ~683 cd/m²
    s = Spectrum([554.5, 554.5001, 555.4999, 555.5], [0.0, 1.0, 1.0, 0.0])
    assert sp.xyz(s)[1] == pytest.approx(683.0, rel=2e-3)
    assert sp.xyz(LineSpectrum([555.0], [1.0]))[1] == pytest.approx(683.0, rel=1e-3)


def test_lines_are_delta_functions():
    lines = sp.line_spectrum("Na")
    narrow = lines.broaden(fwhm=0.02, wavelengths=np.arange(480, 630, 0.001))
    np.testing.assert_allclose(sp.xyz(narrow), sp.xyz(lines), rtol=1e-3)


def test_reflectance_white_and_batch():
    wl = sp.wavelengths(380, 780, 5)
    white = Spectrum(wl, np.ones(wl.size), "reflectance")
    X = sp.xyz(white, illuminant="D65")
    assert X[1] == pytest.approx(1.0)
    np.testing.assert_allclose(sp.xyz_to_xy(X), sp.WHITE_POINTS["D65"], atol=2e-4)
    batch = Spectrum(wl, np.stack([0.2 * np.ones(wl.size), 0.8 * np.ones(wl.size)]), "reflectance")
    assert np.allclose(sp.xyz(batch)[:, 1], [0.2, 0.8])


def test_batch_equals_individual():
    T = np.array([1500.0, 3000.0, 6500.0, 20000.0])
    batch = sp.xyz(sp.blackbody(T))
    single = np.array([sp.xyz(sp.blackbody(t)) for t in T])
    np.testing.assert_allclose(batch, single, rtol=1e-12)


def test_chromaticity_conversions_roundtrip(rng):
    xy = rng.uniform([0.2, 0.2], [0.5, 0.5], (50, 2))
    np.testing.assert_allclose(sp.uv1960_to_xy(sp.xy_to_uv1960(xy)), xy, atol=1e-12)
    np.testing.assert_allclose(sp.uv1976_to_xy(sp.xy_to_uv1976(xy)), xy, atol=1e-12)
    np.testing.assert_allclose(sp.xyz_to_xy(sp.xy_to_xyz(xy, 0.4)), xy, atol=1e-12)


def test_dominant_wavelength_of_monochromatic_light():
    wl = np.array([450.0, 500.0, 550.0, 600.0, 650.0])
    lam, purity = sp.dominant_wavelength(sp.CIE1931_2.spectral_locus(wl))
    np.testing.assert_allclose(lam, wl, atol=0.05)
    np.testing.assert_allclose(purity, 1.0, atol=1e-3)
    # a purple: complementary wavelength (negative), purity between 0 and 1
    lam_p, pur_p = sp.dominant_wavelength(np.array([0.35, 0.2]))
    assert lam_p < 0 and 0 < pur_p < 1


def test_color_and_hex():
    assert sp.hex_color(sp.illuminant("D65")) == "#ffffff"
    rgb = sp.color(sp.blackbody(np.array([1500.0, 3000.0, 10000.0])))
    assert rgb.shape == (3, 3)
    assert np.all((rgb >= 0) & (rgb <= 1))
    assert rgb[0, 0] > rgb[0, 2] and rgb[2, 2] > rgb[2, 0]  # warm vs cool
