import numpy as np
import pytest

import spectrapy as sp
from spectrapy.colorspaces import map_gamut, rgb_to_hex
from spectrapy.uniform import lab_to_lch, xyz_to_oklab

SPACES = [(sp.SRGB, "sRGB", 2e-4), (sp.DISPLAY_P3, "Display P3", 1e-6), (sp.REC2020, "ITU-R BT.2020", 1e-6),
          (sp.ADOBE_RGB, "Adobe RGB (1998)", 1e-4)]


@pytest.mark.parametrize("ours,key,tol", SPACES)
def test_matrices_match_colour(colour, ours, key, tol):
    np.testing.assert_allclose(ours.matrix_to_xyz, colour.RGB_COLOURSPACES[key].matrix_RGB_to_XYZ, atol=tol)


def test_srgb_encoding_matches_colour(colour, rng):
    # colour-science uses the rounded IEC 61966-2-1 matrix, spectrapy derives it from the primaries:
    # compare the linear part with a tolerance and the transfer function exactly.
    xyz = rng.uniform(0.05, 0.6, (200, 3))
    ref_lin = xyz @ colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB.T
    np.testing.assert_allclose(sp.SRGB.from_xyz(xyz), ref_lin, atol=3e-4)
    x = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(sp.colorspaces.srgb_encode(x), colour.models.eotf_inverse_sRGB(x), atol=1e-12)
    ok = sp.SRGB.in_gamut(sp.SRGB.from_xyz(xyz))
    np.testing.assert_allclose(sp.SRGB.to_display(xyz[ok]), colour.XYZ_to_sRGB(xyz[ok]), atol=1e-3)


@pytest.mark.parametrize("cs", [sp.SRGB, sp.DISPLAY_P3, sp.REC2020, sp.ADOBE_RGB])
def test_transfer_roundtrip(cs):
    x = np.linspace(-0.2, 1.2, 301)
    np.testing.assert_allclose(cs.decode(cs.encode(x)), x, atol=1e-12)


@pytest.mark.parametrize("method", ["clip", "desaturate", "uniform", "smooth", "oklch"])
def test_gamut_mapping_produces_displayable_colours(method):
    X = sp.CIE1931_2.at(np.linspace(380, 780, 400))
    rgb = sp.SRGB.to_display(X, brightness="max", gamut=method)
    assert np.all(np.isfinite(rgb)) and np.all((rgb >= 0) & (rgb <= 1))


def test_desaturate_preserves_luminance():
    X = sp.CIE1931_2.at(np.linspace(420, 680, 100))
    rgb = sp.SRGB.from_xyz(X)
    mapped = map_gamut(rgb, sp.SRGB.luminance_weights, "desaturate")
    assert np.all(mapped >= -1e-12)
    np.testing.assert_allclose(mapped @ sp.SRGB.luminance_weights, X[:, 1], rtol=1e-10)


def test_oklch_preserves_hue():
    X = sp.CIE1931_2.at(np.linspace(430, 650, 60))
    rgb = sp.SRGB.from_xyz(X)
    mapped = map_gamut(rgb, sp.SRGB.luminance_weights, "oklch", space=sp.SRGB)
    h0 = lab_to_lch(xyz_to_oklab(X))[:, 2]
    h1 = lab_to_lch(xyz_to_oklab(sp.SRGB.to_xyz(mapped)))[:, 2]
    dh = (h1 - h0 + 180) % 360 - 180
    assert np.max(np.abs(dh)) < 0.5


def test_highlight_modes():
    rgb_over = np.array([[2.0, 1.0, 0.5]])
    w = sp.SRGB.luminance_weights
    from spectrapy.colorspaces import map_highlights

    assert np.allclose(map_highlights(rgb_over, w, "scale"), [[1.0, 0.5, 0.25]])
    assert np.allclose(map_highlights(rgb_over, w, "clip"), [[1.0, 1.0, 0.5]])
    whitened = map_highlights(np.array([[3.0, 3.0, 3.0]]) * [[1, 0.5, 0.2]], w, "whiten")
    assert np.all(whitened <= 1.0 + 1e-12)


def test_hex_and_lookup():
    assert rgb_to_hex([1.0, 0.0, 0.5]) == "#ff0080"
    assert rgb_to_hex(np.zeros((2, 3))) == ["#000000", "#000000"]
    assert sp.get_colorspace("p3") is sp.DISPLAY_P3
    with pytest.raises(ValueError):
        sp.get_colorspace("cmyk")


def test_adapted_object_colours():
    wl = sp.wavelengths(380, 780, 5)
    grey = sp.Spectrum(wl, np.full(wl.size, 0.5), "reflectance")
    under_a = sp.color(grey, illuminant="A")
    adapted = sp.color(grey, illuminant="A", adapt=True)
    assert under_a[0] > under_a[2] + 0.2          # looks orange in a daylight-balanced picture
    assert np.ptp(adapted) < 0.01                  # neutral to an observer adapted to A


def test_whiten_handles_neutral_overexposure():
    import warnings

    from spectrapy.colorspaces import map_highlights

    rgb = np.array([[3.0, 3.0, 3.0], [1.0 + 1e-15, 1.0, 1.0], [2.0, 1.0, 0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = map_highlights(rgb, sp.SRGB.luminance_weights, "whiten")
    assert np.all(np.isfinite(out)) and np.all(out <= 1.0) and np.allclose(out[0], 1.0)
