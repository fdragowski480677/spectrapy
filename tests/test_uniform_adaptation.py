import numpy as np
import pytest

from spectrapy import adaptation, uniform

D65 = np.array([0.3127, 0.3290])


def test_lab_luv_oklab_match_colour(colour, rng):
    xyz = rng.uniform(0.01, 1.0, (100, 3))
    np.testing.assert_allclose(uniform.xyz_to_lab(xyz), colour.XYZ_to_Lab(xyz, D65), atol=1e-6)
    np.testing.assert_allclose(uniform.xyz_to_luv(xyz), colour.XYZ_to_Luv(xyz, D65), atol=1e-6)
    np.testing.assert_allclose(uniform.xyz_to_oklab(xyz), colour.XYZ_to_Oklab(xyz), atol=1e-6)


def test_roundtrips(rng):
    xyz = rng.uniform(0.001, 1.0, (100, 3))
    for f, g in ((uniform.xyz_to_lab, uniform.lab_to_xyz), (uniform.xyz_to_luv, uniform.luv_to_xyz),
                 (uniform.xyz_to_oklab, uniform.oklab_to_xyz)):
        np.testing.assert_allclose(g(f(xyz)), xyz, atol=1e-9)
    lab = uniform.xyz_to_lab(xyz)
    np.testing.assert_allclose(uniform.lch_to_lab(uniform.lab_to_lch(lab)), lab, atol=1e-9)


# Sharma, Wu & Dalal (2005) test data (pairs 1, 7, 13, 17, 19, 25, 34)
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0011), 7.2195),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
]


@pytest.mark.parametrize("a,b,expected", SHARMA)
def test_delta_e_2000_sharma(a, b, expected):
    assert uniform.delta_e_2000(np.array(a), np.array(b)) == pytest.approx(expected, abs=1e-4)
    assert uniform.delta_e_2000(np.array(b), np.array(a)) == pytest.approx(expected, abs=1e-4)


def test_delta_e_2000_matches_colour(colour, rng):
    a = rng.uniform([0, -80, -80], [100, 80, 80], (300, 3))
    b = a + rng.normal(0, 6, (300, 3))
    np.testing.assert_allclose(uniform.delta_e_2000(a, b), colour.delta_E(a, b, method="CIE 2000"), atol=1e-6)


@pytest.mark.parametrize("ours,theirs", [("bradford", "Bradford"), ("cat02", "CAT02"), ("cat16", "CAT16"),
                                         ("von_kries", "Von Kries"), ("xyz_scaling", "XYZ Scaling")])
def test_chromatic_adaptation_matches_colour(colour, rng, ours, theirs):
    from colour.adaptation import chromatic_adaptation_VonKries

    xyz = rng.uniform(0.05, 0.9, (50, 3))
    ws, wd = np.array([0.95047, 1.0, 1.08883]), np.array([0.96422, 1.0, 0.82521])
    ref = chromatic_adaptation_VonKries(xyz, ws, wd, transform=theirs)
    np.testing.assert_allclose(adaptation.adapt(xyz, ws, wd, ours), ref, atol=1e-9)


def test_adaptation_maps_white_to_white():
    out = adaptation.adapt(adaptation.white_to_xyz("A"), "A", "D65")
    np.testing.assert_allclose(out / out[1], adaptation.white_to_xyz("D65"), atol=1e-12)
