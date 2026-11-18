import numpy as np
import pytest

import spectrapy as sp
from spectrapy import temperature


def test_known_cct_values():
    T, duv = sp.cct(sp.xyz(sp.illuminant("D65")))
    assert float(T) == pytest.approx(6503.0, abs=1.5)
    assert float(duv) == pytest.approx(0.0032, abs=1e-4)
    T_a, duv_a = sp.cct(sp.xyz(sp.illuminant("A")))
    assert float(T_a) == pytest.approx(2848 * 1.4388 / 1.435, abs=0.5)  # 2856 K in the ITS-90 c2 scale
    assert abs(float(duv_a)) < 1e-5


def test_cct_matches_colour_ohno(colour, rng):
    T = rng.uniform(1500, 25000, 60)
    duv = rng.uniform(-0.02, 0.02, 60)
    uv = np.array([colour.temperature.CCT_to_uv(np.array([t, d]), method="Ohno 2013") for t, d in zip(T, duv)])
    Tc, Dc = sp.cct(uv=uv)
    np.testing.assert_allclose(Tc, T, rtol=1e-4)
    np.testing.assert_allclose(Dc, duv, atol=2e-6)


def test_cct_roundtrip_is_exact(rng):
    # typical lighting range: essentially exact
    T = rng.uniform(1000, 20000, 500)
    duv = rng.uniform(-0.02, 0.02, 500)
    Tc, Dc = temperature.cct(uv=temperature.uv_from_cct(T, duv))
    np.testing.assert_allclose(Tc, T, rtol=1e-6)
    np.testing.assert_allclose(Dc, duv, atol=1e-10)
    # extreme corner (very hot, far off the locus, near its centre of curvature): still < 1e-5
    T = rng.uniform(20000, 100000, 500)
    duv = rng.uniform(-0.03, 0.03, 500)
    Tc, Dc = temperature.cct(uv=temperature.uv_from_cct(T, duv))
    np.testing.assert_allclose(Tc, T, rtol=1e-5)
    np.testing.assert_allclose(Dc, duv, atol=1e-10)


def test_planck_locus_is_on_duv_zero():
    T = np.geomspace(1200, 50000, 30)
    Tc, Dc = sp.cct(xy=sp.planckian_xy(T))
    np.testing.assert_allclose(Tc, T, rtol=1e-6)
    assert np.max(np.abs(Dc)) < 1e-9


def test_daylight_locus_matches_colour(colour):
    for T in (4000.0, 5003.0, 6504.0, 9000.0, 20000.0):
        np.testing.assert_allclose(temperature.daylight_locus(np.array([T]))[0],
                                   colour.temperature.CCT_to_xy_CIE_D(T), atol=1e-10)


def test_mccamy(colour):
    xy = np.array([0.31271, 0.32902])
    assert temperature.cct_mccamy(xy) == pytest.approx(colour.temperature.xy_to_CCT_McCamy1992(xy), rel=1e-9)


@pytest.mark.parametrize("name", ["FL2", "FL11", "LED-B3", "LED-RGB1", "LED-B5"])
def test_cri_matches_colour(colour, name):
    ours = sp.cri(sp.illuminant(name))
    ref = colour.colour_rendering_index(colour.SDS_ILLUMINANTS[name], additional_data=True)
    assert ours.Ra == pytest.approx(ref.Q_a, abs=0.1)
    refR = np.array([ref.Q_as[i].Q_a for i in range(1, 15)])
    np.testing.assert_allclose(ours.R, refR, atol=0.8)


def test_cri_of_reference_like_sources():
    for s in (sp.illuminant("A"), sp.illuminant("D65"), sp.blackbody(3000)):
        assert sp.cri(s).Ra == pytest.approx(100.0, abs=0.05)
    assert sp.cri(sp.illuminant("FL2")).valid
