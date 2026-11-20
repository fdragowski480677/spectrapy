"""CIECAM02 / CAM16 / UCS and ANSI/IES TM-30, validated against colour-science."""
import numpy as np
import pytest

import spectrapy as sp
from spectrapy import appearance as ap
from spectrapy.quality import ces_spectra, cie224_reference

XYZ_W = np.array([95.05, 100.0, 108.88])
CONDITIONS = [(318.31, 20.0, "average", False), (31.83, 20.0, "dim", False), (4.0, 10.0, "dark", False),
              (100.0, 20.0, "average", True)]


@pytest.mark.parametrize("L_A,Y_b,surround,discount", CONDITIONS)
def test_ciecam02_and_cam16_match_colour(colour, rng, L_A, Y_b, surround, discount):
    xyz = rng.uniform(1.0, 95.0, (300, 3))
    vc02 = colour.VIEWING_CONDITIONS_CIECAM02[surround.capitalize()]
    vc16 = colour.VIEWING_CONDITIONS_CAM16[surround.capitalize()]
    r02 = colour.XYZ_to_CIECAM02(xyz, XYZ_W, L_A, Y_b, vc02, discount_illuminant=discount)
    r16 = colour.XYZ_to_CAM16(xyz, XYZ_W, L_A, Y_b, vc16, discount_illuminant=discount)
    o02 = ap.xyz_to_ciecam02(xyz, XYZ_W, L_A, Y_b, surround, discount)
    o16 = ap.xyz_to_cam16(xyz, XYZ_W, L_A, Y_b, surround, discount)
    for ours, ref in ((o02, r02), (o16, r16)):
        for k in ("J", "C", "h", "M", "s", "Q"):
            np.testing.assert_allclose(getattr(ours, k), getattr(ref, k), atol=1e-9, err_msg=k)


def test_white_scale_invariance():
    xyz = np.array([[20.0, 30.0, 40.0]])
    a = ap.xyz_to_cam16(xyz, XYZ_W)
    b = ap.xyz_to_cam16(xyz / 100.0, XYZ_W / 100.0)
    np.testing.assert_allclose(a.JMh, b.JMh, atol=1e-10)


def test_ucs_matches_colour_and_roundtrips(colour, rng):
    JMh = np.column_stack([rng.uniform(5, 95, 200), rng.uniform(0, 60, 200), rng.uniform(0, 360, 200)])
    np.testing.assert_allclose(ap.jmh_to_ucs(JMh), colour.JMh_CIECAM02_to_CAM02UCS(JMh), atol=1e-10)
    np.testing.assert_allclose(ap.jmh_to_ucs(JMh), colour.JMh_CAM16_to_CAM16UCS(JMh), atol=1e-10)
    np.testing.assert_allclose(ap.ucs_to_jmh(ap.jmh_to_ucs(JMh)), JMh, atol=1e-9)


def test_cam16_inverse(colour, rng):
    xyz = rng.uniform(1.0, 95.0, (300, 3))
    spec = ap.xyz_to_cam16(xyz, XYZ_W, 64.0, 20.0)
    back = ap.cam16_to_xyz(spec.J, spec.M, spec.h, XYZ_W, 64.0, 20.0)
    np.testing.assert_allclose(back, xyz, atol=1e-9)
    ref = colour.CAM16_to_XYZ(colour.CAM_Specification_CAM16(J=spec.J, M=spec.M, h=spec.h), XYZ_W, 64.0, 20.0)
    np.testing.assert_allclose(back, ref, atol=1e-9)


def test_ces_data_matches_colour(colour):
    from colour import SpectralShape
    from colour.quality.cfi2017 import load_TCS_CIE2017

    ref = load_TCS_CIE2017(SpectralShape(380, 780, 1))
    ces = ces_spectra()
    assert ces.shape == (99,)
    np.testing.assert_allclose(ces.values, ref.values.T, atol=5e-7)


@pytest.mark.parametrize("name", ["FL2", "FL11", "LED-B3", "LED-B5", "LED-RGB1", "LED-V1"])
def test_tm30_matches_colour(colour, name):
    from colour.quality import colour_fidelity_index_ANSIIESTM3018

    ours = sp.tm30(sp.illuminant(name))
    ref = colour_fidelity_index_ANSIIESTM3018(colour.SDS_ILLUMINANTS[name], additional_data=True)
    assert ours.Rf == pytest.approx(ref.R_f, abs=0.02)
    assert ours.Rg == pytest.approx(ref.R_g, abs=0.02)
    np.testing.assert_allclose(ours.Rf_samples, ref.R_s, atol=0.02)
    np.testing.assert_allclose(ours.Rf_hues, ref.R_fs, atol=0.02)
    np.testing.assert_allclose(ours.Rcs, ref.R_cs, atol=0.02)
    np.testing.assert_allclose(ours.Rhs, ref.R_hs, atol=1e-3)


def test_tm30_of_reference_sources_is_perfect():
    for s in (sp.blackbody(3000.0), sp.daylight(6500.0)):
        r = sp.tm30(s)
        assert r.Rf == pytest.approx(100.0, abs=0.01) and r.Rg == pytest.approx(100.0, abs=0.01)
        assert np.all(np.abs(r.Rcs) < 0.05)


def test_cie224_reference_blend_is_continuous():
    wl = np.arange(380.0, 781.0, 5.0)

    def xy(T):
        return sp.xyz_to_xy(sp.xyz(sp.Spectrum(wl, cie224_reference(T, wl))))

    for T in (4000.0, 5000.0):
        np.testing.assert_allclose(xy(T - 0.01), xy(T + 0.01), atol=2e-5)


def test_tm30_plots():
    r = sp.tm30(sp.illuminant("LED-B5"))
    ax = sp.plot.tm30_cvg(r)
    assert ax.patches and ax.lines
    sp.plot.tm30_cvg(r, annotate="title", name="LED-B5", background=False)
    for metric in ("Rcs", "Rhs", "Rf"):
        assert len(sp.plot.tm30_bars(r, metric).patches) == 16
    fig = sp.plot.tm30_report(r)
    assert len(fig.axes) == 4
