"""Light sources, thin films, atmosphere, rainbow optics: checks against physical laws."""
import numpy as np
import pytest

import spectrapy as sp
from spectrapy import atmosphere as atm
from spectrapy import optics
from spectrapy import thinfilm as tf
from spectrapy.constants import C2_CIE, SIGMA


# ------------------------------------------------------------------ light sources
def test_stefan_boltzmann_and_wien():
    for T in (1000.0, 3000.0, 6000.0):
        wl = np.geomspace(50, 1e6, 200_000)
        exitance = np.pi * sp.blackbody(T, wavelengths=wl).integrate()  # W/m²
        assert exitance == pytest.approx(SIGMA * T**4, rel=1e-4)
        peak = sp.blackbody(T, wavelengths=np.arange(100, 6000, 0.01)).peak_wavelength
        assert peak == pytest.approx(2.897771955e6 / T, abs=0.02)


def test_planck_matches_colour(colour):
    wl = np.arange(360, 831, 5.0)
    ours = sp.planck(wl, 5000.0, c2=C2_CIE)
    ref = colour.colorimetry.planck_law(wl * 1e-9, 5000.0) * 1e-9
    np.testing.assert_allclose(ours, ref, rtol=1e-6)


def test_illuminant_a_formula_matches_table():
    tab = sp.illuminant("A")
    np.testing.assert_allclose(sp.illuminants.illuminant_a_formula(tab.wavelengths).values, tab.values, rtol=1e-5)


@pytest.mark.parametrize("name", ["D50", "D55", "D75"])
def test_d_series_match_colour(colour, name):
    ref = colour.SDS_ILLUMINANTS[name]
    ours = sp.illuminant(name).resample(ref.wavelengths)
    np.testing.assert_allclose(ours.values, ref.values, atol=2e-3, rtol=1e-4)


def test_daylight_reproduces_d65_table():
    wl = np.arange(300, 831, 5.0)
    d65 = sp.illuminant("D65").resample(wl).values
    np.testing.assert_allclose(sp.daylight(6500 * 1.4388 / 1.4380, wavelengths=wl).values, d65, atol=0.02)


# ---------------------------------------------------------------------- thin films
def _fresnel(n1, n2, theta1_deg):
    t1 = np.radians(theta1_deg)
    ct1 = np.cos(t1)
    ct2 = np.sqrt(1 - (n1 / n2 * np.sin(t1)) ** 2 + 0j)
    rs = (n1 * ct1 - n2 * ct2) / (n1 * ct1 + n2 * ct2)
    rp = (n2 * ct1 - n1 * ct2) / (n2 * ct1 + n1 * ct2)
    return np.abs(rs) ** 2, np.abs(rp) ** 2


def test_single_interface_is_fresnel():
    ang = np.array([0.0, 20.0, 45.0, 70.0, 85.0])
    Rs, Rp = _fresnel(1.0, 1.5, ang)
    R_s, _ = tf.tmm([1.0, 1.5], [], 550.0, ang, "s")
    R_p, _ = tf.tmm([1.0, 1.5], [], 550.0, ang, "p")
    np.testing.assert_allclose(R_s, Rs, atol=1e-12)
    np.testing.assert_allclose(R_p, Rp, atol=1e-12)


def test_brewster_angle_and_total_internal_reflection():
    brewster = np.degrees(np.arctan(1.5))
    assert tf.tmm([1.0, 1.5], [], 550.0, brewster, "p")[0] < 1e-20
    R, T = tf.tmm([1.5, 1.0], [], 550.0, 60.0, "s")
    assert R == pytest.approx(1.0) and T == pytest.approx(0.0, abs=1e-12)


def test_single_absorbing_film_matches_airy_formula():
    wl = np.linspace(400, 700, 31)
    n0, n1, n2, d = 1.0, 2.0 + 0.3j, 1.5, 120.0
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    ph = np.exp(2j * 2 * np.pi * n1 * d / wl)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    R, _ = tf.tmm([n0, n1, n2], [d], wl)
    np.testing.assert_allclose(R, np.abs(r) ** 2, atol=1e-12)


def test_energy_conservation_lossless_multilayer():
    st = tf.Stack([(tf.TIO2, 80), (tf.SIO2, 120), (tf.TIO2, 60), (tf.MGF2, 95)], tf.AIR, tf.BK7)
    for pol in ("s", "p", "unpolarized"):
        R, T = st.rt(sp.wavelengths(), angle=np.array([0, 30, 60, 80])[:, None], polarization=pol)
        np.testing.assert_allclose(R + T, 1.0, atol=1e-12)


def test_quarter_wave_coating():
    nb = tf.BK7(550.0).real
    st = tf.Stack([(tf.MGF2, 550 / (4 * 1.38))], tf.VACUUM, tf.BK7)
    R = st.reflectance([550.0]).values[0]
    assert R == pytest.approx(((nb - 1.38**2) / (nb + 1.38**2)) ** 2, rel=1e-10)


def test_dispersion_models():
    assert tf.BK7(587.56).real == pytest.approx(1.5168, abs=1e-4)
    assert tf.FUSED_SILICA(589.29).real == pytest.approx(1.4584, abs=2e-4)
    assert tf.WATER(589.29).real == pytest.approx(1.3330, abs=3e-4)
    assert tf.TIO2(589.0).real == pytest.approx(2.613, abs=2e-3)


def test_scan_and_map_agree_with_single_calls():
    soap = tf.Stack([(tf.WATER, 0.0)], tf.AIR, tf.AIR)
    d = np.array([100.0, 350.0, 800.0])
    scan = soap.scan(0, d)
    for i, di in enumerate(d):
        np.testing.assert_allclose(scan.values[i], tf.Stack([(tf.WATER, di)]).reflectance().values, atol=1e-14)
    X = soap.xyz_map(0, d, [0.0, 45.0], max_elements=600)
    single = sp.xyz(tf.Stack([(tf.WATER, 350.0)]).reflectance(angle=45.0))
    np.testing.assert_allclose(X[1, 1], single, rtol=1e-12)


# ---------------------------------------------------------------------- atmosphere
def test_rayleigh_optical_depth_and_airmass():
    assert atm.rayleigh_optical_depth(550.0) == pytest.approx(0.0973, abs=5e-4)
    assert atm.rayleigh_optical_depth(400.0) == pytest.approx(0.364, abs=5e-3)
    assert atm.airmass(90.0) == pytest.approx(1.0, abs=1e-3)
    assert atm.airmass(0.0) == pytest.approx(37.9, abs=0.2)


def test_sun_reddens_and_sky_is_blue():
    e = np.array([60.0, 20.0, 5.0, 1.0])
    x_sun = sp.xyz_to_xy(sp.xyz(atm.direct_sunlight(e)))[:, 0]
    assert np.all(np.diff(x_sun) > 0)
    sky = sp.xyz_to_xy(sp.xyz(atm.zenith_sky(np.array([45.0]))))[0]
    assert sky[0] < 0.30 and sky[1] < 0.31


def test_ozone_keeps_twilight_zenith_bluer():
    with_o3 = sp.xyz_to_xy(sp.xyz(atm.zenith_sky(1.0)))
    without = sp.xyz_to_xy(sp.xyz(atm.zenith_sky(1.0, ozone=False)))
    assert with_o3[0] < without[0] - 0.02


# ---------------------------------------------------------------------- optics
def test_rainbow_and_prism_angles():
    n_d = tf.WATER(589.29).real
    assert optics.rainbow_angle(n_d, 1) == pytest.approx(42.0, abs=0.3)
    assert optics.rainbow_angle(n_d, 2) == pytest.approx(50.7, abs=0.4)
    assert optics.prism_deviation(1.5168, 60.0) == pytest.approx(38.65, abs=0.01)
    # minimum deviation really is a minimum
    i = np.linspace(40, 80, 401)
    assert optics.prism_deviation(1.5168, 60.0, i).min() == pytest.approx(38.65, abs=0.01)


def test_rainbow_spectra_shape_and_dark_band():
    th = np.arange(30.0, 60.0, 0.05)
    s = optics.rainbow_spectra(th)
    Y = sp.xyz(s)[:, 1]
    band = (th > 43.5) & (th < 49.5)
    assert Y[band].max() < 0.05 * Y.max()          # Alexander's dark band
    assert Y[th < 40].min() > 0.05 * Y.max()       # bright sky inside the primary bow


def test_prism_ray_trace_matches_analytic_deviation():
    wl = np.array([400.0, 486.1, 550.0, 656.3, 700.0])
    _, paths = optics.prism_rays(wl, optics.SF11, ref_wavelength=550.0)
    d_in, d_out = paths[:, 1] - paths[:, 0], paths[:, 3] - paths[:, 2]
    dev = -np.degrees(np.arctan2(d_out[:, 1], d_out[:, 0]) - np.arctan2(d_in[:, 1], d_in[:, 0]))
    n = optics.SF11(wl).real
    i_min = np.degrees(np.arcsin(optics.SF11(550.0).real * np.sin(np.radians(30.0))))
    np.testing.assert_allclose(dev, optics.prism_deviation(n, 60.0, i_min), atol=1e-9)
    inside = paths[2, 2] - paths[2, 1]
    assert abs(inside[1]) < 1e-12  # minimum deviation: the ray runs parallel to the base
    assert optics.SF11(587.56).real == pytest.approx(1.78472, abs=1e-4)


def test_grating_equation_and_energy_conservation():
    assert optics.grating_angle(550.0, 600.0, 1) == pytest.approx(np.degrees(np.arcsin(0.33)))
    assert np.isnan(optics.grating_angle(550.0, 600.0, 4))
    from spectrapy._util import trapezoid

    xs = np.linspace(0.0, 1.0, 200_001)
    X = optics.grating_screen_xyz(xs, 600.0, orders=(1,), efficiency=lambda m: 1.0)
    src = sp.illuminant("D65").trim(360, 830)
    assert trapezoid(X[:, 1], xs) == pytest.approx(sp.xyz(src)[1] / sp.K_M, rel=1e-5)
    lines = optics.grating_screen_xyz(np.linspace(-2, 2, 4001), 600.0, source=sp.line_spectrum("Hg"))
    assert lines.shape == (4001, 3) and lines[:, 1].max() > 0
