import numpy as np
import pytest

import spectrapy as sp
from spectrapy import io
from spectrapy._util import trapezoid
from spectrapy.lines import binned_lines, line_profile


def test_builtin_line_lists():
    for el in sp.lines.available_elements():
        ls = sp.line_spectrum(el)
        assert len(ls) > 3 and ls.intensities.max() == pytest.approx(1.0)
    assert sp.line_spectrum("sodium").name == "Sodium"
    with pytest.raises(ValueError):
        sp.line_spectrum("Xx")


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_profiles_have_unit_area_and_binning_is_exact(profile):
    x = np.arange(400, 700, 0.001)
    p = line_profile(x, 550.0, 0.8, profile)
    assert trapezoid(p, x) == pytest.approx(1.0, rel=2e-3)
    lines = sp.LineSpectrum([500.0, 600.0], [1.0, 2.0])
    binned = binned_lines(lines, np.linspace(300, 900, 601), 0.8, profile)
    assert binned.sum() == pytest.approx(3.0, rel=2e-3)


def test_absorption_and_solar_spectrum():
    s = sp.solar_spectrum()
    bb = sp.blackbody(5772, wavelengths=s.wavelengths)
    assert np.all(s.values <= bb.values + 1e-9)
    T_lines, _ = sp.cct(sp.xyz(s))
    T_bb, _ = sp.cct(sp.xyz(bb))
    assert float(T_lines) < float(T_bb)  # the blue is more strongly absorbed


def test_read_csv_with_header_and_comments(tmp_path):
    p = tmp_path / "meas.csv"
    p.write_text("# exported by spectrometer\nwavelength;intensity\n400;1.0\n500;2.0\n600;3.0\n")
    s = io.read_spectrum(p)
    assert np.array_equal(s.wavelengths, [400, 500, 600]) and s.name == "meas"
    r = io.parse_spectrum("400 50\n700 80\n", kind="reflectance", scale=0.01)
    assert np.allclose(r.values, [0.5, 0.8])


def test_energy_units_conserve_integrals():
    E = np.linspace(1.6, 3.2, 4000)  # eV
    S_E = np.exp(-0.5 * ((E - 2.3) / 0.1) ** 2)  # W/eV
    text = "\n".join(f"{e} {s}" for e, s in zip(E, S_E))
    s = io.parse_spectrum(text, unit="eV")
    assert s.integrate() == pytest.approx(trapezoid(S_E, E), rel=1e-4)
    # per-nm density = per-eV density x |dE/dλ| = S_E(hc/λ) hc/λ² (the Jacobian also shifts the peak)
    lam = np.array([480.0, 540.0, 600.0])
    hc = sp.constants.HC_EV_NM
    expected = np.exp(-0.5 * ((hc / lam - 2.3) / 0.1) ** 2) * hc / lam**2
    np.testing.assert_allclose(s.at(lam), expected, rtol=1e-4)
    assert s.peak_wavelength < hc / 2.3
    assert io.to_nanometres([20000.0], "cm-1")[0] == pytest.approx(500.0)
    assert io.to_nanometres([5000.0], "angstrom")[0] == pytest.approx(500.0)
