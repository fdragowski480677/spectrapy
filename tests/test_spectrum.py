import numpy as np
import pytest

import spectrapy as sp
from spectrapy import LineSpectrum, Spectrum


def test_construction_sorts_and_validates():
    s = Spectrum([500, 400, 600], [2.0, 1.0, 3.0])
    assert np.array_equal(s.wavelengths, [400, 500, 600])
    assert np.array_equal(s.values, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Spectrum([400, 500], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Spectrum([400, 500], [1.0, 2.0], kind="radiance")
    with pytest.raises(ValueError):
        Spectrum([400, 400], [1.0, 2.0])


def test_batch_shape_indexing_and_iteration():
    wl = sp.wavelengths(400, 700, 10)
    s = Spectrum(wl, np.ones((4, 3, wl.size)))
    assert s.shape == (4, 3)
    assert s[1].shape == (3,)
    assert s[1, 2].shape == ()
    assert len(list(s)) == 4


def test_interpolation_edges_depend_on_kind():
    e = Spectrum([400, 500], [1.0, 3.0], "emission")
    r = Spectrum([400, 500], [0.2, 0.6], "reflectance")
    assert np.allclose(e.at([350, 450, 550]), [0.0, 2.0, 0.0])
    assert np.allclose(r.at([350, 450, 550]), [0.2, 0.4, 0.6])


def test_integrate_and_normalize():
    wl = sp.wavelengths(400, 700, 0.5)
    s = Spectrum(wl, np.full(wl.size, 2.0))
    assert s.integrate() == pytest.approx(600.0)
    assert s.integrate(450, 550) == pytest.approx(200.0)
    assert s.normalize("area").integrate() == pytest.approx(1.0)
    assert np.max(s.normalize("max").values) == pytest.approx(1.0)
    assert s.normalize("at", 560).at([560])[0] == pytest.approx(1.0)
    assert sp.xyz(sp.blackbody(3000).normalize("luminance"))[1] == pytest.approx(1.0)


def test_arithmetic_uses_overlap_and_union_grid():
    a = Spectrum([400, 500, 600], [1.0, 1.0, 1.0], "emission")
    b = Spectrum([450, 550, 650], [0.5, 0.5, 0.5], "reflectance")
    p = a * b
    assert p.kind == "emission"
    assert p.wavelengths[0] == 450 and p.wavelengths[-1] == 600
    assert np.allclose(p.values, 0.5)
    assert np.allclose((a * 2).values, 2.0)
    assert np.allclose((a + a).values, 2.0)


def test_line_broadening_preserves_power():
    lines = LineSpectrum([500.0, 600.0], [1.0, 3.0])
    for profile in ("gaussian", "lorentzian"):
        s = lines.broaden(fwhm=1.0, profile=profile, wavelengths=np.arange(300, 900, 0.01))
        assert s.integrate() == pytest.approx(4.0, rel=2e-3)


def test_csv_roundtrip(tmp_path):
    s = sp.blackbody(np.array([3000.0, 6000.0]), wavelengths=sp.wavelengths(400, 700, 5))
    p = tmp_path / "bb.csv"
    s.to_csv(p)
    back = sp.io.read_spectrum(p, columns=(0, 2))
    assert np.allclose(back.values, s.values[1], rtol=1e-9)
