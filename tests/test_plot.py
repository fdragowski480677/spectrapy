"""Smoke tests: every plotting function runs, returns Axes/Figures and draws something."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrapy as sp


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_spectrum_plots(theme):
    with sp.style.context(theme):
        ax = sp.plot.spectrum(sp.blackbody(4000))
        assert ax.images and ax.lines
        ax = sp.plot.spectrum(sp.blackbody(np.array([2000.0, 6000.0])), label=["a", "b"])
        assert len(ax.lines) == 2
        ax = sp.plot.spectrum(sp.line_spectrum("Hg"))
        assert ax.collections
        refl = sp.thinfilm.film(sp.thinfilm.WATER, 300).reflectance()
        sp.plot.spectrum(refl)


def test_strips_and_spectroscope_views():
    assert sp.plot.visible_spectrum().images
    assert sp.plot.emission_spectrum("Ne", annotate=True).images
    s = sp.solar_spectrum(step=0.05)
    assert sp.plot.absorption_spectrum(s, continuum=sp.blackbody(5772, wavelengths=s.wavelengths)).images
    fig = sp.plot.emission_atlas(("H", "Na"))
    assert len(fig.axes) == 4


def test_cmf_and_wavelength_bar():
    ax = sp.plot.cmf("1931", compare="1964")
    assert len(ax.lines) == 6 and ax.images
    _, ax2 = plt.subplots()
    ax2.plot([400, 700], [0, 1])
    ylim = ax2.get_ylim()
    sp.plot.wavelength_bar(ax2)
    assert ax2.get_ylim()[0] < ylim[0]


@pytest.mark.parametrize("kind", ["xy", "uv"])
def test_chromaticity(kind):
    ax = sp.plot.chromaticity_diagram(kind=kind, gamuts=("sRGB", "Rec. 2020"), daylight=True, resolution=120)
    sp.plot.chromaticity_points(ax, sp.illuminant("D65"), ["D65"], kind=kind)
    sp.plot.chromaticity_points(ax, np.array([[0.4, 0.4], [0.3, 0.3]]), kind=kind)
    assert ax.images


def test_patch_plots():
    ax = sp.plot.swatches(sp.color(sp.blackbody(np.linspace(2000, 9000, 8))), labels=list("abcdefgh"))
    assert ax.patches
    sp.plot.swatches(["#ff0000", "#00ff00"], label_inside=True, sublabels=None)
    x = np.linspace(0, 1000, 50)
    sp.plot.color_strip(x, np.random.default_rng(0).random((50, 3)), xlabel="t")
    sp.plot.color_map(x, x[:10], np.random.default_rng(0).random((10, 50, 3)))
    ax = sp.plot.cri_bars(sp.cri(sp.illuminant("FL2")))
    assert len(ax.patches) == 14


def test_colormap_and_save(tmp_path):
    cm = sp.cmap.wavelength_cmap()
    assert cm.N == 512
    name = sp.cmap.register()
    assert sp.cmap.register() == name
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[450, 650], cmap=cm, norm=sp.cmap.wavelength_norm())
    fig.colorbar(sc)
    for ext in ("png", "svg", "pdf"):
        p = sp.style.save(fig, tmp_path / "sub" / f"fig.{ext}", dpi=60)
        assert p.exists() and p.stat().st_size > 0


def test_gallery_subset(tmp_path):
    from spectrapy import gallery

    paths = gallery.render(["spectrum_rendering", "light_sources", "sky"], out=tmp_path, dpi=50, verbose=False)
    assert all(p.exists() for p in paths)
    assert set(gallery.FIGURES) >= {"hero", "chromaticity", "thin_films", "rainbow"}
