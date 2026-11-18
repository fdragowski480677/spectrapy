"""ICC profiles (validated with LittleCMS through Pillow) and wide-gamut output."""
import io
import struct
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectrapy as sp
from spectrapy.icc import icc_profile

cms = pytest.importorskip("PIL.ImageCms")
from PIL import Image  # noqa: E402


@pytest.mark.parametrize("space", ["p3", "srgb", "rec2020", "adobe"])
def test_profile_structure(space):
    data = icc_profile(space)
    assert struct.unpack(">I", data[:4])[0] == len(data)
    assert data[36:40] == b"acsp" and data[12:16] == b"mntr" and data[8] == 4
    p = cms.ImageCmsProfile(io.BytesIO(data))
    assert p.profile.xcolor_space.strip() == "RGB" and p.profile.connection_space.strip() == "XYZ"


@pytest.mark.parametrize("space", ["p3", "rec2020", "adobe"])
def test_littlecms_agrees_with_spectrapy(space, rng):
    cs = sp.get_colorspace(space)
    src = cms.ImageCmsProfile(io.BytesIO(icc_profile(space)))
    t = cms.buildTransform(src, cms.createProfile("sRGB"), "RGB", "RGB",
                           renderingIntent=cms.Intent.RELATIVE_COLORIMETRIC)
    enc = np.round(rng.uniform(0.05, 0.95, (3000, 3)) * 255) / 255
    lin_s = sp.SRGB.from_xyz(cs.to_xyz(cs.decode(enc)))
    ok = np.all((lin_s >= 0) & (lin_s <= 1), axis=-1)
    img = Image.fromarray(np.round(enc[ok] * 255).astype(np.uint8)[None])
    out = np.asarray(cms.applyTransform(img, t)).reshape(-1, 3) / 255.0
    assert np.max(np.abs(out - sp.SRGB.encode(lin_s[ok]))) < 1.5 / 255


def test_output_space_context():
    wl = np.array([520.0])
    assert sp.get_output_space() is sp.SRGB
    srgb = sp.render.spectral_colors(wl)
    with sp.output_space("p3") as cs:
        assert cs is sp.DISPLAY_P3 and sp.get_output_space() is sp.DISPLAY_P3
        p3 = sp.render.spectral_colors(wl)
        assert sp.hex_color(sp.illuminant("D65")) == "#ffffff"
    assert sp.get_output_space() is sp.SRGB
    # the same green, encoded in P3, needs a smaller red share: P3 can show it more saturated
    assert not np.allclose(srgb, p3)
    prev = sp.set_output_space("rec2020")
    try:
        assert sp.get_colorspace(None) is sp.REC2020
    finally:
        sp.set_output_space(prev)


def test_save_embeds_profile(tmp_path):
    fig, ax = plt.subplots()
    with sp.output_space("p3"):
        sp.plot.visible_spectrum(ax=ax)
        png = sp.style.save(fig, tmp_path / "p3.png", dpi=50)
        with pytest.warns(UserWarning, match="ICC"):
            sp.style.save(fig, tmp_path / "p3.svg")
    assert Image.open(png).info.get("icc_profile") == icc_profile("p3")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = sp.style.save(fig, tmp_path / "srgb.png", dpi=50)
    assert Image.open(plain).info.get("icc_profile") is None
