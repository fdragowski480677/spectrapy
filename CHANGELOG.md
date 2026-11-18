# Changelog

## 0.2.0

- **Wide-gamut output.** `sp.output_space("p3")` (also `"rec2020"`, `"adobe"`) makes plots and colour
  functions work in another RGB space. `sp.style.save` embeds the matching ICC v4 profile in
  PNG/JPEG/TIFF files. The profiles come from the new `spectrapy.icc` module and are validated
  against LittleCMS. The gallery takes `--space p3`.
- **Optics.** `optics.prism_rays` (exact 2D Snell's-law ray tracing), `optics.grating_angle` and
  `optics.grating_screen_xyz` (diffraction-grating spectra on a screen, including line sources),
  and the N-SF11 dense flint glass.
- **Interactive explorers.** `spectrapy.interactive.blackbody_explorer` and `thinfilm_explorer`,
  built on matplotlib widgets.
- **New gallery figures:** prism dispersion, diffraction-grating orders, star colours.
- `render.compress_luminance`, a tone-compression helper for spectral pictures.
- **API reference.** `docs/API.md`, generated from the docstrings by `tools/make_api_docs.py`.
- **Faster CCT/Duv.** Coarse-to-fine search, polyline projection and Newton steps on a cubic
  interpolation of the integrated locus: about 10× faster, accurate to better than 1e-6 relative.
- **Fix:** `"whiten"` highlight handling overflowed for near-neutral over-exposed colours.

## 0.1.0

First version: vectorized colorimetry (`Spectrum`, `LineSpectrum`, CIE observers and illuminants,
XYZ by proper integration), RGB colour spaces with gamut mapping, CIELAB/OKLab/ΔE2000, chromatic
adaptation, CCT/Duv, CIE 13.3 CRI, transfer-matrix thin films, Rayleigh sky, rainbows, line spectra,
matplotlib plotting with light/dark themes, the example gallery, and tests validated against
colour-science.
