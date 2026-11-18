# spectrapy API reference

Generated from the docstrings by `tools/make_api_docs.py`. Everything listed is importable from
its module; the most used names are also available directly as `spectrapy.<name>`.

## Modules

- [`spectrapy.spectrum`](#spectrapyspectrum)
- [`spectrapy.observers`](#spectrapyobservers)
- [`spectrapy.illuminants`](#spectrapyilluminants)
- [`spectrapy.colorimetry`](#spectrapycolorimetry)
- [`spectrapy.colorspaces`](#spectrapycolorspaces)
- [`spectrapy.uniform`](#spectrapyuniform)
- [`spectrapy.adaptation`](#spectrapyadaptation)
- [`spectrapy.temperature`](#spectrapytemperature)
- [`spectrapy.quality`](#spectrapyquality)
- [`spectrapy.lines`](#spectrapylines)
- [`spectrapy.thinfilm`](#spectrapythinfilm)
- [`spectrapy.atmosphere`](#spectrapyatmosphere)
- [`spectrapy.optics`](#spectrapyoptics)
- [`spectrapy.render`](#spectrapyrender)
- [`spectrapy.io`](#spectrapyio)
- [`spectrapy.icc`](#spectrapyicc)
- [`spectrapy.plot`](#spectrapyplot)
- [`spectrapy.style`](#spectrapystyle)
- [`spectrapy.cmap`](#spectrapycmap)
- [`spectrapy.interactive`](#spectrapyinteractive)

## `spectrapy.spectrum`

Spectral distributions: sampled spectra (`Spectrum`) and discrete lines (`LineSpectrum`).

**Conventions**
* Wavelengths are in nanometres and must be ascending.
* `Spectrum.values` has shape `(..., n)`: the *last* axis follows the wavelengths, any leading
  axes form a batch (e.g. 300 blackbodies, or a thickness x angle grid of thin films).
* `kind` says how the values are interpreted:

  `"emission"`
      a spectral density of light per nanometre (radiance, irradiance, power, ...).
      Outside its sampled range the light is zero.
  `"reflectance"` / `"transmittance"`
      a dimensionless factor that needs an illuminant to produce a colour. Outside its sampled
      range the end values are held (CIE 015 practice).

### `spectrum.wavelengths`

```python
function wavelengths(start: "float" = 360.0, stop: "float" = 830.0, step: "float" = 1.0) -> "np.ndarray"
```

Inclusive wavelength grid in nm (default: the CIE 360-830 nm, 1 nm grid).

### `spectrum.Spectrum`

```python
class Spectrum(wavelengths, values, kind: "str" = "emission", name: "str | None" = None)
```

One spectrum or a batch of spectra sampled on a common wavelength grid.

**Parameters**
wavelengths : array_like, shape (n,)
    Wavelengths in nm (sorted automatically if needed).
values : array_like, shape (..., n)
    Spectral values; leading axes form a batch.
kind : {"emission", "reflectance", "transmittance"}
name : str, optional

- **`at(self, wl) -> "np.ndarray"`** — Values at arbitrary wavelengths (linear interpolation; see module notes for outside).
- **`color(self, space=None, **kwargs) -> "np.ndarray"`** — Displayable, encoded RGB in [0, 1]; see `spectrapy.color`.
- **`copy(self) -> "Spectrum"`**
- **`hex(self, space=None, **kwargs)`**
- **`integrate(self, lo: "float | None" = None, hi: "float | None" = None) -> "np.ndarray"`** — Trapezoidal integral over wavelength (optionally over `[lo, hi]` only).
- **`normalize(self, to="max", wavelength: "float | None" = None) -> "Spectrum"`** — Scale the spectrum (each spectrum of a batch separately).
- **`peak_wavelength`** (property): Wavelength of the maximum (per spectrum in a batch).
- **`plot(self, ax=None, **kwargs)`** — Plot with `spectrapy.plot.spectrum`.
- **`resample(self, wl) -> "Spectrum"`** — Spectrum on a new wavelength grid.
- **`shape`** (property): Batch shape (`values.shape[:-1]`); `()` for a single spectrum.
- **`to_csv(self, path, header: "bool" = True) -> "None"`** — Write `wavelength_nm, value[, value, ...]` columns to a CSV file.
- **`trim(self, lo: "float", hi: "float") -> "Spectrum"`** — Keep only samples in `[lo, hi]` nm.
- **`xyz(self, observer="1931", illuminant=None, k=None) -> "np.ndarray"`** — CIE XYZ tristimulus values; see `spectrapy.xyz`.

### `spectrum.LineSpectrum`

```python
class LineSpectrum(wavelengths, intensities=1.0, name: "str | None" = None, labels=None)
```

Discrete spectral lines: wavelengths (nm) and integrated line intensities.

Lines are treated as exact delta functions for colour calculations, which is accurate even when
the lines are far narrower than any wavelength grid (e.g. the sodium D doublet).

- **`broaden(self, fwhm: "float" = 0.5, profile: "str" = "gaussian", wavelengths=None) -> "Spectrum"`** — Sampled emission spectrum with each line spread over a profile of unit area.
- **`color(self, space=None, **kwargs) -> "np.ndarray"`**
- **`normalize(self) -> "LineSpectrum"`** — Strongest line = 1.
- **`scaled(self, factor: "float") -> "LineSpectrum"`**
- **`xyz(self, observer="1931", k=None) -> "np.ndarray"`**

## `spectrapy.observers`

CIE standard colorimetric observers (colour-matching functions).

### `observers.Observer`

```python
class Observer(name: "str", wavelengths, cmf)
```

A set of colour-matching functions x̄(λ), ȳ(λ), z̄(λ) sampled at 1 nm.

- **`at(self, wl) -> "np.ndarray"`** — Colour-matching functions at wavelengths `wl` (nm), shape `wl.shape + (3,)`. Zero outside the tabulated range.
- **`spectral_locus(self, wl=None) -> "np.ndarray"`** — Chromaticities `(x, y)` of monochromatic light (default 380-780 nm, 1 nm).
- **`xbar`** (property)
- **`ybar`** (property)
- **`zbar`** (property)

### `observers.get_observer`

```python
function get_observer(observer="1931") -> "Observer"
```

Return an `Observer` from a name (`"1931"`/`"2"`, `"1964"`/`"10"`) or pass one through.

## `spectrapy.illuminants`

Light sources: Planckian radiators, CIE daylight (D series) and the CIE standard illuminants.

### `illuminants.planck`

```python
function planck(wl_nm, T, c2: "float" = 0.014387768775039337)
```

Planck's law: spectral radiance [W sr^-1 m^-2 nm^-1] at wavelength `wl_nm` and temperature
`T` [K]. Fully broadcasting; very short wavelengths underflow cleanly to 0.

### `illuminants.blackbody`

```python
function blackbody(T, wavelengths=None, normalize: "str | None" = None, c2: "float" = 0.014387768775039337) -> "Spectrum"
```

Spectrum of a Planckian radiator (spectral radiance per nm).

`T` may be an array: the result is then a batch of spectra with `shape == T.shape`.
`normalize`: `None` (absolute radiance), `"max"`, `"560"` (value at 560 nm = 100, the
CIE convention for illuminants) or `"luminance"`.

### `illuminants.daylight_xy`

```python
function daylight_xy(cct)
```

Chromaticity (x, y) of CIE daylight at correlated colour temperature `cct` (4000-25000 K).

### `illuminants.daylight_from_xy`

```python
function daylight_from_xy(xy, wavelengths=None, round_m: "bool" = True) -> "Spectrum"
```

CIE daylight spectrum reconstructed from a chromaticity on the daylight locus.

### `illuminants.daylight`

```python
function daylight(cct, wavelengths=None, round_m: "bool" = True) -> "Spectrum"
```

CIE daylight (D-series) illuminant for a correlated colour temperature [K] (array allowed).

The spectrum comes from the S0/S1/S2 basis functions (5 nm) and is linearly interpolated to
`wavelengths` if given. Values are relative, ~100 at 560 nm.

### `illuminants.illuminant_a_formula`

```python
function illuminant_a_formula(wavelengths=None) -> "Spectrum"
```

CIE standard illuminant A from its defining formula (Planckian radiator, 2856 K).

### `illuminants.available_illuminants`

```python
function available_illuminants() -> "list[str]"
```

Names accepted by `illuminant`.

### `illuminants.illuminant`

```python
function illuminant(name: "str", wavelengths=None) -> "Spectrum"
```

A CIE illuminant by name: `A`, `D50`, `D55`, `D65`, `D75`, `E`, `FL1`-`FL12`,
`LED-B1`... (see `available_illuminants`). Optionally resampled to `wavelengths`.

### `illuminants.resolve_illuminant`

```python
function resolve_illuminant(spec) -> "Spectrum"
```

Accept a `Spectrum`, an illuminant name, or a number (blackbody temperature in K).

## `spectrapy.colorimetry`

From spectra to CIE XYZ and chromaticity coordinates (vectorized over batches of spectra).

### `colorimetry.xyz`

```python
function xyz(obj, observer="1931", illuminant=None, k: "float | None" = None) -> "np.ndarray"
```

CIE XYZ tristimulus values, shape `batch + (3,)`.

* **emission** spectra (and `LineSpectrum`): `XYZ = k ∫ S(λ) cmf(λ) dλ` with
  `k = K_m = 683.002 lm/W` by default, so `Y` is luminance [cd/m²] when `S` is spectral
  radiance [W sr⁻¹ m⁻² nm⁻¹] (illuminance [lx] for irradiance). The integral is evaluated with
  the trapezoid rule on the union of the spectrum's and the observer's grids, so irregular
  or coarse sampling gives the right answer (a plain sum would not).
* **reflectance / transmittance**: relative values under `illuminant` (default D65),
  normalised so that a perfect white has `Y = 1`.
* Lines are exact delta functions: `XYZ = k Σ Pᵢ cmf(λᵢ)`.

### `colorimetry.illuminant_white`

```python
function illuminant_white(illuminant="D65", observer="1931") -> "np.ndarray"
```

XYZ of a perfect white under `illuminant` (`Y = 1`).

### `colorimetry.xyz_to_xy`

```python
function xyz_to_xy(xyz_) -> "np.ndarray"
```

### `colorimetry.xy_to_xyz`

```python
function xy_to_xyz(xy, Y=1.0) -> "np.ndarray"
```

### `colorimetry.xyz_to_uv1960`

```python
function xyz_to_uv1960(xyz_) -> "np.ndarray"
```

### `colorimetry.xyz_to_uv1976`

```python
function xyz_to_uv1976(xyz_) -> "np.ndarray"
```

CIE 1976 UCS chromaticity (u', v').

### `colorimetry.xy_to_uv1960`

```python
function xy_to_uv1960(xy) -> "np.ndarray"
```

### `colorimetry.uv1960_to_xy`

```python
function uv1960_to_xy(uv) -> "np.ndarray"
```

### `colorimetry.xy_to_uv1976`

```python
function xy_to_uv1976(xy) -> "np.ndarray"
```

### `colorimetry.uv1976_to_xy`

```python
function uv1976_to_xy(uvp) -> "np.ndarray"
```

### `colorimetry.dominant_wavelength`

```python
function dominant_wavelength(xy, white="D65", observer="1931")
```

Dominant wavelength [nm] and excitation purity of chromaticities `xy`.

For colours in the purple region the *complementary* wavelength is returned as a negative
number (CIE convention). Returns `(wavelength, purity)` arrays.

### `colorimetry.color`

```python
function color(obj, space=None, observer="1931", illuminant=None, brightness="auto", gamut: "str" = "oklch", highlights: "str" = "scale", exposure: "float" = 1.0, adapt: "bool" = False) -> "np.ndarray"
```

Displayable, encoded RGB in `[0, 1]` (shape `batch + (3,)`).

`obj` is a `Spectrum`, a `LineSpectrum` or XYZ values (relative, white `Y = 1`).
`brightness="auto"` shows light sources as bright as possible (`"max"`) and keeps the
relative lightness of reflecting/transmitting objects and XYZ input (`None`).
With `adapt=True` object colours under a non-D65 `illuminant` are shown as a viewer adapted
to that illuminant would see them (Bradford transform); otherwise they look as in a photo
taken with daylight white balance.

### `colorimetry.hex_color`

```python
function hex_color(obj, **kwargs)
```

Like `color` but returns `'#rrggbb'` string(s).

## `spectrapy.colorspaces`

RGB colour spaces and the XYZ -> displayable RGB pipeline (exposure, gamut mapping, encoding).

Real spectra are often outside every display gamut (monochromatic light always is). How such
colours are brought into gamut matters for how plots look, so the pipeline is explicit:

1. `XYZ -> linear RGB` with the space's matrix (optionally after chromatic adaptation).
2. **gamut** (negative components):
   `"oklch"` (default) reduces chroma at constant OKLab lightness and hue - the perceptual
   approach also used by CSS Color 4 - so out-of-gamut colours keep their perceived hue.
   `"desaturate"` mixes each colour with white of the *same luminance* until it fits - the
   colorimetric equivalent of adding white light (keeps dominant wavelength and luminance).
   `"uniform"` uses one common amount of desaturation for all colours of the call (smooth,
   but paler). `"smooth"` lets the desaturation vary along the last-but-one axis without kinks,
   removing the Mach bands `"desaturate"` shows at the primaries in spectrum strips.
   `"clip"` simply clamps negative components (shifts hue).
3. **brightness**: `None` keeps the absolute scale (`Y = 1` is display white), `"max"` makes
   every colour as bright as possible, `"global"` scales all colours together so the brightest
   one just fits, a number sets the luminance `Y` of every colour.
4. **highlights** (components above 1 after `exposure`): `"scale"` darkens the colour keeping its
   chromaticity, `"whiten"` desaturates towards white keeping luminance (bright cores of
   over-exposed lines turn white, as on a camera sensor), `"clip"` clamps.
5. The transfer function (e.g. the sRGB curve) encodes the result for display.

### `colorspaces.srgb_encode`

```python
function srgb_encode(x)
```

### `colorspaces.srgb_decode`

```python
function srgb_decode(v)
```

### `colorspaces.bt2020_encode`

```python
function bt2020_encode(x)
```

### `colorspaces.bt2020_decode`

```python
function bt2020_decode(v)
```

### `colorspaces.rgb_to_xyz_matrix`

```python
function rgb_to_xyz_matrix(primaries, white) -> "np.ndarray"
```

Normalised primary matrix from primaries `[[xr, yr], [xg, yg], [xb, yb]]` and white xy.

### `colorspaces.RGBColorSpace`

```python
class RGBColorSpace(name, primaries, white, encode, decode, aliases=())
```

An additive RGB colour space defined by primaries, white point and transfer function.

- **`from_xyz(self, xyz, white=None, method: "str" = "bradford") -> "np.ndarray"`** — Linear RGB from XYZ. If `white` (the XYZ's adopted white) differs from the space's white, the colours are chromatically adapted first.
- **`in_gamut(self, rgb, tol: "float" = 1e-09) -> "np.ndarray"`**
- **`to_display(self, xyz, *, brightness=None, gamut: "str" = "oklch", highlights: "str" = "scale", exposure: "float" = 1.0, smooth_radius: "int" = 12, white=None, encode: "bool" = True)`** — XYZ (shape `(..., 3)`) -> RGB in `[0, 1]` ready for display. See the module docs.
- **`to_hex(self, xyz, **kwargs)`** — `'#rrggbb'` string(s) for XYZ colour(s); keyword arguments go to `to_display`.
- **`to_xyz(self, rgb) -> "np.ndarray"`**

### `colorspaces.map_gamut`

```python
function map_gamut(rgb, weights, method: "str" = "desaturate", smooth_radius: "int" = 12, space=None) -> "np.ndarray"
```

Bring negative linear-RGB components to zero; see the module docs for `method`.
`"oklch"` (perceptual hue preserving chroma reduction) needs the colour `space`.

### `colorspaces.map_highlights`

```python
function map_highlights(rgb, weights, method: "str" = "scale") -> "np.ndarray"
```

Handle components above 1; see the module docs for `method`.

### `colorspaces.rgb_to_hex`

```python
function rgb_to_hex(rgb)
```

Encoded RGB in [0, 1] (shape `(..., 3)`) -> `'#rrggbb'` (str or nested list).

### `colorspaces.get_output_space`

```python
function get_output_space() -> "RGBColorSpace"
```

The colour space that displayable colours are produced in (default sRGB).

### `colorspaces.set_output_space`

```python
function set_output_space(space) -> "RGBColorSpace"
```

Set the default output colour space globally; returns the previous one.

### `colorspaces.output_space`

```python
function output_space(space)
```

Produce colours in another space temporarily, e.g. `with output_space("p3"): ...`.

Pair it with `spectrapy.style.save`, which then embeds the matching ICC profile so that
colour-managed viewers show the wider gamut.

### `colorspaces.get_colorspace`

```python
function get_colorspace(space=None) -> "RGBColorSpace"
```

Look up a colour space by name or alias (`"srgb"`, `"p3"`, `"rec2020"`, `"adobe"`...);
`None` means the current output space (see `output_space`).

## `spectrapy.uniform`

Perceptually more uniform colour spaces and colour differences.

All functions accept arrays of shape `(..., 3)`. XYZ is on the relative scale where the reference
white has `Y = 1`.

### `uniform.xyz_to_lab`

```python
function xyz_to_lab(xyz, white="D65") -> "np.ndarray"
```

CIE 1976 L*a*b*.

### `uniform.lab_to_xyz`

```python
function lab_to_xyz(lab, white="D65") -> "np.ndarray"
```

### `uniform.xyz_to_luv`

```python
function xyz_to_luv(xyz, white="D65") -> "np.ndarray"
```

CIE 1976 L*u*v*.

### `uniform.luv_to_xyz`

```python
function luv_to_xyz(luv, white="D65") -> "np.ndarray"
```

### `uniform.lab_to_lch`

```python
function lab_to_lch(lab) -> "np.ndarray"
```

Cartesian (L, a, b) -> cylindrical (L, C, h[degrees]); works for Lab, Luv and OKLab.

### `uniform.lch_to_lab`

```python
function lch_to_lab(lch) -> "np.ndarray"
```

### `uniform.xyz_to_oklab`

```python
function xyz_to_oklab(xyz) -> "np.ndarray"
```

OKLab from XYZ (D65 white, Y = 1).

### `uniform.oklab_to_xyz`

```python
function oklab_to_xyz(lab) -> "np.ndarray"
```

### `uniform.delta_e_76`

```python
function delta_e_76(lab1, lab2) -> "np.ndarray"
```

Euclidean CIE 1976 colour difference.

### `uniform.delta_e_2000`

```python
function delta_e_2000(lab1, lab2, kL: "float" = 1.0, kC: "float" = 1.0, kH: "float" = 1.0) -> "np.ndarray"
```

CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), vectorized.

## `spectrapy.adaptation`

Chromatic adaptation transforms (von Kries type), vectorized over any batch shape.

### `adaptation.white_to_xyz`

```python
function white_to_xyz(white) -> "np.ndarray"
```

Accept a white point as XYZ (3 values), chromaticity xy (2 values) or a name like `"D65"`.

### `adaptation.adaptation_matrix`

```python
function adaptation_matrix(src_white, dst_white, method: "str" = "bradford") -> "np.ndarray"
```

3x3 matrix mapping XYZ seen under `src_white` to corresponding colours under `dst_white`.

### `adaptation.adapt`

```python
function adapt(xyz, src_white, dst_white, method: "str" = "bradford") -> "np.ndarray"
```

Corresponding colours: `xyz` (shape `(..., 3)`) adapted from `src_white` to `dst_white`.

White points can be XYZ, xy or names; only their chromaticity matters for the transform, so
`Y` of the output follows the input scale.

## `spectrapy.temperature`

Correlated colour temperature (CCT), Duv and the Planckian / daylight loci.

The Planckian locus is computed by integrating Planck's law against the chosen observer (CIE 015
convention, c2 = 1.4388e-2 m K), so CCT and Duv are consistent with any observer. CCT is found
with a table search plus Newton refinement in mired space - accurate to well below 0.1 K and
fully vectorized over many chromaticities.

### `temperature.planckian_uv`

```python
function planckian_uv(T, observer="1931") -> "np.ndarray"
```

CIE 1960 (u, v) of a Planckian radiator at temperature(s) `T` [K] (`np.inf` allowed).

### `temperature.planckian_xy`

```python
function planckian_xy(T, observer="1931") -> "np.ndarray"
```

CIE xy of a Planckian radiator at temperature(s) `T` [K].

### `temperature.cct`

```python
function cct(xyz=None, *, xy=None, uv=None, observer="1931", iterations: "int" = 2)
```

Correlated colour temperature [K] and Duv of colours given as XYZ, xy or CIE 1960 uv.

Returns `(T, duv)` arrays. Duv is the signed distance from the Planckian locus in the
CIE 1960 uv diagram (positive = above the locus, i.e. greenish). Colours outside
1000-100000 K are clamped to the end of the range.

Method: nearest node of a dense locus table (coarse-to-fine search), exact projection onto the
adjacent polyline segments, then Newton steps on a cubic interpolation of the integrated
Planckian locus (interpolation error ~1e-11 in uv: CCT to better than 1e-6 relative for lighting-like
colours, ~1e-5 in the extreme corner of very hot colours far from the locus).

### `temperature.uv_from_cct`

```python
function uv_from_cct(T, duv=0.0, observer="1931") -> "np.ndarray"
```

CIE 1960 uv for a CCT [K] and Duv (inverse of `cct`).

### `temperature.xy_from_cct`

```python
function xy_from_cct(T, duv=0.0, observer="1931") -> "np.ndarray"
```

CIE xy for a CCT [K] and Duv.

### `temperature.isotherms`

```python
function isotherms(T, duv: "float" = 0.02, observer="1931") -> "np.ndarray"
```

End points (shape `T.shape + (2, 2)`, xy) of isotemperature lines spanning `±duv`.

### `temperature.daylight_locus`

```python
function daylight_locus(T=None) -> "np.ndarray"
```

xy of the CIE daylight locus (4000-25000 K).

### `temperature.cct_mccamy`

```python
function cct_mccamy(xy) -> "np.ndarray"
```

McCamy's (1992) cubic approximation of CCT from xy (quick, ~±2 K near the locus, 2856-6504 K).

## `spectrapy.quality`

Colour rendering of light sources: CIE 13.3-1995 colour rendering index (CRI).

### `quality.tcs_spectra`

```python
function tcs_spectra() -> "Spectrum"
```

The 14 CIE 13.3 test colour samples as a batch of reflectance spectra (5 nm).

### `quality.CRIResult`

```python
class CRIResult(Ra: "float", R: "np.ndarray", cct: "float", duv: "float", reference: "Spectrum", delta_e: "np.ndarray") -> None
```

Outcome of `cri`.

- **`valid`** (property): CIE 13.3 considers the index meaningful only for |Duv| < 5.4e-3.

### `quality.cri`

```python
function cri(source: "Spectrum") -> "CRIResult"
```

CIE 13.3-1995 colour rendering index of an emission spectrum (CIE 1931 2° observer).

## `spectrapy.lines`

Spectral lines: line profiles, built-in emission line lists and Fraunhofer absorption lines.

Wavelengths are standard air wavelengths (nm) as listed in the NIST Atomic Spectra Database.
**Relative intensities and the Fraunhofer line strengths are approximate** - they depend on the
source conditions and are meant for illustration (spectroscope views, teaching), not for
quantitative spectroscopy. Use `spectrapy.LineSpectrum` with your own data for that.

### `lines.line_profile`

```python
function line_profile(wl, center, fwhm: "float", profile: "str" = "gaussian")
```

Unit-area line profile (per nm) evaluated at `wl`; broadcasting over `wl` and `center`.

### `lines.line_profile_cdf`

```python
function line_profile_cdf(wl, center, fwhm: "float", profile: "str" = "gaussian")
```

Cumulative distribution of `line_profile` (used for exact pixel-binned rendering).

### `lines.binned_lines`

```python
function binned_lines(lines: "LineSpectrum", edges, fwhm: "float", profile: "str" = "gaussian") -> "np.ndarray"
```

Power of each line falling into wavelength bins `edges` (exact integration of the profile),
summed over lines. Returns an array of `len(edges) - 1` values (power per bin).

### `lines.available_elements`

```python
function available_elements() -> "list[str]"
```

### `lines.line_spectrum`

```python
function line_spectrum(element: "str") -> "LineSpectrum"
```

Approximate visible emission lines of a discharge lamp: `"H"`, `"He"`, `"Ne"`, `"Na"`
or `"Hg"` (strongest line = 1).

### `lines.absorption`

```python
function absorption(continuum: "Spectrum", lines=None, wavelengths=None, step: "float" = 0.01, width_scale: "float" = 1.0) -> "Spectrum"
```

Imprint absorption lines on a continuum: `S(λ) = C(λ) Π (1 - dᵢ Lᵢ(λ))` with Lorentzian
profiles of unit peak. `lines` is a sequence of `(label, wavelength, species, depth, fwhm)`
(default `FRAUNHOFER`); the result is sampled every `step` nm. `width_scale` widens all
lines (e.g. to keep them visible in small figures).

### `lines.solar_spectrum`

```python
function solar_spectrum(T: "float" = 5772.0, lo: "float" = 380.0, hi: "float" = 780.0, step: "float" = 0.01, width_scale: "float" = 1.0) -> "Spectrum"
```

Illustrative solar spectrum: a blackbody photosphere with the classic Fraunhofer lines.

## `spectrapy.thinfilm`

Thin-film interference with the transfer-matrix method (multilayers, oblique incidence, s/p
polarisation, dispersive and absorbing materials), vectorized over wavelength, angle and thickness.

Conventions: time dependence exp(-iωt), complex refractive index `n + iκ` with `κ >= 0` for
absorbing media, angles in degrees, thicknesses in nm. All layers are treated coherently; for
layers much thicker than the coherence length of the light (> ~ 10 µm for sunlight) the fringes
become finer than the 1 nm colour-matching grid and should be averaged instead.

### `thinfilm.Material`

```python
class Material(func, name: "str" = "material")
```

A (possibly dispersive, absorbing) optical material: `material(wl_nm) -> complex n`.

- **`n(self, wl_nm) -> "np.ndarray"`**

### `thinfilm.constant`

```python
function constant(n, name: "str | None" = None) -> "Material"
```

Non-dispersive material with refractive index `n` (complex allowed).

### `thinfilm.cauchy`

```python
function cauchy(A: "float", B: "float" = 0.0, C: "float" = 0.0, name: "str" = "Cauchy") -> "Material"
```

Cauchy dispersion `n = A + B/λ² + C/λ⁴` with λ in µm.

### `thinfilm.sellmeier`

```python
function sellmeier(B, C, name: "str" = "Sellmeier") -> "Material"
```

Sellmeier dispersion `n² = 1 + Σ Bᵢ λ² / (λ² - Cᵢ)` with λ in µm and `Cᵢ` in µm².

### `thinfilm.tabulated`

```python
function tabulated(wl_nm, n, k=0.0, name: "str" = "tabulated") -> "Material"
```

Material from tabulated `n` and `k` (linear interpolation, end values held outside).

### `thinfilm.as_material`

```python
function as_material(m) -> "Material"
```

### `thinfilm.tmm`

```python
function tmm(indices, thicknesses, wl, angle=0.0, polarization: "str" = "unpolarized")
```

Reflectance and transmittance of a layer stack (transfer-matrix / characteristic matrix method).

**Parameters**
indices : sequence of array_like
    Complex refractive indices `[n_ambient, n_1, ..., n_N, n_substrate]`, each broadcastable
    with `wl`. The ambient medium must be lossless.
thicknesses : sequence of array_like
    Thicknesses [nm] of the N inner layers (broadcastable, e.g. an array of shape `(m, 1)`
    to scan a thickness for every wavelength at once).
wl : array_like
    Vacuum wavelengths [nm].
angle : array_like
    Angle of incidence in the ambient medium [degrees], broadcastable.
polarization : {"s", "p", "unpolarized"}

**Returns**
R, T : ndarray
    Intensity reflectance and transmittance (broadcast shape of all inputs).

### `thinfilm.Stack`

```python
class Stack(layers=(), ambient=Material("air"), substrate=Material("air"), name: "str | None" = None)
```

A multilayer: `ambient | layer 1 | ... | layer N | substrate`.

Example::

    soap = Stack([(WATER, 400)], ambient=AIR, substrate=AIR)   # 400 nm soap film
    R = soap.reflectance()                                      # Spectrum (reflectance)
    sp.color(R)                                                 # its colour under D65

- **`reflectance(self, wavelengths=None, angle=0.0, polarization="unpolarized") -> "Spectrum"`** — Reflectance spectrum (a batch over `angle` if it is an array).
- **`rt(self, wavelengths=None, angle=0.0, polarization="unpolarized", thicknesses=None)`** — Raw `(R, T)` arrays; see `tmm` for broadcasting rules.
- **`scan(self, layer: "int", thicknesses, wavelengths=None, angle=0.0, polarization="unpolarized", quantity: "str" = "R") -> "Spectrum"`** — Spectra for a range of thicknesses of one layer: a batch of shape `(len(thicknesses),)`.
- **`transmittance(self, wavelengths=None, angle=0.0, polarization="unpolarized") -> "Spectrum"`** — Transmittance into the substrate (a batch over `angle` if it is an array).
- **`xyz_map(self, layer: "int", thicknesses, angles, illuminant="D65", observer="1931", polarization="unpolarized", wavelengths=None, quantity: "str" = "R", max_elements: "float" = 4000000.0)`** — XYZ (white `Y = 1`) of the stack for every (angle, thickness): shape `(n_angles, n_thick, 3)`.

### `thinfilm.film`

```python
function film(material, thickness, ambient=Material("air"), substrate=Material("air")) -> "Stack"
```

A single film, e.g. `film(WATER, 500)` for a soap film in air.

## `spectrapy.atmosphere`

Colours of the sun and sky from Rayleigh scattering (single-scattering, plane-parallel model).

A deliberately simple, transparent model for illustration and teaching:

* Rayleigh optical depth of the whole atmosphere: Bodhaine et al. (1999), eq. 30.
* Relative air mass: Kasten & Young (1989).
* Optional ozone absorption in the Chappuis band, approximated by a Gaussian band
  (peak cross-section ~5e-21 cm² near 602 nm) for a layer at ~22 km. It is what keeps the
  twilight zenith sky blue (Hulburt 1953) - without it the model predicts a greyish-yellow sky.
* Aerosols, multiple scattering and the spherical geometry of Rayleigh scattering are ignored.

### `atmosphere.rayleigh_optical_depth`

```python
function rayleigh_optical_depth(wl_nm, pressure_hpa: "float" = 1013.25) -> "np.ndarray"
```

Vertical Rayleigh optical depth of the atmosphere (Bodhaine et al. 1999, eq. 30).

### `atmosphere.airmass`

```python
function airmass(elevation_deg) -> "np.ndarray"
```

Relative optical air mass for a solar elevation in degrees (Kasten & Young 1989).

### `atmosphere.ozone_airmass`

```python
function ozone_airmass(elevation_deg, height_km: "float" = 22.0) -> "np.ndarray"
```

Air mass of a thin absorbing shell at `height_km` (spherical geometry).

### `atmosphere.ozone_optical_depth`

```python
function ozone_optical_depth(wl_nm, column_du: "float" = 300.0) -> "np.ndarray"
```

Approximate vertical ozone optical depth in the visible (Gaussian Chappuis band).

### `atmosphere.direct_sunlight`

```python
function direct_sunlight(elevation_deg, wavelengths=None, ozone: "bool" = True, pressure_hpa: "float" = 1013.25, column_du: "float" = 300.0) -> "Spectrum"
```

Spectrum of the direct solar beam at the ground (batch over `elevation_deg`).

### `atmosphere.zenith_sky`

```python
function zenith_sky(elevation_deg, wavelengths=None, ozone: "bool" = True, pressure_hpa: "float" = 1013.25, column_du: "float" = 300.0) -> "Spectrum"
```

Spectral radiance of the zenith sky (single Rayleigh scattering, relative units).

For an exponential atmosphere the scattering integral has a closed form:
`I ∝ S P(Θ) (e^(-τ m) - e^(-τ)) / (1 - m)`, with Θ the scattering angle (= solar zenith
angle), `P` the Rayleigh phase function and `m` the solar air mass. Ozone is treated as a
layer above the scatterers.

## `spectrapy.optics`

Geometric optics of dispersion: prisms, diffraction gratings and rainbows (Descartes' theory).

The rainbow model is geometric optics only (no Airy supernumerary bows): for every wavelength the
ray density near the minimum-deviation angle θ_k of order `k` (number of internal reflections)
diverges as `|θ_k - θ|^(-1/2)` on the bright side, which is then smoothed by the finite size of
the solar disk (radius 0.266°). The secondary bow is weaker (factor `secondary_gain`) and has
reversed colours; Alexander's dark band between the bows appears automatically.

### `optics.grating_angle`

```python
function grating_angle(wl_nm, lines_per_mm: "float" = 600.0, order: "int" = 1, incidence_deg: "float" = 0.0) -> "np.ndarray"
```

Diffraction angle (degrees) from the grating equation `d (sin θ_m - sin θ_i) = m λ`;
NaN where the order does not exist.

### `optics.grating_screen_xyz`

```python
function grating_screen_xyz(x, lines_per_mm: "float" = 600.0, orders=(-3, -2, -1, 1, 2, 3), source=None, efficiency=None, observer="1931", line_width: "float" = 0.004) -> "np.ndarray"
```

XYZ of the light from a diffraction grating landing on a flat screen at distance 1 (normal
incidence), at screen positions `x`: shape `x.shape + (3,)`.

Each order spreads the source spectrum (a `Spectrum`, default D65) over the screen; the
power per unit screen length includes the Jacobian `|dλ/dx|`, and overlapping orders simply
add up. For a `spectrapy.LineSpectrum` every line appears in every order as a narrow
Gaussian of width `line_width` (screen units). `efficiency` maps the order number to a
relative efficiency (default `1/|m|`).

### `optics.prism_deviation`

```python
function prism_deviation(n, apex_deg: "float" = 60.0, incidence_deg=None) -> "np.ndarray"
```

Deviation angle (degrees) of a ray through a prism; minimum deviation if `incidence_deg` is None.

### `optics.prism_rays`

```python
function prism_rays(wavelengths, material: "Material" = Material("N-SF11"), apex_deg: "float" = 60.0, incidence_deg: "float | None" = None, ref_wavelength: "float" = 550.0, side: "float" = 1.0, screen_x: "float" = 2.2, start_x: "float" = -1.6)
```

Trace rays of several wavelengths through a prism (exact vector Snell's law, in 2D).

The prism has its apex at the top and its base on `y = 0`; light enters through the left face.
By default the incidence angle is chosen for minimum deviation at `ref_wavelength`.

Returns `(prism_xy, paths)`: the triangle's three vertices, and `paths` of shape
`(n_wavelengths, 4, 2)` - source point, entry point, exit point, and the point where each ray
meets a vertical screen at `x = screen_x`.

### `optics.rainbow_angle`

```python
function rainbow_angle(n, order: "int" = 1) -> "np.ndarray"
```

Rainbow angle (degrees from the antisolar point) for refractive index `n` and `order`.

### `optics.rainbow_spectra`

```python
function rainbow_spectra(theta_deg, material=Material("water"), wavelengths=None, orders=(1, 2), secondary_gain: "float" = 0.43, sun: "Spectrum | None" = None, sun_radius_deg: "float" = 0.266) -> "Spectrum"
```

Spectral radiance (relative) of rainbow light at angles `theta_deg` from the antisolar point.

`theta_deg` must be a uniformly spaced 1-D array. Returns a batch `Spectrum` with one
spectrum per angle (sunlight defaults to a 5772 K blackbody).

## `spectrapy.render`

Turning spectra into arrays of display colours (no matplotlib needed).

These helpers produce the colours behind the plots - use them directly to build your own figures.

### `render.bin_edges`

```python
function bin_edges(lo: "float", hi: "float", n: "int") -> "np.ndarray"
```

### `render.binned_xyz`

```python
function binned_xyz(obj, edges, observer="1931", fwhm: "float" = 0.5, profile: "str" = "gaussian") -> "np.ndarray"
```

XYZ of the light falling in each wavelength bin (`len(edges) - 1` bins) - what one pixel
column of a spectroscope image would record.

`obj` is an emission `Spectrum` (integrated exactly over each bin on a fine grid) or a
`LineSpectrum` (lines broadened to `fwhm` and integrated analytically, so even lines
much narrower than a pixel keep their full power).

### `render.compress_luminance`

```python
function compress_luminance(xyz_, tone: "float" = 0.5) -> "np.ndarray"
```

Compress the brightness range of an array of colours: `Y -> Ymax (Y / Ymax)^tone` with the
chromaticity of every colour unchanged. A display aid (like a camera's tone curve) that keeps dim
parts of a spectrum visible next to bright ones; `tone=1` returns the input unchanged.

### `render.spectral_colors`

```python
function spectral_colors(wl, brightness: "str" = "full", blur_nm: "float | None" = None, space=None, observer="1931", softness: "float" = 6.0, gamut: "str" = "oklch") -> "np.ndarray"
```

Encoded RGB (shape `wl.shape + (3,)`) of monochromatic light at wavelengths `wl` [nm].

`brightness="full"` shows every wavelength as bright as possible (the classic rainbow);
`"luminance"` keeps equal-power light at its true relative brightness (dim violet and red ends).

Monochromatic light lies outside every display gamut. It is first brought into gamut (`gamut`:
`"oklch"` reduces chroma at constant perceptual hue and lightness; `"desaturate"` mixes in
white light of equal luminance), then blurred along the wavelength axis by `blur_nm` (default
8 nm for `"full"`, 6 nm for `"luminance"`) - physically, a spectroscope of that resolution.
The blur removes the bright/dark bands that per-colour gamut mapping produces at the display
primaries; since the RGB cube is convex, blurred colours stay displayable. Full brightness uses a
soft maximum (p-norm, `softness`) instead of max() for the same reason.

### `render.spectrum_colors`

```python
function spectrum_colors(spectra: "Spectrum", **kwargs) -> "np.ndarray"
```

Colour of each spectrum in a batch (alias of `spectrapy.color`).

### `render.strip_colors`

```python
function strip_colors(obj, lo: "float" = 380.0, hi: "float" = 750.0, n: "int" = 1200, space=None, exposure: "float" = 1.0, highlights: "str" = "whiten", gamut: "str" = "oklch", fwhm: "float" = 0.5, profile: "str" = "gaussian", flatten=None, tone: "float" = 1.0) -> "np.ndarray"
```

Encoded RGB of `n` pixel columns spanning `lo`-`hi` nm for an emission spectrum or line
list, with one exposure for the whole strip (the brightest column at 1 x `exposure`).

`tone` < 1 compresses the brightness range (column luminance `Y -> Ymax (Y/Ymax)^tone`,
chromaticity unchanged) - a display aid so that weak lines, e.g. violet ones the eye adapts to,
stay visible next to bright ones. `tone=1` keeps true relative luminance.
`flatten`: an optional continuum `Spectrum`; columns are then shown relative to it
(e.g. a solar spectrum as a uniform rainbow crossed by dark Fraunhofer lines).

## `spectrapy.io`

Reading and writing spectra (CSV / whitespace-delimited text, e.g. spectrometer exports).

### `io.parse_spectrum`

```python
function parse_spectrum(text: "str", unit: "str" = "nm", kind: "str" = "emission", columns=(0, 1), delimiter=None, density: "bool" = True, name: "str | None" = None, scale: "float" = 1.0) -> "Spectrum"
```

Parse a two-column spectrum from text; see `read_spectrum` for the parameters.

### `io.read_spectrum`

```python
function read_spectrum(path, unit: "str" = "nm", kind: "str" = "emission", columns=(0, 1), delimiter=None, density: "bool" = True, name: "str | None" = None, scale: "float" = 1.0) -> "Spectrum"
```

Read a two-column spectrum from a text file (header and comment lines are skipped).

**Parameters**
unit : str
    Unit of the spectral column: `nm`, `um`, `angstrom`, `m` (wavelength), `eV`,
    `cm-1` (wavenumber), `THz` or `Hz`.
kind : {"emission", "reflectance", "transmittance"}
columns : (int, int)
    Zero-based indices of the spectral and value columns.
density : bool
    If the values are a spectral *density* per energy/wavenumber/frequency unit, convert them to
    a density per nm with the Jacobian |dx/dλ|, so integrals and colours stay correct.
scale : float
    Factor applied to the values (e.g. 0.01 for reflectance given in percent).

### `io.to_nanometres`

```python
function to_nanometres(x, unit: "str" = "nm") -> "np.ndarray"
```

Convert spectral coordinates (wavelength, photon energy, wavenumber or frequency) to nm.

### `io.write_spectrum`

```python
function write_spectrum(spectrum: "Spectrum", path) -> "None"
```

Write a spectrum (or batch) as CSV with a `wavelength_nm` column.

## `spectrapy.icc`

Minimal ICC v4 display profiles (matrix/TRC) for the built-in RGB colour spaces.

Used to tag saved images, so that colour-managed viewers (browsers, macOS, iOS, modern Windows)
show e.g. Display P3 figures with their full gamut. The profile contains: description,
copyright, media white point (D50, as v4 requires for displays), a Bradford `chad` matrix from
the space's white to D50, the D50-adapted primaries, and parametric tone curves.

### `icc.icc_profile`

```python
function icc_profile(space="p3", description: "str | None" = None) -> "bytes"
```

ICC v4.3 display profile (bytes) for a built-in colour space (`"p3"`, `"srgb"`, ...).

### `icc.trc_parameters`

```python
function trc_parameters(space: "RGBColorSpace")
```

ICC parametric-curve (function type, parameters) of a colour space's decoding curve.

## `spectrapy.plot`

Plotting with matplotlib. Every function draws into `ax` (or a new one) and returns it -
nothing is saved or shown behind your back, so panels compose freely into your own figures.

Use `spectrapy.style` for the light/dark themes and the `save` helper.

### `plot.absorption_spectrum`

```python
function absorption_spectrum(spectrum_: "Spectrum", ax=None, lo: "float" = 380.0, hi: "float" = 720.0, n: "int" = 1400, continuum: "Spectrum | None" = None, lines=[("K", 393.366), ("H", 396.847), ("h", 410.175), ("G", 430.79), ("F", 486.134), ("b", 517.6), ("E", 527.039), ("D", 589.29), ("a", 627.661), ("C", 656.281), ("B", 686.719), ("A", 759.37)], annotate: "bool" = True, exposure: "float" = 1.0, ticks: "bool" = True, label: "str | None" = None, min_sep: "float" = 0.012)
```

Spectroscope view of an absorption spectrum (e.g. `spectrapy.solar_spectrum`).

With a `continuum` the strip is shown relative to it - a uniform rainbow crossed by dark lines,
as in textbook pictures of the Fraunhofer lines; without, the true brightness is kept.
`lines` (`(label, wavelength, ...)` tuples) are labelled above the strip; labels closer than
`min_sep` (fraction of the width) to the previous one are skipped.

### `plot.chromaticity_diagram`

```python
function chromaticity_diagram(ax=None, kind: "str" = "xy", observer="1931", fill: "bool" = True, resolution: "int" = 700, gamuts=("sRGB",), planckian: "bool" = True, temperatures=(1500, 2000, 3000, 4000, 6500, 10000), daylight: "bool" = False, wavelength_labels: "bool" = True, white_points=(), grid: "bool" = True, blur: "float" = 2.0, locus_lw: "float" = 1.0, legend_loc: "str | None" = None)
```

CIE chromaticity diagram (`kind="xy"` or `"uv"` for CIE 1976 u'v').

The horseshoe is filled with the brightest displayable colour of every chromaticity (colours
outside sRGB are desaturated towards white, so the fill is an honest *approximation* there).
`gamuts` draws RGB colour-space triangles, `planckian` the blackbody locus with
isotemperature lines at `temperatures`, `daylight` the CIE daylight locus.
Returns the `Axes`; overlay sources with `chromaticity_points`.

### `plot.chromaticity_points`

```python
function chromaticity_points(ax, values, labels=None, kind: "str" = "xy", observer="1931", size: "float" = 30, marker: "str" = "o", annotate: "bool" = True, offsets=None, edgecolor=None, **kwargs)
```

Overlay colours on a chromaticity diagram.

`values`: a `Spectrum` (batch allowed) / `LineSpectrum`, XYZ (`(..., 3)`) or
xy (`(..., 2)`). Points are filled with their own colour.

### `plot.cmf`

```python
function cmf(observer="1931", ax=None, compare=None, bar: "bool" = True, fill: "bool" = True)
```

Colour-matching functions x̄, ȳ, z̄ (optionally compared with a second observer, dashed).

### `plot.color_map`

```python
function color_map(x, y, colors, ax=None, xlabel: "str | None" = None, ylabel: "str | None" = None)
```

2D colour image: `colors` of shape `(len(y), len(x), 3)`, encoded RGB.

### `plot.color_strip`

```python
function color_strip(x, colors, ax=None, xlabel: "str | None" = None, ticks: "bool" = True, label: "str | None" = None)
```

Colour as a function of a parameter (`colors` encoded RGB, one row per `x`).

### `plot.cri_bars`

```python
function cri_bars(result, ax=None, show_values: "bool" = True, title: "str | None" = None)
```

Bar chart of the special colour rendering indices R1-R14 of a `spectrapy.cri` result,
each bar in the colour of its test sample (under the reference illuminant).

### `plot.emission_atlas`

```python
function emission_atlas(elements=("H", "He", "Ne", "Na", "Hg"), lo: "float" = 380.0, hi: "float" = 720.0, continuum: "bool" = True, sun: "bool" = True, width: "float" = 8.0, strip_height: "float" = 0.42, exposure: "float" = 1.6, tone: "float" = 0.5, fig=None)
```

A stacked spectroscope figure: continuous spectrum, the Sun with Fraunhofer lines (widths
exaggerated 3x for visibility), and the emission lines of several elements. Returns the `Figure`.

### `plot.emission_spectrum`

```python
function emission_spectrum(obj, ax=None, lo: "float" = 380.0, hi: "float" = 720.0, n: "int" = 1400, fwhm: "float" = 0.8, exposure: "float" = 1.6, tone: "float" = 0.5, label: "str | None" = None, annotate: "bool" = False, min_label: "float" = 0.12, ticks: "bool" = True, background: "str" = "#000000")
```

Spectroscope view of an emission spectrum or line list: bright lines on black.

`obj` is a `LineSpectrum`, an element symbol (`"H"`, `"He"`, `"Ne"`, `"Na"`,
`"Hg"`) or an emission `Spectrum`. `exposure` > 1 over-exposes bright lines, whose
cores then turn white as on a camera sensor. `tone` < 1 compresses the brightness range so weak
lines stay visible (`tone=1`: true relative luminance); see `spectrapy.render.strip_colors`.

### `plot.fill_colors`

```python
function fill_colors(wl, brightness="full", fade_invisible=True, ax=None, space=None) -> "np.ndarray"
```

Spectral colours for filling under spectra; invisible wavelengths fade to a quiet grey.

### `plot.spectrum`

```python
function spectrum(s, ax=None, *, fill: "bool" = True, brightness: "str" = "full", fade_invisible: "bool" = True, alpha: "float" = 1.0, color=None, lw: "float | None" = None, label=None, swatch: "bool" = True, normalize: "str | None" = None, xlim=None, xlabel: "str" = "Wavelength (nm)", ylabel: "str | None" = None, illuminant=None, legend: "bool" = True, space=None)
```

Plot a spectrum with the area under the curve filled by the colours of its wavelengths.

* single `Spectrum`: coloured fill, outline, and (`swatch=True`) a patch on the
  right showing the colour of the whole spectrum with its hex code and CCT.
* batch of spectra: one line per spectrum, drawn in that spectrum's own colour.
* `LineSpectrum`: coloured stems.

Returns the matplotlib `Axes`.

### `plot.swatches`

```python
function swatches(colors, labels=None, ax=None, ncols: "int | None" = None, sublabels="hex", title: "str | None" = None, gap: "float" = 0.14, rounding: "float" = 0.14, label_inside: "bool" = False, fontsize: "float" = 8.0)
```

A grid of rounded colour patches with labels.

`colors`: encoded RGB in [0, 1] (shape `(n, 3)`) or hex strings - e.g. `sp.color(spectra)`.
`sublabels`: `"hex"` (default), a list of strings, or `None`.

### `plot.visible_band`

```python
function visible_band(ax, lo: "float" = 380.0, hi: "float" = 750.0, alpha: "float" = 0.16, zorder: "float" = 0.5)
```

Tint the visible part of a wide wavelength axis with the spectral colours (full height).

### `plot.visible_spectrum`

```python
function visible_spectrum(ax=None, lo: "float" = 380.0, hi: "float" = 750.0, brightness: "str" = "full", blur_nm: "float | None" = None, n: "int" = 1500, ticks: "bool" = True, xlabel: "str | None" = None)
```

The visible spectrum as a colour strip (see `spectrapy.render.spectral_colors`).

### `plot.wavelength_bar`

```python
function wavelength_bar(ax, lo: "float | None" = None, hi: "float | None" = None, height: "float" = 0.045, brightness: "str" = "full", position: "str" = "bottom", pad: "bool" = True)
```

Add a thin visible-spectrum bar along the bottom (or top) edge of a wavelength axis.

With `pad=True` the y range is extended so the bar sits outside the existing data.

## `spectrapy.style`

Figure styles (light / dark) and a small save helper.

Plot functions never change global settings; they read the active matplotlib style, so these
themes apply to everything inside `with sp.style.context("dark"):` or after `sp.style.use()`.

### `style.theme`

```python
function theme(name: "str" = "light") -> "dict"
```

The rcParams dictionary of a theme (`"light"` or `"dark"`).

### `style.use`

```python
function use(name: "str" = "light") -> "None"
```

Apply a theme globally (like `plt.style.use`).

### `style.context`

```python
function context(name: "str" = "light")
```

Apply a theme temporarily: `with sp.style.context("dark"): ...`.

### `style.is_dark`

```python
function is_dark() -> "bool"
```

True if the active axes background is dark.

### `style.save`

```python
function save(fig, path, dpi: "float | None" = None, colorspace=None, **kwargs) -> "pathlib.Path"
```

Save a figure, creating parent folders; the format follows the file extension.

If the colours were produced in a wide-gamut space (`colorspace`, default: the active
`output_space`, e.g. `"p3"`), raster formats get the matching ICC profile embedded so
colour-managed viewers show the full gamut. Vector formats cannot carry the profile; a warning
is issued because viewers will then interpret the colours as sRGB.

### `style.close`

```python
function close(fig) -> "None"
```

## `spectrapy.cmap`

A wavelength colormap: colour-code anything by wavelength in your own matplotlib plots.

Example::

    sc = ax.scatter(x, y, c=wavelength_nm, cmap=sp.cmap.wavelength_cmap(), norm=sp.cmap.wavelength_norm())
    fig.colorbar(sc, label="Wavelength (nm)")

### `cmap.wavelength_cmap`

```python
function wavelength_cmap(lo: "float" = 380.0, hi: "float" = 750.0, brightness: "str" = "full", n: "int" = 512, name: "str | None" = None) -> "ListedColormap"
```

Colormap whose 0..1 range spans `lo`-`hi` nm (use with `wavelength_norm`).

### `cmap.wavelength_norm`

```python
function wavelength_norm(lo: "float" = 380.0, hi: "float" = 750.0) -> "Normalize"
```

### `cmap.register`

```python
function register(name: "str" = "spectrapy_wavelength") -> "str"
```

Register the default wavelength colormap with matplotlib under `name` (idempotent).

## `spectrapy.interactive`

Interactive explorers built on matplotlib's own widgets (no extra dependencies).

Run them from a script or a notebook with an interactive backend (`%matplotlib widget` or
`qt`)::

    import spectrapy as sp
    ex = sp.interactive.blackbody_explorer()
    ex.show()

Keep a reference to the returned `Explorer` - matplotlib widgets stop responding when
they are garbage-collected. `ex.set(T=3000)` drives the explorer from code.

### `interactive.Explorer`

```python
class Explorer(fig: "object", sliders: "dict" = <factory>, update: "object" = None) -> None
```

Handle to an interactive figure: the figure, its sliders and its update function.

- **`set(self, **values) -> "Explorer"`** — Move sliders programmatically (each change triggers a redraw).
- **`show(self) -> "None"`**

### `interactive.blackbody_explorer`

```python
function blackbody_explorer(T: "float" = 5772.0, T_range=(1000.0, 20000.0)) -> "Explorer"
```

Planck spectrum (normalised), its colour and CCT for a temperature chosen with a slider.

### `interactive.thinfilm_explorer`

```python
function thinfilm_explorer(material=Material("water"), thickness: "float" = 400.0, angle: "float" = 0.0, ambient=Material("air"), substrate=Material("air"), illuminant="D65", max_thickness: "float" = 1500.0, index_slider: "bool" = True) -> "Explorer"
```

Reflectance and colour of a single film; sliders for thickness, angle of incidence and (for a
constant index) the refractive index. A strip shows the colour against thickness.
