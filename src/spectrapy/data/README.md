# Bundled data tables

All tables are plain CSV: comment lines start with `#`, the first other line holds the column
names, and the first column is the wavelength in nanometres (in air for line lists).
`tools/build_data.py` regenerates every file from its source.

| File | Contents | Step | Source |
|---|---|---|---|
| `cmf_cie1931_2.csv` | CIE 1931 2° colour-matching functions | 1 nm, 360–830 | CVRL, `ciexyz31_1.csv` |
| `cmf_cie1964_10.csv` | CIE 1964 10° colour-matching functions | 1 nm, 360–830 | CVRL, `ciexyz64_1.csv` |
| `illuminants_cie_1nm.csv` | CIE standard illuminants A and D65 | 1 nm, 300–830 | CVRL |
| `daylight_basis.csv` | CIE daylight basis functions S0, S1, S2 | 5 nm, 300–830 | CIE 015:2004, via colour-science |
| `illuminants_fl.csv` | CIE fluorescent illuminants FL1–FL12 | 5 nm, 380–780 | CIE 015:2004, via colour-science |
| `illuminants_led.csv` | CIE LED illuminants (B1–B5, BH1, RGB1, V1, V2) | 5 nm, 380–780 | CIE 015:2018, via colour-science |
| `cri_tcs.csv` | CRI test colour samples TCS01–TCS14 | 5 nm, 360–830 | CIE 13.3-1995, via colour-science |

The colorimetric tables are standard data published by the CIE (Commission Internationale de
l'Éclairage). The CVRL files come from the Colour & Vision Research Laboratory database
(<http://www.cvrl.org>); the remaining tables were extracted from the
[colour-science](https://www.colour-science.org) package (BSD-3-Clause), which reproduces the
CIE publications. They are redistributed here with attribution for convenience.

Spectral line lists (`spectrapy.lines`) are curated in the source code. Wavelengths are standard
air wavelengths as listed in the NIST Atomic Spectra Database; relative intensities and the
Fraunhofer line strengths are approximate and meant for illustration, not for quantitative work.
