"""Build and execute ``examples/getting_started.ipynb`` (the cells live here, in one place).

    python tools/build_notebook.py                 # write the executed notebook
    python tools/build_notebook.py --out x.ipynb   # write elsewhere (CI uses this as a smoke test)

Needs ``nbformat``, ``nbclient`` and ``ipykernel`` (not runtime dependencies of spectrapy).
"""
from __future__ import annotations

import argparse
import pathlib

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]

CELLS = [
    ("md", """# Getting started with spectrapy

spectrapy turns spectra (power versus wavelength) into colours and draws spectral pictures with
matplotlib. This notebook is a guided tour; every figure function returns an `Axes` or a `Figure`
that you can keep editing.

```bash
pip install spectrapy-colour      # then: import spectrapy
```"""),
    ("code", """import numpy as np
import matplotlib.pyplot as plt
import spectrapy as sp

sp.style.use("light")                   # or "dark"; plain matplotlib styles work too
plt.rcParams["figure.dpi"] = 90         # keep the notebook light
sp.__version__"""),
    ("md", """## 1. Spectra and their colours

A `Spectrum` is a wavelength grid (nm) plus values. Light sources are `kind="emission"`; for
those, `sp.xyz` integrates against the CIE colour-matching functions and `Y` comes out as luminance."""),
    ("code", """sun = sp.blackbody(5772)                # the Sun as a 5772 K blackbody
print(sun)
print("XYZ:", sp.xyz(sun))
print("colour:", sp.hex_color(sun))
T, duv = sp.cct(sp.xyz(sun))
print(f"CCT = {float(T):.0f} K, Duv = {float(duv):+.4f}")"""),
    ("code", """ax = sp.plot.spectrum(sun.normalize("max"))
ax.set_title("A 5772 K blackbody");"""),
    ("md", """Arrays of temperatures give *batches* of spectra, and everything is vectorized:"""),
    ("code", """T = np.array([2000, 3000, 4500, 6500, 10000])
batch = sp.blackbody(T, normalize="max")
ax = sp.plot.spectrum(batch, label=[f"{t} K" for t in T])   # lines in each spectrum's own colour
ax.set_title("Blackbodies (normalised)");"""),
    ("code", """%%time
colors = sp.color(sp.blackbody(np.linspace(1000, 20000, 10_000)))   # 10 000 spectra -> sRGB
colors.shape"""),
    ("md", """## 2. CIE illuminants and the chromaticity diagram"""),
    ("code", """names = ["A", "D65", "FL2", "FL11", "LED-B3", "LED-RGB1"]
ax = sp.plot.chromaticity_diagram(gamuts=("sRGB", "Display P3"), resolution=400)
for n in names:
    sp.plot.chromaticity_points(ax, sp.illuminant(n), [n])"""),
    ("md", """## 3. Line spectra and the spectroscope view

Discrete lines are exact delta functions for colour calculations; they are only broadened for
drawing."""),
    ("code", """with sp.style.context("dark"):
    fig = sp.plot.emission_atlas(("H", "He", "Na", "Hg"))
plt.show()"""),
    ("md", """## 4. Your own measured spectra

`sp.io.read_spectrum` (or `parse_spectrum` for strings) reads two-column text in nm, Å, µm, eV,
cm⁻¹ or THz. Densities per energy unit are converted with the Jacobian, so colours stay right."""),
    ("code", """energies = np.linspace(1.9, 3.0, 300)                    # photon energy, eV
led = np.exp(-0.5 * ((energies - 2.33) / 0.06) ** 2)      # a green LED, per eV
text = "\\n".join(f"{e:.4f}, {v:.6f}" for e, v in zip(energies, led))
measured = sp.io.parse_spectrum(text, unit="eV", name="green LED")
print(measured, "peak at", round(float(measured.peak_wavelength), 1), "nm")
sp.plot.spectrum(measured, xlim=(450, 650));"""),
    ("md", """## 5. Thin-film interference

Multilayers are computed with the transfer-matrix method, vectorized over wavelength, angle and
thickness."""),
    ("code", """tf = sp.thinfilm
d = np.linspace(0, 1200, 600)
soap = tf.Stack([(tf.WATER, 0)], ambient=tf.AIR, substrate=tf.AIR)
oxide = tf.Stack([(tf.SIO2, 0)], ambient=tf.AIR, substrate=tf.SI)
fig, axes = plt.subplots(2, 1, figsize=(8, 1.8), sharex=True)
sp.plot.color_strip(d, sp.color(soap.scan(0, d), brightness="global"), ax=axes[0], ticks=False,
                    label="soap film")
sp.plot.color_strip(d, sp.color(oxide.scan(0, d), brightness="global"), ax=axes[1], label="SiO₂ on Si")
axes[1].set_xlabel("thickness (nm)");"""),
    ("md", """## 6. Colour rendering: CRI and TM-30"""),
    ("code", """led = sp.illuminant("LED-B5")
print(sp.cri(led))
print(sp.tm30(led))
fig = sp.plot.tm30_report(sp.tm30(led))"""),
    ("md", """## 7. Wide-gamut figures

Spectral colours are far more saturated than sRGB allows. Render in Display P3 and save; the PNG
carries an ICC profile, so colour-managed viewers (browsers, macOS, iOS, Windows) show the
extra saturation:

```python
with sp.output_space("p3"):
    fig = sp.plot.emission_atlas()
    sp.style.save(fig, "atlas_p3.png")
```"""),
    ("md", """## Where next

* The [gallery](https://github.com/fdragowski480677/spectrapy#gallery) shows what else is in
  the box: rainbows, prisms, gratings, sky colours, star colours, AR coatings and more. Each figure
  is a short recipe in `src/spectrapy/gallery.py`.
* [`docs/API.md`](https://github.com/fdragowski480677/spectrapy/blob/main/docs/API.md) is the full
  API reference.
* `sp.interactive.blackbody_explorer()` and `sp.interactive.thinfilm_explorer()` give slider-driven
  explorers (they need an interactive matplotlib backend)."""),
]


def build(out: pathlib.Path, timeout: int = 600) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in CELLS:
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    for cell in nb.cells:  # make outputs reproducible: drop wall-clock timings and execution metadata
        cell.metadata.pop("execution", None)
        if cell.cell_type == "code" and cell.source.startswith("%%time"):
            cell.outputs = [o for o in cell.outputs if o.get("name") != "stdout"]
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "examples" / "getting_started.ipynb")
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
