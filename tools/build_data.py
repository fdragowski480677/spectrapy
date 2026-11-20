"""Regenerate the bundled data tables in ``src/spectrapy/data``.

Sources
-------
* CIE 1931 2 deg and CIE 1964 10 deg colour-matching functions, CIE illuminants
  A and D65 (1 nm): Colour & Vision Research Laboratory (CVRL), http://www.cvrl.org
* CIE daylight basis functions S0/S1/S2, fluorescent FL1-FL12, LED illuminants
  (CIE 015:2018), the CRI test colour samples TCS01-TCS14 (CIE 13.3-1995) and the
  99 colour evaluation samples of CIE 224:2017 / ANSI/IES TM-30: taken from the
  ``colour-science`` package (BSD-3-Clause), which reproduces the CIE tables.

Usage::

    python tools/build_data.py                # downloads the CVRL files
    python tools/build_data.py --cvrl DIR     # use already downloaded CVRL csv files
"""
from __future__ import annotations

import argparse
import gzip
import io
import pathlib
import urllib.request

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "src" / "spectrapy" / "data"
CVRL = {
    "ciexyz31_1.csv": "http://www.cvrl.org/database/data/cmfs/ciexyz31_1.csv",
    "ciexyz64_1.csv": "http://www.cvrl.org/database/data/cmfs/ciexyz64_1.csv",
    "Illuminantd65.csv": "http://www.cvrl.org/database/data/cie/Illuminantd65.csv",
    "Illuminanta.csv": "http://www.cvrl.org/database/data/cie/Illuminanta.csv",
}


def read_cvrl(name: str, cvrl_dir: pathlib.Path | None) -> np.ndarray:
    if cvrl_dir is not None:
        text = (cvrl_dir / name).read_text()
    else:
        with urllib.request.urlopen(CVRL[name], timeout=60) as r:  # noqa: S310
            text = r.read().decode()
    return np.loadtxt(io.StringIO(text), delimiter=",")


def write(name: str, header: list[str], columns: list[str], wl, values, fmt="%.9g"):
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[0] != len(wl):
        values = values.T
    lines = [f"# {h}" for h in header]
    lines.append(",".join(["wavelength_nm", *columns]))
    for w, row in zip(wl, values):
        lines.append(",".join([f"{w:g}", *(fmt % v for v in row)]))
    (OUT / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {name}: {len(wl)} rows x {len(columns)} columns")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvrl", type=pathlib.Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cmf31 = read_cvrl("ciexyz31_1.csv", args.cvrl)
    write("cmf_cie1931_2.csv",
          ["CIE 1931 2 degree standard observer colour-matching functions, 1 nm",
           "Source: CVRL, http://www.cvrl.org/database/data/cmfs/ciexyz31_1.csv"],
          ["xbar", "ybar", "zbar"], cmf31[:, 0], cmf31[:, 1:])
    cmf64 = read_cvrl("ciexyz64_1.csv", args.cvrl)
    write("cmf_cie1964_10.csv",
          ["CIE 1964 10 degree standard observer colour-matching functions, 1 nm",
           "Source: CVRL, http://www.cvrl.org/database/data/cmfs/ciexyz64_1.csv"],
          ["xbar", "ybar", "zbar"], cmf64[:, 0], cmf64[:, 1:])
    d65 = read_cvrl("Illuminantd65.csv", args.cvrl)
    a = read_cvrl("Illuminanta.csv", args.cvrl)
    assert np.array_equal(d65[:, 0], a[:, 0])
    write("illuminants_cie_1nm.csv",
          ["CIE standard illuminants A and D65, relative spectral power, 1 nm",
           "Source: CVRL, http://www.cvrl.org/database/data/cie/"],
          ["A", "D65"], d65[:, 0], np.column_stack([a[:, 1], d65[:, 1]]))

    import colour  # only needed when rebuilding the data
    import colour.colorimetry.datasets.illuminants as ill
    from colour.quality.datasets import tcs

    basis = ill.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES
    wl = basis["S0"].wavelengths
    write("daylight_basis.csv",
          ["CIE daylight (D-series) basis functions S0, S1, S2, 5 nm (CIE 015:2004 Table T.2)",
           "Source: colour-science (BSD-3-Clause) reproduction of the CIE table"],
          ["S0", "S1", "S2"], wl, np.column_stack([basis[k].values for k in ("S0", "S1", "S2")]))

    sds = colour.SDS_ILLUMINANTS
    fl = [f"FL{i}" for i in range(1, 13)]
    write("illuminants_fl.csv",
          ["CIE fluorescent illuminants FL1-FL12, relative spectral power, 5 nm (CIE 015:2004)",
           "Source: colour-science (BSD-3-Clause) reproduction of the CIE table"],
          fl, sds["FL1"].wavelengths, np.column_stack([sds[k].values for k in fl]))
    led = ["LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5", "LED-BH1", "LED-RGB1", "LED-V1", "LED-V2"]
    write("illuminants_led.csv",
          ["CIE LED illuminants, relative spectral power, 5 nm (CIE 015:2018)",
           "Source: colour-science (BSD-3-Clause) reproduction of the CIE table"],
          led, sds["LED-B1"].wavelengths, np.column_stack([sds[k].values for k in led]))

    t = tcs.SDS_TCS_CIE1995
    names = [f"TCS{i:02d}" for i in range(1, 15)]
    write("cri_tcs.csv",
          ["CIE 13.3-1995 test colour samples TCS01-TCS14, spectral reflectance, 5 nm",
           "Source: colour-science (BSD-3-Clause) reproduction of the CIE table"],
          names, t["TCS01"].wavelengths, np.column_stack([t[k].values for k in names]))

    from colour import SpectralShape
    from colour.quality.cfi2017 import load_TCS_CIE2017

    ces = load_TCS_CIE2017(SpectralShape(380, 780, 1))
    write("tm30_ces.csv",
          ["CIE 224:2017 / ANSI/IES TM-30 colour evaluation samples CES01-CES99, spectral reflectance, 1 nm",
           "Source: colour-science (BSD-3-Clause) reproduction of the CIE 224:2017 data"],
          [f"CES{i:02d}" for i in range(1, 100)], ces.wavelengths, ces.values, fmt="%.6g")
    raw = (OUT / "tm30_ces.csv").read_bytes()
    (OUT / "tm30_ces.csv.gz").write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    (OUT / "tm30_ces.csv").unlink()
    print("compressed tm30_ces.csv -> tm30_ces.csv.gz")


if __name__ == "__main__":
    main()
