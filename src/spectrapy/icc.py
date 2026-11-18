"""Minimal ICC v4 display profiles (matrix/TRC) for the built-in RGB colour spaces.

Used to tag saved images, so that colour-managed viewers (browsers, macOS, iOS, modern Windows)
show e.g. Display P3 figures with their full gamut. The profile contains: description,
copyright, media white point (D50, as v4 requires for displays), a Bradford ``chad`` matrix from
the space's white to D50, the D50-adapted primaries, and parametric tone curves.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from .adaptation import adaptation_matrix
from .colorspaces import RGBColorSpace, get_colorspace

D50_PCS = np.array([0.9642, 1.0, 0.8249])  # ICC profile connection space illuminant


def _s15f16(v: float) -> bytes:
    return struct.pack(">i", int(round(float(v) * 65536.0)))


def _xyz_tag(xyz) -> bytes:
    return b"XYZ " + bytes(4) + b"".join(_s15f16(v) for v in xyz)


def _para_tag(function_type: int, params) -> bytes:
    return b"para" + bytes(4) + struct.pack(">HH", function_type, 0) + b"".join(_s15f16(p) for p in params)


def _mluc_tag(text: str) -> bytes:
    s = text.encode("utf-16-be")
    return b"mluc" + bytes(4) + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(s), 28) + s


def _sf32_tag(matrix) -> bytes:
    return b"sf32" + bytes(4) + b"".join(_s15f16(v) for v in np.asarray(matrix).ravel())


def trc_parameters(space: RGBColorSpace):
    """ICC parametric-curve (function type, parameters) of a colour space's decoding curve."""
    key = space.name
    if key in ("sRGB", "Display P3"):
        return 3, (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)
    if key == "Rec. 2020":
        a = 1.09929682680944
        return 3, (1 / 0.45, 1 / a, (a - 1) / a, 1 / 4.5, 4.5 * 0.018053968510807)
    if key == "Adobe RGB (1998)":
        return 0, (563.0 / 256.0,)
    if key == "linear sRGB":
        return 0, (1.0,)
    raise ValueError(f"no ICC tone curve known for {space.name!r}")


def icc_profile(space="p3", description: str | None = None) -> bytes:
    """ICC v4.3 display profile (bytes) for a built-in colour space (``"p3"``, ``"srgb"``, ...)."""
    cs = get_colorspace(space)
    chad = adaptation_matrix(cs.white_xyz, D50_PCS, "bradford")
    m50 = chad @ cs.matrix_to_xyz  # D50-adapted RGB -> XYZ: columns are the colorants
    ftype, params = trc_parameters(cs)
    trc = _para_tag(ftype, params)
    tags = [
        (b"desc", _mluc_tag(description or f"{cs.name} (spectrapy)")),
        (b"cprt", _mluc_tag("No copyright, use freely")),
        (b"wtpt", _xyz_tag(D50_PCS)),
        (b"chad", _sf32_tag(chad)),
        (b"rXYZ", _xyz_tag(m50[:, 0])),
        (b"gXYZ", _xyz_tag(m50[:, 1])),
        (b"bXYZ", _xyz_tag(m50[:, 2])),
        (b"rTRC", trc),
        (b"gTRC", trc),
        (b"bTRC", trc),
    ]
    # lay out tag data (4-byte aligned); identical data (the three TRCs) is stored once
    offset = 128 + 4 + 12 * len(tags)
    table, blobs, seen = [], [], {}
    for sig, data in tags:
        if data in seen:
            table.append((sig, *seen[data]))
            continue
        pad = (-len(data)) % 4
        seen[data] = (offset, len(data))
        table.append((sig, offset, len(data)))
        blobs.append(data + bytes(pad))
        offset += len(data) + pad
    body = struct.pack(">I", len(tags)) + b"".join(struct.pack(">4sII", s, o, n) for s, o, n in table) + b"".join(blobs)
    size = 128 + len(body)
    header = bytearray(128)
    struct.pack_into(">I", header, 0, size)
    struct.pack_into(">I", header, 8, 0x04300000)          # version 4.3
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    struct.pack_into(">6H", header, 24, 2026, 1, 1, 0, 0, 0)
    header[36:40] = b"acsp"
    header[68:80] = b"".join(_s15f16(v) for v in D50_PCS)  # PCS illuminant
    header[80:84] = b"sppy"                                 # creator
    profile = bytes(header) + body
    digest = hashlib.md5(profile).digest()  # flags, intent and ID fields are zero at this point
    return profile[:84] + digest + profile[100:]


__all__ = ["icc_profile", "trc_parameters"]
